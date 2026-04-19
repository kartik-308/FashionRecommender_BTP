"""
Core recommendation logic — BTP demo version (self-contained)

Architecture overview:
  1. Images from multiple fashion datasets are encoded into 512-d vectors using CLIP
     (openai/clip-vit-base-patch32).
  2. All vectors are stored in a unified FAISS flat inner-product index plus per-source
     sub-indexes for stratified retrieval.
  3. At query time a fused vector (query + EMA preference) is searched against the
     per-source sub-indexes so every source gets proportional representation in the
     candidate pool.
  4. Candidates are scored with a three-term objective:
       final_score = W1*norm_sim_query + W2*norm_sim_pref
                     - W3*redundancy   - BETA*rejection_penalty
     Scores are normalised per-source before combining so no single dataset dominates.
  5. The top-K results are selected with a proportional-diversity pass that guarantees
     at least one result per source before filling remaining slots by score.

All cache files (embeddings .npy + metadata .parquet) are written to ./data/ so that
subsequent runs skip re-encoding.  Image directories still point to the original archive
— update _BASE if you move the image folders.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
# Directory containing this script — used to anchor all relative paths.
_SELF = os.path.dirname(os.path.abspath(__file__))

# All cached embeddings (.npy) and metadata (.parquet / .csv) are stored here
# so they survive Python re-imports without re-encoding.
_DATA = os.path.join(_SELF, "data")

# Root of the image dataset folders on disk.
# ⚠ Update this path if the archive is moved to a different location.
_BASE = r"C:\Users\Kartikeya Singh\OneDrive - LNMIIT\Desktop\archive"


# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry describes one image collection: where images live, where optional
# annotation files live, an optional CSV index, a human-readable source label,
# and the cache directory for pre-computed embeddings.
DATASET_CONFIGS = {
    "df2_test": {
        "label":     "DeepFashion2 Test",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "test", "test", "image"),
        "annos_dir": None,          # No per-image JSON annotations for the test split
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": _DATA,
    },
    "df2_train": {
        "label":     "DeepFashion2 Train",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "train", "image"),
        # Training split ships with per-image JSON annotation files (category, keypoints, etc.)
        "annos_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "train", "annos"),
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": _DATA,
    },
    "df2_val": {
        "label":     "DeepFashion2 Validation",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "validation", "image"),
        "annos_dir": None,
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": _DATA,
    },
    "ebay_small": {
        "label":     "eBay Fashion",
        "image_dir": os.path.join(_BASE, "ebayDataset", "ebay_fashion_dataset", "images"),
        "annos_dir": None,
        # Metadata (category, title, price, …) comes from a flat CSV index
        "csv":       os.path.join(_BASE, "ebayDataset", "ebay_fashion_dataset", "dataset_index.csv"),
        "source":    "eBay",
        "cache_dir": _DATA,
    },
    "ebay_large": {
        "label":     "Amazon Fashion",
        "image_dir": os.path.join(_BASE, "ebay_fashion_dataset", "images"),
        "annos_dir": None,
        "csv":       os.path.join(_BASE, "ebay_fashion_dataset", "dataset_index.csv"),
        "source":    "Amazon",
        "cache_dir": _DATA,
    },
}


# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE  = 64     # Images per CLIP forward pass (tune to GPU VRAM)
TOP_K       = 5      # Final recommendations returned to the user
ALPHA       = 0.7    # EMA decay for the preference vector (higher = more inertia)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Scoring weights for the final ranking objective
W1_QUERY    = 0.55   # Weight for query-similarity term
W2_PREF     = 0.30   # Weight for user-preference-similarity term
W3_REDUND   = 0.15   # Penalty weight for redundancy (similarity to already-shown items)

CANDIDATE_K = 200    # Number of ANN candidates retrieved per source before reranking
BETA        = 0.40   # Hard penalty multiplier applied to rejected items


# ── Utility helpers ───────────────────────────────────────────────────────────

def to_tensor(out):
    """
    Extract a 2-D float tensor from a HuggingFace model output object.

    CLIP's get_image_features / get_text_features returns a BaseModelOutput
    whose exact attribute name varies by model version.  This function tries
    the common attribute names in order and falls back to a full attribute scan
    so the rest of the code stays model-agnostic.

    Args:
        out: A HuggingFace model output (BaseModelOutput or similar).

    Returns:
        torch.Tensor of shape (batch, hidden_dim).

    Raises:
        ValueError: If no tensor attribute is found.
    """
    # Direct tensor — nothing to unwrap
    if isinstance(out, torch.Tensor):
        return out

    # Try well-known attribute names in order of preference
    for attr in ["pooler_output", "last_hidden_state"]:
        if hasattr(out, attr):
            val = getattr(out, attr)
            # last_hidden_state is (batch, seq_len, dim) — take the [CLS] token
            return val[:, 0, :] if attr == "last_hidden_state" else val

    # Last resort: return the first tensor attribute found
    for val in vars(out).values():
        if isinstance(val, torch.Tensor):
            return val

    raise ValueError(f"Cannot extract tensor from {type(out)}")


def _cache_paths(key):
    """
    Return the three canonical cache file paths for a dataset key:
      (embeddings .npy,  metadata .parquet,  metadata .csv)

    The .parquet file is the preferred metadata format (faster I/O, typed
    columns).  The .csv is written as a human-readable backup and as a
    migration source when only the old CSV cache exists.
    """
    d = DATASET_CONFIGS[key]["cache_dir"]
    return (
        os.path.join(d, f"cache_{key}_embeddings.npy"),
        os.path.join(d, f"cache_{key}_metadata.parquet"),
        os.path.join(d, f"cache_{key}_metadata.csv"),
    )


def _read_meta(pq_path, csv_path):
    """
    Load dataset metadata, preferring parquet for speed.

    If only the legacy CSV cache exists it is read and immediately migrated to
    parquet so subsequent loads are faster.  Both UTF-8 and latin-1 encodings
    are handled because some eBay product titles contain non-ASCII characters.

    Args:
        pq_path  (str): Path to the parquet file.
        csv_path (str): Path to the CSV fallback.

    Returns:
        pd.DataFrame with string columns, NaNs replaced by empty strings.
    """
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)

    # Load CSV and silently migrate to parquet for future runs
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1").fillna("")

    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass  # Migration failure is non-fatal; CSV will be used next time too

    return df


# ── Per-dataset loader ────────────────────────────────────────────────────────

def _load_one_dataset(key, model, processor):
    """
    Load (or build) embeddings and metadata for a single dataset.

    Cache hit path  — fast:
        Reads embeddings.npy and metadata.parquet/csv from disk if both exist
        and their row counts match.

    Cache miss path — slow:
        1. Discovers all .jpg/.jpeg/.png files in image_dir.
        2. Enriches a base DataFrame with category/title/price metadata sourced
           from either per-image JSON annotation files (DeepFashion2 train) or
           a flat CSV index (eBay / Amazon datasets).
        3. Encodes all images through CLIP in batches of BATCH_SIZE.
        4. Saves embeddings and metadata to disk.

    Args:
        key       (str):           Key into DATASET_CONFIGS.
        model     (CLIPModel):     Pre-loaded CLIP model on DEVICE.
        processor (CLIPProcessor): Matching CLIP processor.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]:
            embeddings — float32 array of shape (N, 512), L2-normalised.
            df         — metadata DataFrame with columns: filename, full_path,
                         dataset, source, category, title, price (+ any extras
                         from the CSV index).

    Raises:
        FileNotFoundError: If image_dir does not exist on disk.
    """
    cfg     = DATASET_CONFIGS[key]
    img_dir = cfg["image_dir"]
    emb_path, pq_path, csv_path = _cache_paths(key)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    # Collect image filenames in deterministic order
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # ── Try loading from cache ────────────────────────────────────────────
    if os.path.exists(emb_path) and (os.path.exists(pq_path) or os.path.exists(csv_path)):
        embeddings = np.load(emb_path)
        df_cached  = _read_meta(pq_path, csv_path)

        # Validate cache integrity: row counts must match
        if len(embeddings) == len(df_cached):
            # Coerce popularity to numeric (may have been stored as string)
            if "popularity" in df_cached.columns:
                df_cached["popularity"] = pd.to_numeric(
                    df_cached["popularity"], errors="coerce").fillna(0.0)

            # Re-attach runtime columns that aren't persisted
            df_cached["dataset"]   = key
            df_cached["source"]    = cfg["source"]
            df_cached["full_path"] = df_cached["filename"].apply(
                lambda f: os.path.join(img_dir, f))

            print(f"  [{key}] {len(embeddings):,} loaded")
            return embeddings, df_cached

    # ── Build fresh DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        "filename":  files,
        "full_path": [os.path.join(img_dir, f) for f in files],
        "dataset":   key,
        "source":    cfg["source"],
    })

    annos_dir = cfg.get("annos_dir")
    csv_src   = cfg.get("csv")

    if annos_dir and os.path.exists(annos_dir):
        # ── DeepFashion2 train: parse per-image JSON annotation files ─────
        # Each JSON has numbered keys ("1", "2", …) mapping to item dicts
        # that contain "category_name".  We take the first item found.
        cats = {}
        for fname in df["filename"]:
            ann_file = os.path.join(annos_dir, os.path.splitext(fname)[0] + ".json")
            if os.path.exists(ann_file):
                try:
                    with open(ann_file) as f:
                        ann = json.load(f)
                    for v in ann.values():
                        if isinstance(v, dict) and "category_name" in v:
                            cats[fname] = v["category_name"]; break
                except Exception:
                    pass  # Malformed annotation — leave category as "unknown"

        df["category"] = df["filename"].map(cats).fillna("unknown")
        df["title"] = ""; df["price"] = ""

    elif csv_src and os.path.exists(csv_src):
        # ── eBay / Amazon: merge flat CSV index on filename ───────────────
        meta = pd.read_csv(csv_src, dtype=str).fillna("")

        # Numeric IDs in the CSV need zero-padding to match "<id>.jpg" filenames
        if "id" in meta.columns:
            meta["filename"] = meta["id"].str.zfill(6) + ".jpg"

        df = df.merge(meta, on="filename", how="left")

        # Accept either "category" or "category_name" column from the CSV
        cat_col = next((c for c in ["category","category_name"] if c in df.columns), None)
        df["category"] = df[cat_col].fillna("unknown") if cat_col else "unknown"
        df["title"]    = df["title"].fillna("") if "title" in df.columns else ""
        df["price"]    = df["price"].fillna("") if "price" in df.columns else ""

        # Merge may create duplicate source columns (source_x / source_y) — clean up
        if "source_x" in df.columns:
            df["source"] = cfg["source"]
            df.drop(columns=["source_x","source_y"], errors="ignore", inplace=True)
    else:
        # Fallback when no annotation/CSV source is available
        df["category"] = "unknown"; df["title"] = ""; df["price"] = ""

    # ── Encode all images through CLIP ────────────────────────────────────
    print(f"  [{key}] encoding {len(df):,} images ...")
    all_embeds = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df["full_path"].iloc[i:i+BATCH_SIZE].tolist()

        # Open images; substitute a grey placeholder for any corrupt file
        imgs = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (180, 180, 180)))

        # Tokenise and move pixel values to GPU/CPU
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        pv     = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            out   = model.get_image_features(pixel_values=pv)
            feats = to_tensor(out)
            # L2-normalise so inner-product equals cosine similarity
            feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embeds.append(feats.cpu().numpy())

    embeddings = np.vstack(all_embeds).astype("float32")

    # Persist to disk for future runs
    np.save(emb_path, embeddings)
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"  [{key}] cached {embeddings.shape}")
    return embeddings, df


# ── Unified loader (parallel) ─────────────────────────────────────────────────

def load_all_datasets(model, processor):
    """
    Load all configured datasets in parallel and merge into a single index.

    Uses a ThreadPoolExecutor so that cache reads and CLIP encoding for
    different datasets can overlap (I/O-bound operations benefit even without
    the GIL being released).  Datasets that fail to load are skipped with a
    warning rather than crashing the whole process.

    Post-merge steps:
      - String columns are coerced to str and NaNs filled with "".
      - The unified embedding matrix is L2-re-normalised (float32).

    Args:
        model     (CLIPModel):     Pre-loaded CLIP model.
        processor (CLIPProcessor): Matching CLIP processor.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]:
            emb_matrix — float32 array (N_total, 512), row-normalised.
            merged     — metadata DataFrame for all N_total images with
                         consistent columns across datasets.

    Raises:
        RuntimeError: If every dataset fails to load.
    """
    results = {}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_load_one_dataset, k, model, processor): k
                   for k in DATASET_CONFIGS}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                results[k] = fut.result()
            except Exception as e:
                print(f"  [{k}] skipped: {e}")

    if not results:
        raise RuntimeError("No datasets loaded.")

    # Concatenate in config-definition order so FAISS index row i matches df row i
    all_dfs, all_embs = [], []
    for k in DATASET_CONFIGS:
        if k in results:
            emb, df = results[k]
            all_dfs.append(df)
            all_embs.append(emb)

    merged = pd.concat(all_dfs, ignore_index=True)

    # Ensure consistent dtypes across datasets
    for col in ["title", "price", "source", "dataset"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
    if "category" in merged.columns:
        merged["category"] = merged["category"].fillna("unknown").astype(str)
        merged["category"] = merged["category"].replace("", "unknown")

    # Stack and re-normalise the combined embedding matrix
    emb_matrix = np.vstack(all_embs).astype("float32")
    norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_matrix = (emb_matrix / norms).astype("float32")

    print(f"\n[OK] Unified index: {len(merged):,} images")
    print(merged["dataset"].value_counts().to_string())
    return emb_matrix, merged


# ── CLIP model helpers ────────────────────────────────────────────────────────

def load_clip():
    """
    Download (or load from HuggingFace cache) and return a CLIP model + processor.

    Uses the ViT-B/32 checkpoint which balances quality and inference speed.
    The model is set to eval mode so dropout/BN behave correctly at inference.

    Returns:
        Tuple[CLIPModel, CLIPProcessor]
    """
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def build_faiss_index(embeddings):
    """
    Build a brute-force FAISS inner-product (cosine) index from L2-normalised embeddings.

    IndexFlatIP is exact (no approximation) — suitable for datasets up to a few
    hundred thousand vectors.  For larger collections consider IndexIVFFlat or HNSW.

    Args:
        embeddings (np.ndarray): float32 array of shape (N, dim), L2-normalised.

    Returns:
        faiss.IndexFlatIP: Populated FAISS index ready for search.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_per_source_indexes(embeddings, df):
    """
    Build a dedicated FAISS sub-index for each unique source in the dataset.

    Separate sub-indexes enable stratified (per-source) retrieval so that
    dominant datasets don't crowd out smaller ones in the candidate pool.

    Args:
        embeddings (np.ndarray): Unified float32 embedding matrix (N_total, dim).
        df         (pd.DataFrame): Metadata DataFrame aligned with embeddings.

    Returns:
        Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]]:
            Keys are source names.
            Values are (sub_index, global_row_indices) tuples where
            global_row_indices maps a sub-index row back to a row in `embeddings`/`df`.
    """
    indexes = {}
    for src in df["source"].unique():
        mask        = (df["source"] == src).values
        global_idxs = np.where(mask)[0]           # Row indices in the unified matrix
        sub         = embeddings[mask].astype("float32")
        dim         = sub.shape[1]
        ix          = faiss.IndexFlatIP(dim)
        ix.add(sub)
        indexes[src] = (ix, global_idxs)
        print(f"  Sub-index [{src}]: {len(global_idxs):,} vectors")
    return indexes


# ── Query encoding ────────────────────────────────────────────────────────────

def encode_query(model, processor, text=None, image_path=None,
                 text_weight=0.5, image_weight=0.5):
    """
    Encode a text query, an image query, or a weighted combination of both.

    Text and image embeddings are individually L2-normalised before combining
    so that the weighted sum lives in the same unit-sphere space as the
    image database embeddings.

    Args:
        model        (CLIPModel):     Pre-loaded CLIP model.
        processor    (CLIPProcessor): Matching CLIP processor.
        text         (str | None):    Free-text search string.
        image_path   (str | None):    Path to a query image on disk.
        text_weight  (float):         Relative weight of the text embedding
                                      when both modalities are provided.
        image_weight (float):         Relative weight of the image embedding.

    Returns:
        Tuple[np.ndarray, str]:
            query_vec  — float32 L2-normalised embedding of shape (512,).
            query_mode — human-readable string describing the modality used.

    Raises:
        ValueError: If neither text nor a valid image_path is provided.
    """
    tv = iv = None  # Text and image embedding accumulators

    # ── Text branch ───────────────────────────────────────────────────────
    if text:
        inp = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inp["input_ids"].to(DEVICE),
                attention_mask=inp["attention_mask"].to(DEVICE))
            f  = to_tensor(out)
            f  = f / f.norm(dim=-1, keepdim=True)   # Normalise to unit sphere
        tv = f.cpu().numpy()[0].astype("float32")

    # ── Image branch ──────────────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        inp = processor(images=[img], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=inp["pixel_values"].to(DEVICE))
            f   = to_tensor(out)
            f   = f / f.norm(dim=-1, keepdim=True)
        iv = f.cpu().numpy()[0].astype("float32")

    # ── Combine modalities ────────────────────────────────────────────────
    if tv is not None and iv is not None:
        # Weighted sum then re-normalise
        combined = text_weight * tv + image_weight * iv
        return combined / np.linalg.norm(combined), f"text+image ({int(text_weight*100)}/{int(image_weight*100)})"
    if tv is not None:
        return tv, "text only"
    if iv is not None:
        return iv, "image only"

    raise ValueError("Provide text or image_path")


# ── Preference tracker (EMA + reject) ─────────────────────────────────────────

class PreferenceTracker:
    """
    Maintains a running user-preference embedding using Exponential Moving Average
    (EMA) and tracks items the user has explicitly rejected.

    Preference vector:
        pref_vec_t = α * pref_vec_{t-1} + (1-α) * new_vec
        where new_vec is the L2-normalised embedding of an item the user liked.
        A high α (close to 1) gives more weight to historical preferences;
        a low α adapts quickly to the most recent interaction.

    Rejected items:
        Filenames in `rejected` receive a hard score penalty (BETA) during
        reranking so they are pushed to the bottom of results.

    Shown items:
        Embeddings of previously shown items are stored in `shown` to compute
        a redundancy penalty that discourages re-surfacing near-duplicates.
    """

    def __init__(self, alpha=ALPHA):
        """
        Args:
            alpha (float): EMA decay factor in [0, 1].  Default: ALPHA (0.7).
        """
        self.alpha    = alpha
        self.pref_vec = None   # Will be initialised on the first update() call
        self.rejected = set()  # Set of rejected filenames
        self.shown    = []     # List of L2-normalised embeddings already shown

    def update(self, vec):
        """
        Incorporate a new liked embedding into the running preference vector.

        Args:
            vec (np.ndarray): 1-D float32 embedding, need not be normalised.
        """
        vec = vec / (np.linalg.norm(vec) + 1e-8)  # Defensive normalisation
        if self.pref_vec is None:
            # Cold-start: first liked item becomes the initial preference
            self.pref_vec = vec.copy()
        else:
            # EMA update
            self.pref_vec = self.alpha * self.pref_vec + (1 - self.alpha) * vec
            self.pref_vec /= (np.linalg.norm(self.pref_vec) + 1e-8)

    def add_shown(self, emb):
        """
        Record an embedding as having been shown to the user.
        Used to compute the redundancy penalty in reranking.

        Args:
            emb (np.ndarray): 1-D float32 embedding of a shown item.
        """
        self.shown.append(emb / (np.linalg.norm(emb) + 1e-8))

    def reject(self, filename):
        """
        Mark a filename as rejected so it is suppressed in future results.

        Args:
            filename (str): The filename key as stored in the metadata DataFrame.
        """
        self.rejected.add(filename)

    def get(self):
        """
        Return the current preference vector, or None if no likes have been recorded.

        Returns:
            np.ndarray | None
        """
        return self.pref_vec

    def redundancy_score(self, v):
        """
        Compute the maximum cosine similarity between `v` and all shown items.
        A high score indicates the item is very similar to something already shown.

        Args:
            v (np.ndarray): 1-D float32 embedding of a candidate item.

        Returns:
            float: Maximum cosine similarity in [0, 1], or 0.0 if nothing shown yet.
        """
        if not self.shown:
            return 0.0
        v = v / (np.linalg.norm(v) + 1e-8)
        return max(float(np.dot(v, s)) for s in self.shown)

    def rejection_penalty(self, f):
        """
        Return 1.0 if the filename has been explicitly rejected, 0.0 otherwise.

        Args:
            f (str): Filename to check.

        Returns:
            float: 1.0 (penalise) or 0.0 (no penalty).
        """
        return 1.0 if f in self.rejected else 0.0


# ── Reranker with per-source normalization & diversity ─────────────────────────

def rerank(candidates_df, candidate_indices, all_embeddings, query_vec, tracker):
    """
    Score a candidate pool and return the top-K most relevant, diverse results.

    Scoring pipeline:
      1. Compute raw query-similarity (sq) and preference-similarity (sp) for
         each candidate.
      2. Normalise sq and sp independently within each source so that datasets
         with naturally lower CLIP similarities are not systematically ranked lower.
      3. Combine into a final scalar:
             final = W1*norm_sq + W2*norm_sp - W3*redundancy - BETA*rej_penalty
      4. Sort by final score descending.
      5. Select top-K via a proportional-diversity pass: guarantee at least
         floor(TOP_K / n_sources) results per source, then fill remaining slots
         by score.

    Args:
        candidates_df      (pd.DataFrame): Metadata rows for the candidate pool.
        candidate_indices  (np.ndarray):   Corresponding row indices into all_embeddings.
        all_embeddings     (np.ndarray):   Full unified embedding matrix.
        query_vec          (np.ndarray):   L2-normalised query embedding.
        tracker            (PreferenceTracker): Current session preference state.

    Returns:
        pd.DataFrame: Top-K results with columns:
            filename, full_path, category, title, price, source, dataset,
            sim_query, sim_pref, redundancy, final_score.
            Empty DataFrame if all candidates were rejected.
    """
    pref_vec = tracker.get()
    rows_out = []

    # ── Compute raw similarity scores ─────────────────────────────────────
    for idx, row in zip(candidate_indices, candidates_df.itertuples()):
        iv  = all_embeddings[idx]
        iv  = iv / (np.linalg.norm(iv) + 1e-8)

        sq  = float(np.dot(query_vec, iv))                                  # Query cosine sim
        sp  = float(np.dot(pref_vec, iv)) if pref_vec is not None else 0.0  # Pref cosine sim
        rd  = tracker.redundancy_score(iv)                                  # Max sim to shown items
        rp  = tracker.rejection_penalty(row.filename)                       # 1 if rejected

        rows_out.append({
            "filename":    row.filename,
            "full_path":   row.full_path,
            "category":    str(getattr(row, "category", "---")),
            "title":       str(getattr(row, "title", "")),
            "price":       str(getattr(row, "price", "")),
            "source":      str(getattr(row, "source", "")),
            "dataset":     str(getattr(row, "dataset", "")),
            "sim_query":   round(sq, 4),
            "sim_pref":    round(sp, 4),
            "redundancy":  round(rd, 4),
            "rej_penalty": round(rp, 4),
        })

    res = pd.DataFrame(rows_out)

    # Hard-remove rejected items before scoring (avoids wasting slots)
    res = res[~res["filename"].isin(tracker.rejected)].copy()
    if res.empty:
        return res.reset_index(drop=True)

    # ── Per-source min-max normalisation ──────────────────────────────────
    # Ensures that a source with uniformly lower raw similarities still
    # contributes candidates at the same normalised scale as richer sources.
    res["norm_sim_query"] = 0.0
    res["norm_sim_pref"]  = 0.0

    for src in res["source"].unique():
        mask = res["source"] == src

        # Normalise query similarity within this source
        sq_vals = res.loc[mask, "sim_query"]
        sq_min, sq_max = sq_vals.min(), sq_vals.max()
        rng = sq_max - sq_min
        if rng > 1e-8:
            res.loc[mask, "norm_sim_query"] = (sq_vals - sq_min) / rng
        else:
            res.loc[mask, "norm_sim_query"] = 1.0  # All identical — treat as perfect match

        # Normalise preference similarity within this source
        sp_vals = res.loc[mask, "sim_pref"]
        sp_min, sp_max = sp_vals.min(), sp_vals.max()
        rng_p = sp_max - sp_min
        if rng_p > 1e-8:
            res.loc[mask, "norm_sim_pref"] = (sp_vals - sp_min) / rng_p
        else:
            res.loc[mask, "norm_sim_pref"] = 0.5  # No spread — neutral score

    # ── Final scoring ─────────────────────────────────────────────────────
    res["final_score"] = (
          W1_QUERY   * res["norm_sim_query"]
        + W2_PREF    * res["norm_sim_pref"]
        - W3_REDUND  * res["redundancy"]
        - BETA       * res["rej_penalty"]
    ).round(4)

    res = res.sort_values("final_score", ascending=False)

    # ── Proportional diversity selection ──────────────────────────────────
    # Guarantee at least base_per_src results from every source, then fill
    # remaining slots greedily by final_score.
    sources     = res["source"].unique().tolist()
    n_src       = len(sources)

    if n_src == 0:
        return res.head(TOP_K).reset_index(drop=True)

    base_per_src = max(1, TOP_K // n_src)  # Minimum guaranteed slots per source
    selected, used = [], set()

    # First pass: fill guaranteed quota for each source
    for src in sources:
        src_rows = res[res["source"] == src]
        count = 0
        for idx2, row in src_rows.iterrows():
            if count >= base_per_src:
                break
            if row["filename"] not in used:
                selected.append(idx2)
                used.add(row["filename"])
                count += 1

    # Second pass: fill remaining slots by descending final_score (any source)
    for idx2, row in res.iterrows():
        if len(selected) >= TOP_K:
            break
        if row["filename"] not in used:
            selected.append(idx2)
            used.add(row["filename"])

    # Return selected rows sorted by final_score; drop internal normalisation columns
    out = res.loc[selected].sort_values("final_score", ascending=False).reset_index(drop=True)
    out.drop(columns=["norm_sim_query", "norm_sim_pref", "rej_penalty"],
             inplace=True, errors="ignore")
    return out


def retrieve_and_rerank(query_vec, index, df, all_embeddings, tracker,
                        per_source_indexes=None):
    """
    End-to-end retrieval pipeline: update preferences → ANN search → rerank → log shown.

    Step 1 — Preference update:
        The tracker EMA is updated with the query vector so that repeated queries
        gradually shift the preference towards the user's area of interest.

    Step 2 — Fused query vector:
        A 60/40 blend of the raw query and the current preference vector is used
        for ANN search.  This biases retrieval towards the user's historical taste
        without completely ignoring the current query.

    Step 3 — Stratified ANN retrieval:
        If per_source_indexes is provided, CANDIDATE_K nearest neighbours are
        fetched independently from each source sub-index, then merged.  This
        guarantees every source contributes to the candidate pool regardless of
        its size.  Falls back to a single global index search if no sub-indexes
        are available.

    Step 4 — Reranking:
        The merged candidate pool is scored and filtered by rerank().

    Step 5 — Shown-item logging:
        Embeddings of the returned results are registered in the tracker so
        future queries penalise redundant recommendations.

    Args:
        query_vec           (np.ndarray):         L2-normalised query embedding (512,).
        index               (faiss.IndexFlatIP):  Global FAISS index (used as fallback).
        df                  (pd.DataFrame):       Unified metadata DataFrame.
        all_embeddings      (np.ndarray):         Unified embedding matrix.
        tracker             (PreferenceTracker):  Current session state.
        per_source_indexes  (dict | None):        Per-source sub-indexes from
                                                  build_per_source_indexes().

    Returns:
        pd.DataFrame: Top-K recommendations (may be empty if all candidates are
                      filtered out by rejection penalties).
    """
    # Update the EMA preference with the current query intent
    tracker.update(query_vec)
    pref = tracker.get()

    # Fuse query and preference for richer ANN retrieval
    fused = 0.6 * query_vec + 0.4 * pref if pref is not None else query_vec
    fused = (fused / np.linalg.norm(fused)).reshape(1, -1).astype("float32")

    # ── Candidate retrieval ───────────────────────────────────────────────
    if per_source_indexes:
        # Stratified: gather CANDIDATE_K candidates per source sub-index
        all_cand_idxs = []
        for src, (sub_ix, global_idxs) in per_source_indexes.items():
            k = min(CANDIDATE_K, sub_ix.ntotal)   # Guard against tiny sub-indexes
            if k == 0:
                continue
            _, local_idxs = sub_ix.search(fused, k)
            # Map local sub-index row numbers back to global matrix indices
            mapped = global_idxs[local_idxs[0]]
            all_cand_idxs.append(mapped)

        # Deduplicate in case the same item appears in multiple sub-indexes
        cand_idxs = (np.unique(np.concatenate(all_cand_idxs))
                     if all_cand_idxs else np.array([], dtype=int))
    else:
        # Fallback: single global ANN search
        _, idxs   = index.search(fused, CANDIDATE_K)
        cand_idxs = idxs[0]

    if len(cand_idxs) == 0:
        return pd.DataFrame()

    # Slice the metadata DataFrame to the candidate rows
    candidates_df = df.iloc[cand_idxs].reset_index(drop=True)

    # Score and select top-K with diversity
    results = rerank(candidates_df, cand_idxs, all_embeddings, query_vec, tracker)

    # Register returned items so future queries can penalise re-showing them
    for _, row in results.iterrows():
        m = df[df["filename"] == row["filename"]].index
        if len(m):
            tracker.add_shown(all_embeddings[m[0]])

    return results