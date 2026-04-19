"""
Flask backend — BTP demo (no auth, no admin, no loader)
"""
import os
import uuid
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from PIL import Image
import recommender as rec

# ── App Initialization ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "btp-demo-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Global Model State ────────────────────────────────────────────────────────
_index              = None
_df                 = None
_embeddings         = None
_model              = None
_processor          = None
_ready              = False
_error              = None
_total_images       = 0
_per_source_indexes = None

# ── In-memory preference tracker (resets on restart) ──────────────────────────
_tracker = None


def init_models():
    global _model, _processor, _index, _df, _embeddings, _ready, _error
    global _total_images, _per_source_indexes, _tracker
    try:
        print("Loading CLIP...")
        _model, _processor = rec.load_clip()
        print(f"  CLIP ready on {rec.DEVICE.upper()}")

        _embeddings, _df = rec.load_all_datasets(_model, _processor)
        _index = rec.build_faiss_index(_embeddings)
        _per_source_indexes = rec.build_per_source_indexes(_embeddings, _df)
        _total_images = len(_df)

        _tracker = rec.PreferenceTracker()
        _ready = True
        print(f"Unified FAISS index: {_total_images:,} images")
    except Exception as e:
        import traceback
        _error = str(e)
        print(f"Init failed: {e}")
        traceback.print_exc()


# ── Utility Functions ─────────────────────────────────────────────────────────
def img_to_b64(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((400, 400))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    if not _ready:
        return jsonify({"error": "Model not ready yet."}), 503

    text         = request.form.get("text", "").strip() or None
    image_file   = request.files.get("image")
    text_weight  = float(request.form.get("text_weight", 0.5))
    image_weight = 1.0 - text_weight

    # ── Debug: trace upload ───────────────────────────────────────────────
    print(f"[DEBUG] form keys  = {list(request.form.keys())}")
    print(f"[DEBUG] files keys = {list(request.files.keys())}")
    print(f"[DEBUG] text       = {text!r}")
    print(f"[DEBUG] image_file = {image_file}")
    if image_file:
        print(f"[DEBUG]   .filename       = {image_file.filename!r}")
        print(f"[DEBUG]   .content_length = {image_file.content_length}")
        print(f"[DEBUG]   .content_type   = {image_file.content_type}")

    # ── Handle image upload ───────────────────────────────────────────────
    image_path = None
    if image_file and image_file.filename:
        ext        = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        fname      = f"{uuid.uuid4().hex}{ext}"
        image_path = os.path.join(UPLOAD_FOLDER, fname)
        image_file.save(image_path)
        fsize = os.path.getsize(image_path)
        print(f"[DEBUG] saved {image_path}  ({fsize} bytes)")
        try:
            Image.open(image_path).verify()
            print(f"[DEBUG] verify OK")
        except Exception as ve:
            print(f"[DEBUG] verify FAILED: {ve}")
            os.remove(image_path)
            image_path = None
    else:
        print(f"[DEBUG] image upload skipped (file_obj={image_file is not None}, "
              f"filename={getattr(image_file, 'filename', None)!r})")

    print(f"[DEBUG] final image_path = {image_path!r}")

    if not text and not image_path:
        return jsonify({"error": "Provide a text query or upload an image."}), 400

    # ── Encode query ──────────────────────────────────────────────────────
    try:
        query_vec, mode = rec.encode_query(
            _model, _processor,
            text=text, image_path=image_path,
            text_weight=text_weight, image_weight=image_weight,
        )
    except Exception as e:
        return jsonify({"error": f"Encoding failed: {e}"}), 500

    # ── Retrieve & rerank ─────────────────────────────────────────────────
    results = rec.retrieve_and_rerank(query_vec, _index, _df, _embeddings, _tracker,
                                      per_source_indexes=_per_source_indexes)

    # ── Build response ────────────────────────────────────────────────────
    items = []
    for _, row in results.iterrows():
        items.append({
            "filename":    row["filename"],
            "image_b64":   img_to_b64(row["full_path"]),
            "category":    row.get("category", ""),
            "title":       row.get("title", ""),
            "price":       row.get("price", ""),
            "source":      row.get("source", ""),
            "dataset":     row.get("dataset", ""),
            "final_score": row["final_score"],
            "sim_query":   row["sim_query"],
            "sim_pref":    row["sim_pref"],
            "redundancy":  row["redundancy"],
        })

    return jsonify({
        "mode":          mode,
        "results":       items,
        "query_img_b64": img_to_b64(image_path) if image_path else None,
    })


@app.route("/api/reject", methods=["POST"])
def reject():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    _tracker.reject(filename)
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    global _tracker
    _tracker = rec.PreferenceTracker()
    return jsonify({"ok": True})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_models()
    app.run(debug=False, port=5000)
