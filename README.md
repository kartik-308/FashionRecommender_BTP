# Multi-Modal Fashion Recommendation System

**B.Tech Project (BTP) — LNMIIT**

A content-based fashion recommendation engine that supports **multi-modal queries** (text, image, or a weighted combination) with adaptive user preference tracking.

## Architecture

- **CLIP ViT-B/32** encodes images and text into a shared 512-d embedding space
- **FAISS IndexFlatIP** for cosine-similarity-based retrieval (~423K images)
- **Stratified retrieval** with per-source sub-indexes to ensure diversity across datasets
- **EMA-based preference tracking** that adapts to user interactions in real-time
- **Rejection & redundancy penalties** to suppress disliked and duplicate items

## Datasets

| Source | Images | Origin |
|--------|-------:|--------|
| DeepFashion2 (Train/Test/Val) | ~286K | [DeepFashion2 GitHub](https://github.com/switchablenorms/DeepFashion2) |
| eBay Fashion | ~38K | Web scraping |
| Amazon Fashion | ~99K | Web scraping |

## Scoring Formula

```
final_score = 0.55 × norm_sim_query + 0.30 × norm_sim_pref − 0.15 × redundancy − 0.40 × rejection_penalty
```

Scores are **normalised per-source** (min-max) before combining, ensuring no single dataset dominates results.

## Key Features

- **Multi-modal search**: Text-only, image-only, or combined with adjustable weights
- **Stratified retrieval**: Per-source FAISS sub-indexes prevent dataset imbalance
- **Per-source normalisation**: Fair scoring across datasets with different similarity distributions
- **Proportional diversity**: Guarantees representation from all sources in top-K results
- **EMA preference tracking** (α=0.7): Session-level personalisation without login
- **Reject button**: Hard penalty (β=0.40) suppresses disliked items

## Tech Stack

- Python, Flask
- PyTorch, HuggingFace Transformers (CLIP)
- FAISS (Facebook AI Similarity Search)
- HTML/CSS/JS frontend

## Setup

```bash
pip install -r requirements.txt
python app.py
```

> **Note:** On first run, the system encodes all images through CLIP (takes time on GPU). Embeddings are cached to `data/*.npy` for subsequent runs.

## Project Structure

```
├── app.py              # Flask backend
├── recommender.py      # Core recommendation engine
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend template
├── static/
│   ├── app.js          # Frontend logic
│   └── style.css       # Styling
└── data/               # Cached embeddings & metadata
```
