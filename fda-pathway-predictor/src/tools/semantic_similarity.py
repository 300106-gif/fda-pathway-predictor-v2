"""Semantic Similarity — rank similar FDA devices against a user query.

Degradation chain:
  sentence_transformers + FAISS  (best quality, optional install)
  -> sklearn TF-IDF + cosine     (always available, good quality)
  -> existing similar_devices.py (structural fallback)

Never raises — any failure returns None so caller can use structural search.
"""
import json, logging, numpy as np
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_TFIDF_CACHE: dict = {}   # keyed by data_path to avoid re-fitting per request


def _build_query(device_name: str, device_description: str) -> str:
    parts = [p.strip() for p in [device_name, device_description] if p and p.strip()]
    return " ".join(parts) or "medical device"


def _device_text(row: pd.Series) -> str:
    parts = [str(row.get("device_name", "")), str(row.get("product_code", ""))]
    return " ".join(p for p in parts if p and p != "nan")


# ── TF-IDF path (scikit-learn — always available) ─────────────────────────────
def _tfidf_search(query: str, df: pd.DataFrame, top_n: int, data_path: str) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_key = data_path
    if corpus_key not in _TFIDF_CACHE:
        corpus = df.apply(_device_text, axis=1).tolist()
        vec = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
        )
        matrix = vec.fit_transform(corpus)
        _TFIDF_CACHE[corpus_key] = (vec, matrix, corpus)
        logger.info(f"TF-IDF index built: {matrix.shape}")

    vec, matrix, _ = _TFIDF_CACHE[corpus_key]
    q_vec = vec.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_n]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "submission_id": str(row.get("submission_id", "")),
            "device_name": str(row.get("device_name", "")),
            "pathway": str(row.get("pathway", "")),
            "decision_code": str(row.get("decision_code", "")),
            "applicant": str(row.get("applicant", "")),
            "decision_date": str(row.get("decision_date", "")),
            "product_code": str(row.get("product_code", "")),
            "similarity_score": round(float(scores[idx]), 4),
            "similarity_method": "tfidf",
        })
    return results


# ── sentence-transformers + FAISS path (optional) ─────────────────────────────
_ST_CACHE: dict = {}   # keyed by data_path


def _st_search(query: str, df: pd.DataFrame, top_n: int, data_path: str) -> list:
    import faiss
    from sentence_transformers import SentenceTransformer

    cache_key = data_path
    if cache_key not in _ST_CACHE:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus = df.apply(_device_text, axis=1).tolist()
        embeddings = model.encode(corpus, batch_size=128, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        _ST_CACHE[cache_key] = (model, index, corpus)
        logger.info(f"FAISS index built: {embeddings.shape}")

    model, index, _ = _ST_CACHE[cache_key]
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), top_n)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[idx]
        results.append({
            "submission_id": str(row.get("submission_id", "")),
            "device_name": str(row.get("device_name", "")),
            "pathway": str(row.get("pathway", "")),
            "decision_code": str(row.get("decision_code", "")),
            "applicant": str(row.get("applicant", "")),
            "decision_date": str(row.get("decision_date", "")),
            "product_code": str(row.get("product_code", "")),
            "similarity_score": round(float(score), 4),
            "similarity_method": "sentence_transformers",
        })
    return results


# ── Public API ─────────────────────────────────────────────────────────────────
def find_semantic_similar(
    device_name: str = "",
    device_description: str = "",
    device_class: int = None,
    advisory_committee: str = None,
    top_n: int = 10,
    data_path: str = "artifacts/clean_data.csv",
    output_path: str = "artifacts/similar_device_report.json",
) -> dict | None:
    """
    Find semantically similar devices. Returns None if data unavailable.

    Returns:
        {
            "total_matches": 847,
            "pathway_distribution": {"510k": 94.2, "PMA": 3.8, "De_Novo": 2.0},
            "top_similar_devices": [...],
            "similarity_method": "tfidf" | "sentence_transformers",
        }
    """
    if not Path(data_path).exists():
        logger.warning(f"clean_data.csv not found at {data_path}")
        return None

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.warning(f"Could not load {data_path}: {e}")
        return None

    # Optional pre-filter to speed up search (same device class if known)
    df_search = df.copy()
    if device_class is not None and "device_class" in df_search.columns:
        filtered = df_search[df_search["device_class"] == device_class]
        if len(filtered) >= 50:
            df_search = filtered

    query = _build_query(device_name, device_description)
    top_devices = []

    # Try sentence_transformers + FAISS first
    try:
        top_devices = _st_search(query, df_search, top_n, data_path)
        method = "sentence_transformers"
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"ST search failed: {e}")

    # Fall back to TF-IDF
    if not top_devices:
        try:
            top_devices = _tfidf_search(query, df_search, top_n, data_path)
            method = "tfidf"
        except Exception as e:
            logger.warning(f"TF-IDF search failed: {e}")
            return None

    if not top_devices:
        return None

    method = top_devices[0].get("similarity_method", "tfidf") if top_devices else "tfidf"

    # Compute pathway distribution over the full filtered set
    total = len(df_search)
    pw_dist = {}
    if "pathway" in df_search.columns:
        pw_dist = (
            df_search["pathway"].value_counts() / total * 100
        ).round(1).to_dict()

    result = {
        "total_matches": total,
        "pathway_distribution": pw_dist,
        "top_similar_devices": top_devices,
        "similarity_method": method,
        "query": query[:100],
    }

    try:
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save similar device report: {e}")

    return result


if __name__ == "__main__":
    import pprint
    r = find_semantic_similar(
        device_name="Wireless cardiac rhythm monitor",
        device_description="ECG monitoring wearable",
        device_class=2,
        advisory_committee="CV",
    )
    if r:
        print(f"Method: {r['similarity_method']}, Total: {r['total_matches']}")
        pprint.pprint(r["top_similar_devices"][:3])
