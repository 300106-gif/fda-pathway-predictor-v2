"""Predicate Lookup — find likely predicate devices from historical 510(k) records.

Sources (in priority order):
  1. Semantic similar devices (if semantic_similarity available)
  2. Structural filter on clean_data.csv (device_class + advisory_committee + product_code)
  3. openFDA 510k API (if product_code provided and internet available)

Output: artifacts/predicate_candidates.json
Never raises — returns empty list on any failure.
"""
import json, logging, requests
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CLEARED_CODES = {"SESE", "SEKD", "SESD", "SESI", "SESK", "SESP", "SEKN", "SESK"}


def _score_recency(decision_date: str) -> float:
    """Linear recency score 0-1 (recent = higher)."""
    try:
        year = int(str(decision_date)[:4])
        return min(1.0, max(0.0, (year - 1990) / (2025 - 1990)))
    except Exception:
        return 0.5


def _from_clean_data(
    device_class: int,
    advisory_committee: str,
    product_code: str,
    data_path: str,
    top_n: int,
) -> list:
    """Pull 510k records from local clean_data.csv using structural filters."""
    if not Path(data_path).exists():
        return []
    try:
        df = pd.read_csv(data_path)
        # Only 510k submissions
        mask = df["pathway"] == "510k"
        if device_class is not None:
            mask &= df["device_class"] == device_class
        if advisory_committee and advisory_committee != "UNKNOWN":
            mask &= df["advisory_committee"] == advisory_committee
        subset = df[mask].copy()
        # Tighten to product code if enough matches
        if product_code and product_code not in ("UNK", ""):
            pc_subset = subset[subset["product_code"] == product_code]
            if len(pc_subset) >= 3:
                subset = pc_subset

        if subset.empty:
            return []

        # Score = cleared flag * 0.6 + recency * 0.4
        subset = subset.copy()
        subset["_cleared"] = subset["decision_code"].isin(CLEARED_CODES).astype(float)
        subset["_recency"] = subset["decision_date"].apply(_score_recency)
        subset["_score"] = subset["_cleared"] * 0.6 + subset["_recency"] * 0.4
        subset = subset.sort_values("_score", ascending=False).head(top_n)

        return [
            {
                "submission_id": str(row.get("submission_id", "")),
                "device_name": str(row.get("device_name", "")),
                "applicant": str(row.get("applicant", "")),
                "decision_date": str(row.get("decision_date", "")),
                "decision_code": str(row.get("decision_code", "")),
                "product_code": str(row.get("product_code", "")),
                "similarity_score": round(float(row["_score"]), 3),
                "source": "local",
            }
            for _, row in subset.iterrows()
        ]
    except Exception as e:
        logger.warning(f"Local predicate lookup failed: {e}")
        return []


def _from_fda_api(product_code: str, top_n: int) -> list:
    """Fetch predicate candidates from openFDA 510k API."""
    if not product_code or product_code in ("UNK", ""):
        return []
    try:
        r = requests.get(
            f"https://api.fda.gov/device/510k.json"
            f"?search=product_code:{product_code}"
            f"&sort=decision_date:desc&limit={top_n}",
            timeout=10,
        )
        if r.status_code == 200:
            return [
                {
                    "submission_id": rec.get("k_number", ""),
                    "device_name": rec.get("device_name", ""),
                    "applicant": rec.get("applicant", ""),
                    "decision_date": rec.get("decision_date", ""),
                    "decision_code": rec.get("decision_code", ""),
                    "product_code": rec.get("product_code", product_code),
                    "similarity_score": _score_recency(rec.get("decision_date", "")),
                    "source": "fda_api",
                }
                for rec in r.json().get("results", [])
            ]
    except Exception as e:
        logger.warning(f"FDA API predicate lookup failed: {e}")
    return []


def _from_semantic(semantic_result: dict, top_n: int) -> list:
    """Convert semantic similarity results into predicate candidates (510k only)."""
    if not semantic_result:
        return []
    candidates = []
    for dev in semantic_result.get("top_similar_devices", []):
        if dev.get("pathway") == "510k":
            candidates.append({
                "submission_id": dev.get("submission_id", ""),
                "device_name": dev.get("device_name", ""),
                "applicant": dev.get("applicant", ""),
                "decision_date": dev.get("decision_date", ""),
                "decision_code": dev.get("decision_code", ""),
                "product_code": dev.get("product_code", ""),
                "similarity_score": dev.get("similarity_score", 0.0),
                "source": f"semantic_{dev.get('similarity_method', 'tfidf')}",
            })
        if len(candidates) >= top_n:
            break
    return candidates


def find_predicates(
    device_name: str = "",
    device_description: str = "",
    device_class: int = None,
    advisory_committee: str = None,
    product_code: str = None,
    pathway: str = "510k",
    top_n: int = 5,
    data_path: str = "artifacts/clean_data.csv",
    output_path: str = "artifacts/predicate_candidates.json",
    semantic_result: dict = None,
) -> list:
    """
    Find predicate device candidates. Returns list of candidates (may be empty).

    Each candidate:
        {
            "submission_id": "K241234",
            "device_name": "CardioTrack Pro",
            "applicant": "...",
            "decision_date": "20240101",
            "decision_code": "SESE",
            "product_code": "DQK",
            "similarity_score": 0.94,
            "source": "local" | "fda_api" | "semantic_tfidf",
        }
    """
    # Predicates only relevant for 510k and De Novo (De Novo has NO predicate by definition
    # but we still search to confirm novelty)
    candidates: list = []

    # 1. Semantic results (highest quality when available)
    if semantic_result:
        candidates = _from_semantic(semantic_result, top_n)
        logger.info(f"Semantic predicates: {len(candidates)}")

    # 2. Structural local lookup
    if len(candidates) < top_n:
        local = _from_clean_data(device_class, advisory_committee, product_code, data_path, top_n)
        # Merge: avoid submission_id duplicates
        seen = {c["submission_id"] for c in candidates}
        for item in local:
            if item["submission_id"] not in seen:
                candidates.append(item)
                seen.add(item["submission_id"])
        logger.info(f"After local lookup: {len(candidates)} candidates")

    # 3. FDA API (if product_code provided)
    if len(candidates) < top_n and product_code:
        api_results = _from_fda_api(product_code, top_n)
        seen = {c["submission_id"] for c in candidates}
        for item in api_results:
            if item["submission_id"] not in seen:
                candidates.append(item)
                seen.add(item["submission_id"])
        logger.info(f"After FDA API: {len(candidates)} candidates")

    candidates = candidates[:top_n]

    result = {
        "predicates": candidates,
        "total_found": len(candidates),
        "pathway_context": pathway,
    }

    try:
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save predicate candidates: {e}")

    return candidates


if __name__ == "__main__":
    import pprint
    r = find_predicates(
        device_name="Cardiac rhythm monitor",
        device_class=2,
        advisory_committee="CV",
        product_code="DQK",
        pathway="510k",
    )
    pprint.pprint(r)
