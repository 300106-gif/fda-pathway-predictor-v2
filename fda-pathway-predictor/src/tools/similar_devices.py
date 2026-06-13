"""
Similar Device Lookup
Finds historically similar devices and shows their pathway/decision outcomes.
Uses clean_data.csv as the reference database.
"""

import pandas as pd
from pathlib import Path


def find_similar_devices(
    device_class=None,
    advisory_committee=None,
    medical_specialty=None,
    product_code=None,
    data_path="artifacts/clean_data.csv",
    top_n=10,
):
    """
    Find devices with similar characteristics in historical FDA data.
    Returns summary statistics and example devices.
    """
    df = pd.read_csv(data_path)

    # Build similarity filter — progressively relax if too few matches
    mask = pd.Series([True] * len(df))

    # Exact match on device_class (strongest signal)
    if device_class is not None:
        mask &= df["device_class"] == device_class

    # Match on advisory committee / specialty
    if advisory_committee and advisory_committee != "UNKNOWN":
        mask &= df["advisory_committee"] == advisory_committee
    elif medical_specialty and medical_specialty != "UNKNOWN":
        mask &= df["medical_specialty"] == medical_specialty

    # Match on product code (if available)
    if product_code and product_code != "UNK":
        product_mask = mask & (df["product_code"] == product_code)
        if product_mask.sum() >= 5:
            mask = product_mask

    similar = df[mask]

    if len(similar) == 0:
        return None

    # Pathway distribution
    pathway_dist = similar["pathway"].value_counts()
    pathway_pct = (pathway_dist / len(similar) * 100).round(1).to_dict()

    # Decision code distribution (the evidence)
    decision_dist = similar["decision_code"].value_counts()
    decision_pct = (decision_dist / len(similar) * 100).round(1).to_dict()

    # Success rate by pathway
    success_codes = {
        "510k": ["SESE", "SEKD", "SESD", "SESI", "SESK", "SESP", "SEKN"],
        "PMA": ["APPR", "APCV"],
        "De_Novo": ["DENG"],
    }

    success_rates = {}
    for pathway, codes in success_codes.items():
        pathway_devices = similar[similar["pathway"] == pathway]
        if len(pathway_devices) > 0:
            cleared = pathway_devices["decision_code"].isin(codes).sum()
            rate = round(cleared / len(pathway_devices) * 100, 1)
            success_rates[pathway] = {
                "total": len(pathway_devices),
                "cleared": int(cleared),
                "rate": rate,
            }

    # Top example devices (most recent)
    examples = (
        similar.sort_values("decision_date", ascending=False)
        .head(top_n)[["submission_id", "device_name", "pathway", "decision_code", "applicant", "decision_date"]]
        .to_dict("records")
    )

    return {
        "total_similar": len(similar),
        "pathway_distribution": pathway_pct,
        "decision_distribution": decision_pct,
        "success_rates": success_rates,
        "examples": examples,
        "filters_used": {
            "device_class": device_class,
            "advisory_committee": advisory_committee,
            "product_code": product_code,
        },
    }
