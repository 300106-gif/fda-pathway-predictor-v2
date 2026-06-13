"""
FDA Data Ingestion Tool — fetches from openFDA API endpoints.
Supports API key authentication for higher rate limits (120K requests/day).
"""
import os
import requests
import pandas as pd
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/device"
API_KEY = os.getenv("OPENFDA_API_KEY", "")

ENDPOINTS = {
    "510k": f"{BASE_URL}/510k.json",
    "pma": f"{BASE_URL}/pma.json",
    "classification": f"{BASE_URL}/classification.json",
}

FIELDS_510K = [
    "k_number","device_name","product_code","advisory_committee",
    "advisory_committee_description","decision_code","decision_description",
    "clearance_type","date_received","decision_date","applicant",
    "country_code","state","city","third_party_flag","expedited_review_flag",
]
FIELDS_PMA = [
    "pma_number","trade_name","product_code","advisory_committee",
    "advisory_committee_description","decision_code","decision_date",
    "applicant","city","state","country_code",
]
FIELDS_CLASSIFICATION = [
    "product_code","device_name","device_class","medical_specialty",
    "medical_specialty_description","regulation_number","definition",
    "gmp_exempt_flag","submission_type_id",
]


def fetch_from_api(endpoint_url, search=None, limit=1000, max_retries=3):
    """Fetch records from an openFDA endpoint with API key and retry logic."""
    params = {"limit": limit}
    if API_KEY:
        params["api_key"] = API_KEY
    if search:
        params["search"] = search

    for attempt in range(max_retries):
        try:
            resp = requests.get(endpoint_url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", []), data.get("meta", {})
            elif resp.status_code == 404:
                return [], {}
            elif resp.status_code == 403:
                logger.error(f"API blocked (403): {resp.text[:100]}")
                return [], {}
            elif resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return [], {}


def fetch_paginated(endpoint_url, date_field, start_year=2000, end_year=2026, fields=None):
    """Fetch all records by paginating through yearly date ranges."""
    all_records = []
    for year in range(start_year, end_year + 1):
        search = f"{date_field}:[{year}0101 TO {year}1231]"
        records, meta = fetch_from_api(endpoint_url, search=search, limit=1000)
        if records:
            total = meta.get("results", {}).get("total", len(records))
            logger.info(f"  Year {year}: fetched {len(records)}/{total} records")
            if total > 1000:
                logger.info(f"  Year {year} has {total} records, splitting by quarter...")
                records = []
                quarters = [
                    (f"{year}0101", f"{year}0331"), (f"{year}0401", f"{year}0630"),
                    (f"{year}0701", f"{year}0930"), (f"{year}1001", f"{year}1231"),
                ]
                for q_start, q_end in quarters:
                    q_search = f"{date_field}:[{q_start} TO {q_end}]"
                    q_records, _ = fetch_from_api(endpoint_url, search=q_search, limit=1000)
                    if q_records:
                        records.extend(q_records)
                        logger.info(f"    Q {q_start[:6]}: {len(q_records)} records")
                    time.sleep(0.25)
            all_records.extend(records)
        else:
            logger.info(f"  Year {year}: no records")
        time.sleep(0.25)

    if fields:
        cleaned = []
        for record in all_records:
            row = {f: record.get(f) for f in fields}
            openfda = record.get("openfda", {})
            if openfda:
                row["regulation_number"] = openfda.get("regulation_number", [None])[0] if openfda.get("regulation_number") else None
            cleaned.append(row)
        return cleaned
    return all_records


def _classify_pathway(submission_id):
    if not submission_id: return "unknown"
    sid = str(submission_id).upper().strip()
    if sid.startswith("DEN"): return "De_Novo"
    elif sid.startswith("K") or sid.startswith("BK"): return "510k"
    return "unknown"


def fetch_510k_data(start_year=2000, end_year=2026):
    logger.info("Fetching 510(k) / De Novo data...")
    records = fetch_paginated(ENDPOINTS["510k"], "decision_date", start_year, end_year, FIELDS_510K)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df["pathway"] = df["k_number"].apply(lambda x: _classify_pathway(x) if pd.notna(x) else "unknown")
        df = df.rename(columns={"k_number": "submission_id"})
    logger.info(f"Total 510(k)/De Novo records: {len(df)}")
    return df


def fetch_pma_data(start_year=2000, end_year=2026):
    logger.info("Fetching PMA data...")
    records = fetch_paginated(ENDPOINTS["pma"], "decision_date", start_year, end_year, FIELDS_PMA)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df["pathway"] = "PMA"
        df = df.rename(columns={"pma_number": "submission_id", "trade_name": "device_name"})
    logger.info(f"Total PMA records: {len(df)}")
    return df


def fetch_classification_data():
    logger.info("Fetching classification data...")
    all_records = []
    records, meta = fetch_from_api(ENDPOINTS["classification"], limit=1000)
    if records:
        all_records.extend(records)
    cleaned = [{f: r.get(f) for f in FIELDS_CLASSIFICATION} for r in all_records]
    df = pd.DataFrame(cleaned)
    logger.info(f"Total classification records: {len(df)}")
    return df


def merge_and_unify(df_510k, df_pma, df_classification):
    logger.info("Merging datasets...")
    common_cols = [
        "submission_id","pathway","device_name","product_code",
        "advisory_committee","advisory_committee_description",
        "decision_code","decision_date","applicant","country_code","state","city",
    ]
    for col in common_cols:
        if col not in df_510k.columns: df_510k[col] = None
        if col not in df_pma.columns: df_pma[col] = None
    df_unified = pd.concat([df_510k[common_cols], df_pma[common_cols]], ignore_index=True)
    if len(df_classification) > 0:
        df_class_unique = df_classification.drop_duplicates(subset=["product_code"])
        class_cols = [c for c in ["product_code","device_class","medical_specialty",
                      "medical_specialty_description","regulation_number",
                      "gmp_exempt_flag","submission_type_id"]
                      if c in df_class_unique.columns]
        df_unified = df_unified.merge(df_class_unique[class_cols], on="product_code", how="left")
    logger.info(f"Unified dataset: {len(df_unified)} records")
    logger.info(f"Pathway distribution:\n{df_unified['pathway'].value_counts()}")
    return df_unified


def ingest_fda_data(output_path="artifacts/raw_data.csv", start_year=2000, end_year=2026):
    df_510k = fetch_510k_data(start_year, end_year)
    df_pma = fetch_pma_data(start_year, end_year)
    df_classification = fetch_classification_data()
    df_unified = merge_and_unify(df_510k, df_pma, df_classification)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_unified.to_csv(output, index=False)
    df_classification.to_csv(output.parent / "classification_reference.csv", index=False)
    logger.info(f"Raw data saved to {output_path}")
    return df_unified


if __name__ == "__main__":
    df = ingest_fda_data(start_year=2000, end_year=2026)
    print(f"\nShape: {df.shape}")
    print(f"\nPathway distribution:\n{df['pathway'].value_counts()}")
