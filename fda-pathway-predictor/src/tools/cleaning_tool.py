"""Data Cleaning Tool — handles missing values, dedup, type conversion, dataset contract."""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def clean_data(input_path="artifacts/raw_data.csv", output_dir="artifacts"):
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    initial = df.shape[0]
    logger.info(f"Raw data: {initial} rows")

    df = df.drop_duplicates(subset=["submission_id"], keep="first")
    logger.info(f"After dedup: {len(df)} (removed {initial - len(df)})")
    df = df.dropna(subset=["pathway"])
    df = df[df["pathway"].isin(["510k","PMA","De_Novo"])]

    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["decision_year"] = df["decision_date"].dt.year
    df["decision_month"] = df["decision_date"].dt.month

    if "date_received" in df.columns:
        mask = df["date_received"].notna() & df["decision_date"].notna()
        df.loc[mask, "review_days"] = (df.loc[mask, "decision_date"] - df.loc[mask, "date_received"]).dt.days
        df.loc[df["review_days"] < 0, "review_days"] = np.nan
        df.loc[df["review_days"] > 1000, "review_days"] = np.nan
    else:
        df["review_days"] = np.nan

    # Impute missing device_class using decision_code as evidence
    df["device_class"] = pd.to_numeric(df["device_class"], errors="coerce")
    missing_class = df["device_class"].isna()
    se_codes = ["SESE","SEKD","SESD","SESI","SESK","SESP","SEKN"]
    df.loc[missing_class & df["decision_code"].isin(se_codes), "device_class"] = 2
    pma_codes = ["APPR","APCV","APWD"]
    df.loc[missing_class & df["decision_code"].isin(pma_codes), "device_class"] = 3
    df.loc[missing_class & df["decision_code"].isin(["DENG"]), "device_class"] = 2
    # Remaining unknown
    still_missing = df["device_class"].isna().sum()
    if still_missing > 0:
        logger.info(f"  {still_missing} records still missing device_class after imputation")
    mode_class = df["device_class"].mode()[0] if not df["device_class"].mode().empty else 2
    df["device_class_unknown"] = df["device_class"].isna().astype(int)
    df["device_class"] = df["device_class"].fillna(mode_class).astype(int)

    df["country_code"] = df["country_code"].fillna("UNKNOWN").str.upper().str.strip()
    df["is_us"] = (df["country_code"] == "US").astype(int)

    # Cross-fill advisory_committee ↔ medical_specialty before setting UNKNOWN
    # (same concept in FDA data — one is often present when the other is null)
    ac_null = df["advisory_committee"].isna()
    ms_null = df["medical_specialty"].isna()
    df.loc[ac_null & ~ms_null, "advisory_committee"] = df.loc[ac_null & ~ms_null, "medical_specialty"]
    df.loc[ms_null & ~ac_null, "medical_specialty"] = df.loc[ms_null & ~ac_null, "advisory_committee"]

    # Keyword-based rescue: classify still-null specialties from device name
    SAMD_KEYWORDS = [
        "software", "app", "algorithm", "ai ", "artificial intelligence",
        "machine learning", "clinical decision", "decision support",
        "image analysis", "cad ", "computer-aided", "neural network",
        "deep learning", "digital health", "mobile health", "mhealth",
        "samd", "remote monitoring", "telemedicine",
    ]
    IVD_KEYWORDS = [
        "ivd", "in vitro", "test strip", "glucose monitor", "glucometer",
        "reagent", "assay", "immunoassay", "analyzer", "analyser",
        "lateral flow", "pcr", "elisa", "immunodiagnostic", "diagnostic kit",
        "blood glucose", "hba1c", "cholesterol test", "pregnancy test",
        "urinalysis", "urine test", "rapid test", "point of care",
        "culture media", "hematology analyzer", "coagulation analyzer",
    ]

    if "device_name" in df.columns:
        name_lower = df["device_name"].fillna("").str.lower()
        still_null = df["advisory_committee"].isna()

        is_samd = name_lower.apply(lambda n: any(kw in n for kw in SAMD_KEYWORDS))
        is_ivd  = name_lower.apply(lambda n: any(kw in n for kw in IVD_KEYWORDS))

        # SaMD takes precedence over IVD if both match
        df.loc[still_null & is_samd, "advisory_committee"] = "SAMD"
        df.loc[still_null & ~is_samd & is_ivd, "advisory_committee"] = "IVD"

        # Mirror to medical_specialty
        ms_still_null = df["medical_specialty"].isna()
        df.loc[ms_still_null & is_samd, "medical_specialty"] = "SAMD"
        df.loc[ms_still_null & ~is_samd & is_ivd, "medical_specialty"] = "IVD"

        rescued = (still_null & (is_samd | is_ivd)).sum()
        logger.info(f"  Keyword rescue: {rescued} records classified as SaMD/IVD")

    df["advisory_committee"] = df["advisory_committee"].fillna("UNKNOWN").str.upper().str.strip()
    df["medical_specialty"] = df["medical_specialty"].fillna("UNKNOWN").str.upper().str.strip()
    df["decision_code"] = df["decision_code"].fillna("UNKNOWN").str.upper().str.strip()
    df["applicant"] = df["applicant"].fillna("Unknown Applicant").str.strip()
    for col in ["third_party_flag","clearance_type"]:
        if col in df.columns: df[col] = df[col].fillna("UNKNOWN")
    df["product_code"] = df["product_code"].fillna("UNK")

    drop_cols = ["city","state","date_received","regulation_number","advisory_committee_description","medical_specialty_description"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.sort_values("decision_date").reset_index(drop=True)

    logger.info(f"Clean data: {df.shape}")
    df.to_csv(output_dir / "clean_data.csv", index=False)

    contract = _gen_contract(df)
    with open(output_dir / "dataset_contract.json", "w") as f:
        json.dump(contract, f, indent=2, default=str)
    logger.info("Dataset contract saved")
    return df

def _gen_contract(df):
    contract = {
        "name": "FDA Medical Device Submissions - Clean Dataset", "version": "1.0",
        "target_variable": "pathway", "target_classes": sorted(df["pathway"].unique().tolist()),
        "total_records": len(df),
        "schema": {},
        "constraints": {
            "no_nulls_in": ["submission_id","pathway","device_class","advisory_committee"],
            "class_distribution": df["pathway"].value_counts().to_dict(),
            "date_range": {"min": str(df["decision_date"].min()), "max": str(df["decision_date"].max())},
        },
        "recommended_features": ["device_class","advisory_committee","medical_specialty","is_us","country_code","decision_year","decision_month","review_days","clearance_type","third_party_flag"],
    }
    for col in df.columns:
        info = {"dtype": str(df[col].dtype), "null_count": int(df[col].isna().sum())}
        if df[col].dtype == "object":
            info["n_unique"] = int(df[col].nunique())
        elif pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min()) if not df[col].isna().all() else None
            info["max"] = float(df[col].max()) if not df[col].isna().all() else None
        contract["schema"][col] = info
    return contract

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_data()
