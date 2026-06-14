"""Data Cleaning Tool — handles missing values, dedup, type conversion, dataset contract."""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

# submission_type_id values that mean 510(k) exempt in the foiclass database
_EXEMPT_SUB_TYPES = {"", "4", "7"}

def _load_foiclass(foiclass_path):
    """Load foiclass CSV; return (flags_lookup, raw_df)."""
    p = Path(foiclass_path)
    if not p.exists():
        logger.warning(f"foiclass not found at {p}; skipping flag enrichment")
        return {}, pd.DataFrame()
    foi = pd.read_csv(p, dtype=str).fillna("")
    foi.columns = [c.strip().lower() for c in foi.columns]
    flags = {}
    for _, row in foi.iterrows():
        pc = str(row.get("productcode", "")).upper().strip()
        if pc:
            flags[pc] = {
                "implant_flag":     1 if row.get("implant_flag", "") == "Y" else 0,
                "life_sustain_flag": 1 if row.get("life_sustain_support_flag", "") == "Y" else 0,
                "gmp_exempt":       1 if row.get("gmpexemptflag", "") == "Y" else 0,
            }
    return flags, foi

def _build_exempt_records(foi_df):
    """Create one synthetic training row per exempt product-code entry in foiclass."""
    exempt = foi_df[foi_df["submission_type_id"].isin(_EXEMPT_SUB_TYPES)].copy()
    logger.info(f"Exempt product codes in foiclass: {len(exempt)}")
    rows = []
    years = list(range(2015, 2025))
    for i, (_, row) in enumerate(exempt.iterrows()):
        pc  = str(row.get("productcode", "")).upper().strip()
        cls = str(row.get("deviceclass", "1")).strip()
        cls = cls if cls in ("1", "2", "3") else "1"
        spec = str(row.get("medicalspecialty", "")).upper().strip() or "UNKNOWN"
        yr   = years[i % len(years)]
        mo   = (i % 12) + 1
        rows.append({
            "submission_id":                f"EXEMPT_{pc}_{i}",
            "pathway":                      "510k_exempt",
            "device_name":                  str(row.get("devicename", "")).strip(),
            "product_code":                 pc,
            "advisory_committee":           spec,
            "advisory_committee_description": "",
            "decision_code":                "EXEMPT",
            "decision_date":                f"{yr}-{mo:02d}-01",
            "applicant":                    "Exempt Device Manufacturer",
            "country_code":                 "US",
            "state":                        "",
            "city":                         "",
            "device_class":                 cls,
            "medical_specialty":            spec,
            "medical_specialty_description": "",
            "regulation_number":            str(row.get("regulationnumber", "")).strip(),
            "gmp_exempt_flag":              str(row.get("gmpexemptflag", "N")).strip(),
            "submission_type_id":           str(row.get("submission_type_id", "")).strip(),
            "implant_flag":                 str(row.get("implant_flag", "N")).strip(),
            "life_sustain_flag":            str(row.get("life_sustain_support_flag", "N")).strip(),
        })
    return pd.DataFrame(rows)

def clean_data(input_path="artifacts/raw_data.csv", output_dir="artifacts"):
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    initial = df.shape[0]
    logger.info(f"Raw data: {initial} rows")

    # ── Load foiclass for flag enrichment + exempt records ─────────────────
    foiclass_path = Path(output_dir) / "foiclass.csv"
    foi_flags, foi_df = _load_foiclass(foiclass_path)

    # Enrich existing records with device-level flags from foiclass
    pc_upper = df["product_code"].fillna("").str.upper().str.strip()
    df["implant_flag"]     = pc_upper.map(lambda p: "Y" if foi_flags.get(p, {}).get("implant_flag", 0) else "N")
    df["life_sustain_flag"] = pc_upper.map(lambda p: "Y" if foi_flags.get(p, {}).get("life_sustain_flag", 0) else "N")
    # Prefer raw gmp_exempt_flag when present, otherwise fall back to foiclass lookup
    if "gmp_exempt_flag" not in df.columns:
        df["gmp_exempt_flag"] = "N"
    df["gmp_exempt_flag"] = df["gmp_exempt_flag"].where(
        df["gmp_exempt_flag"].isin(["Y", "N"]),
        other=pc_upper.map(lambda p: "Y" if foi_flags.get(p, {}).get("gmp_exempt", 0) else "N"),
    )

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

    # Impute missing device_class from dataset median ONLY.
    # Do NOT use decision_code or pathway as a proxy — that creates a circular feature
    # where device_class encodes the label, inflating model accuracy to 1.00.
    df["device_class"] = pd.to_numeric(df["device_class"], errors="coerce")
    still_missing = df["device_class"].isna().sum()
    if still_missing > 0:
        logger.info(f"  {still_missing} records missing device_class — filling with dataset median")
    mode_class = int(df["device_class"].mode()[0]) if not df["device_class"].mode().empty else 2
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
        "ivd", "in vitro", "test strip", "glucometer",
        "reagent", "assay", "immunoassay",
        "lateral flow", "pcr", "elisa", "immunodiagnostic", "diagnostic kit",
        "blood glucose", "hba1c", "cholesterol test", "pregnancy test",
        "urinalysis", "urine test", "rapid test",
        "culture media",
    ]
    ANALYZER_POC_KEYWORDS = [
        "glucose monitor", "analyzer", "analyser",
        "hematology analyzer", "coagulation analyzer",
        "point of care", "poc ",
    ]
    IMPLANT_KEYWORDS = [
        "implant", "implantable", "implanted",
        "pacemaker", "stent", "defibrillator", "icd ",
        "hip replacement", "knee replacement", "joint replacement",
        "intraocular lens", "iol ", "cochlear implant",
        "spinal cord stimulator", "deep brain stimulator",
        "breast implant", "vascular graft", "cardiac implant",
        "orthopedic implant", "dental implant", "bone screw",
        "neurostimulator", "sacral neuromodulation",
    ]

    if "device_name" in df.columns:
        name_lower = df["device_name"].fillna("").str.lower()
        still_null = df["advisory_committee"].isna()

        is_samd         = name_lower.apply(lambda n: any(kw in n for kw in SAMD_KEYWORDS))
        is_ivd          = name_lower.apply(lambda n: any(kw in n for kw in IVD_KEYWORDS))
        is_implant      = name_lower.apply(lambda n: any(kw in n for kw in IMPLANT_KEYWORDS))
        is_analyzer_poc = name_lower.apply(lambda n: any(kw in n for kw in ANALYZER_POC_KEYWORDS))

        # Priority: SaMD > IVD > IMPLANT > ANALYZER_POC
        df.loc[still_null & is_samd, "advisory_committee"] = "SAMD"
        df.loc[still_null & ~is_samd & is_ivd, "advisory_committee"] = "IVD"
        df.loc[still_null & ~is_samd & ~is_ivd & is_implant, "advisory_committee"] = "IMPLANT"
        df.loc[still_null & ~is_samd & ~is_ivd & ~is_implant & is_analyzer_poc, "advisory_committee"] = "ANALYZER_POC"

        # Mirror to medical_specialty
        ms_still_null = df["medical_specialty"].isna()
        df.loc[ms_still_null & is_samd, "medical_specialty"] = "SAMD"
        df.loc[ms_still_null & ~is_samd & is_ivd, "medical_specialty"] = "IVD"
        df.loc[ms_still_null & ~is_samd & ~is_ivd & is_implant, "medical_specialty"] = "IMPLANT"
        df.loc[ms_still_null & ~is_samd & ~is_ivd & ~is_implant & is_analyzer_poc, "medical_specialty"] = "ANALYZER_POC"

        rescued = (still_null & (is_samd | is_ivd | is_implant | is_analyzer_poc)).sum()
        logger.info(f"  Keyword rescue: {rescued} records classified as SaMD/IVD/IMPLANT/ANALYZER_POC")

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

    # ── Append 510k_exempt synthetic records from foiclass ─────────────────
    if not foi_df.empty:
        exempt_df = _build_exempt_records(foi_df)
        # Apply same date parsing & year/month extraction
        exempt_df["decision_date"] = pd.to_datetime(exempt_df["decision_date"], errors="coerce")
        exempt_df["decision_year"]  = exempt_df["decision_date"].dt.year
        exempt_df["decision_month"] = exempt_df["decision_date"].dt.month
        exempt_df["review_days"]    = np.nan
        exempt_df["device_class"]   = pd.to_numeric(exempt_df["device_class"], errors="coerce").fillna(1).astype(int)
        exempt_df["device_class_unknown"] = 0
        exempt_df["is_us"]          = 1
        exempt_df["country_code"]   = "US"
        exempt_df["decision_code"]  = "EXEMPT"
        # Align columns to main df
        for col in df.columns:
            if col not in exempt_df.columns:
                exempt_df[col] = np.nan
        exempt_df = exempt_df[df.columns]
        df = pd.concat([df, exempt_df], ignore_index=True)
        logger.info(f"After adding 510k_exempt records: {len(df)} total rows")

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
            "pathway_classes": ["510k", "510k_exempt", "De_Novo", "PMA"],
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
