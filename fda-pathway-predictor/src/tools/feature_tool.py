"""Feature Engineering Tool — transforms clean data into model-ready features."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json, logging
from pathlib import Path
logger = logging.getLogger(__name__)

def engineer_features(input_path="artifacts/clean_data.csv", contract_path="artifacts/dataset_contract.json", output_dir="artifacts"):
    output_dir = Path(output_dir)
    df = pd.read_csv(input_path)
    with open(contract_path) as f: contract = json.load(f)
    logger.info(f"Loaded {len(df)} records for feature engineering")
    _validate(df, contract)
    target = df["pathway"].copy()
    features = pd.DataFrame()
    features["submission_id"] = df["submission_id"]
    features["device_class"] = df["device_class"]
    features["device_class_unknown"] = df["device_class_unknown"] if "device_class_unknown" in df.columns else 0

    ac_freq = df["advisory_committee"].value_counts(normalize=True).to_dict()
    features["advisory_committee_freq"] = df["advisory_committee"].map(ac_freq)
    le_ac = LabelEncoder(); features["advisory_committee_encoded"] = le_ac.fit_transform(df["advisory_committee"].fillna("UNKNOWN"))

    ms_freq = df["medical_specialty"].value_counts(normalize=True).to_dict()
    features["medical_specialty_freq"] = df["medical_specialty"].map(ms_freq)
    le_ms = LabelEncoder(); features["medical_specialty_encoded"] = le_ms.fit_transform(df["medical_specialty"].fillna("UNKNOWN"))

    features["is_us"] = df["is_us"] if "is_us" in df.columns else (df["country_code"]=="US").astype(int)
    country_freq = df["country_code"].value_counts(normalize=True).to_dict()
    features["country_freq"] = df["country_code"].map(country_freq)

    if "decision_year" in df.columns: features["decision_year"] = df["decision_year"]
    if "decision_month" in df.columns:
        features["decision_month"] = df["decision_month"]
        features["month_sin"] = np.sin(2*np.pi*df["decision_month"]/12)
        features["month_cos"] = np.cos(2*np.pi*df["decision_month"]/12)

    if "review_days" in df.columns:
        median_days = df["review_days"].median()
        features["review_days"] = df["review_days"].fillna(median_days if pd.notna(median_days) else 0)
        features["has_review_days"] = df["review_days"].notna().astype(int)

    le_ct = None
    if "clearance_type" in df.columns:
        le_ct = LabelEncoder(); features["clearance_type_encoded"] = le_ct.fit_transform(df["clearance_type"].fillna("UNKNOWN"))
    if "third_party_flag" in df.columns:
        features["third_party"] = (df["third_party_flag"]=="Y").astype(int)

    pc_freq = df["product_code"].value_counts(normalize=True).to_dict()
    features["product_code_freq"] = df["product_code"].map(pc_freq).fillna(0)
    app_freq = df["applicant"].value_counts(normalize=True).to_dict()
    features["applicant_freq"] = df["applicant"].map(app_freq)
    app_counts = df["applicant"].value_counts().to_dict()
    features["applicant_submission_count"] = df["applicant"].map(app_counts)

    le_target = LabelEncoder()
    features["pathway"] = target
    features["pathway_encoded"] = le_target.fit_transform(target)

    logger.info(f"Engineered {features.shape[1]-2} features")
    features.to_csv(output_dir/"features.csv", index=False)

    encoders = {
        "advisory_committee": dict(zip(le_ac.classes_.tolist(), range(len(le_ac.classes_)))),
        "medical_specialty": dict(zip(le_ms.classes_.tolist(), range(len(le_ms.classes_)))),
        "pathway": dict(zip(le_target.classes_.tolist(), range(len(le_target.classes_)))),
        "pathway_inverse": dict(zip(range(len(le_target.classes_)), le_target.classes_.tolist())),
    }
    if le_ct: encoders["clearance_type"] = dict(zip(le_ct.classes_.tolist(), range(len(le_ct.classes_))))
    with open(output_dir/"label_encoders.json","w") as f: json.dump(encoders, f, indent=2, default=str)
    return features

def _validate(df, contract):
    target = contract.get("target_variable","pathway")
    if target not in df.columns: raise ValueError(f"Target '{target}' missing")
    for col in contract.get("constraints",{}).get("no_nulls_in",[]):
        if col in df.columns and df[col].isna().any():
            logger.warning(f"Column '{col}' has nulls")
    logger.info("Contract validation passed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); engineer_features()
