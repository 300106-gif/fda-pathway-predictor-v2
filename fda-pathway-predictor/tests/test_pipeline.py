"""Basic pipeline tests — validate each tool module independently."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ── Sample data fixture ────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame that mimics raw_data.csv."""
    np.random.seed(42)
    n = 200
    records = []
    for i in range(150):
        records.append({
            "submission_id": f"K{i:04d}", "pathway": "510k",
            "device_name": f"Device {i}", "product_code": "ABC",
            "advisory_committee": "CV", "decision_code": "SESE",
            "decision_date": f"2020-{(i%12)+1:02d}-01",
            "applicant": "Test Corp", "country_code": "US",
            "device_class": 2, "medical_specialty": "CV",
            "date_received": f"2020-{(i%12)+1:02d}-01",
            "clearance_type": "Traditional", "third_party_flag": "N",
        })
    for i in range(40):
        records.append({
            "submission_id": f"P{i:04d}", "pathway": "PMA",
            "device_name": f"PMA Device {i}", "product_code": "XYZ",
            "advisory_committee": "CV", "decision_code": "APPR",
            "decision_date": f"2020-{(i%12)+1:02d}-01",
            "applicant": "Big Corp", "country_code": "US",
            "device_class": 3, "medical_specialty": "CV",
            "date_received": None, "clearance_type": None, "third_party_flag": None,
        })
    for i in range(10):
        records.append({
            "submission_id": f"DEN{i:04d}", "pathway": "De_Novo",
            "device_name": f"Novel Device {i}", "product_code": "DNV",
            "advisory_committee": "CH", "decision_code": "DENG",
            "decision_date": f"2020-{(i%12)+1:02d}-01",
            "applicant": "Startup", "country_code": "DE",
            "device_class": 2, "medical_specialty": "CH",
            "date_received": None, "clearance_type": None, "third_party_flag": None,
        })
    return pd.DataFrame(records)


# ── generate_sample_data ───────────────────────────────────────────────────────

def test_generate_sample_data_shape():
    from src.tools.generate_sample_data import generate_records
    df = generate_records(n_510k=100, n_pma=20, n_denovo=10)
    assert len(df) >= 130
    assert "pathway" in df.columns
    assert set(df["pathway"].unique()).issubset({"510k", "PMA", "De_Novo"})


def test_generate_sample_data_pathways():
    from src.tools.generate_sample_data import generate_records
    df = generate_records(n_510k=50, n_pma=10, n_denovo=5)
    counts = df["pathway"].value_counts()
    assert counts.get("510k", 0) >= 50
    assert counts.get("PMA", 0) >= 10
    assert counts.get("De_Novo", 0) >= 5


# ── cleaning_tool ──────────────────────────────────────────────────────────────

def test_clean_data_removes_dupes(tmp_path, sample_df):
    dupe = sample_df.iloc[:5].copy()
    dupe = dupe.assign(submission_id=dupe["submission_id"] + "A")
    # Insert one exact duplicate of the first row
    exact_dupe = sample_df.iloc[:1].copy()
    df_with_dupe = pd.concat([sample_df, dupe, exact_dupe], ignore_index=True)
    raw_path = tmp_path / "raw_data.csv"
    df_with_dupe.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    df_clean = clean_data(input_path=str(raw_path), output_dir=str(tmp_path))
    assert len(df_clean) < len(df_with_dupe)


def test_clean_data_produces_contract(tmp_path, sample_df):
    raw_path = tmp_path / "raw_data.csv"
    sample_df.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    clean_data(input_path=str(raw_path), output_dir=str(tmp_path))

    contract_path = tmp_path / "dataset_contract.json"
    assert contract_path.exists()

    import json
    with open(contract_path) as f:
        contract = json.load(f)
    assert contract["target_variable"] == "pathway"
    assert "total_records" in contract


def test_clean_data_imputes_device_class(tmp_path, sample_df):
    sample_df.loc[sample_df["decision_code"] == "SESE", "device_class"] = None
    raw_path = tmp_path / "raw_data.csv"
    sample_df.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    df_clean = clean_data(input_path=str(raw_path), output_dir=str(tmp_path))
    assert df_clean["device_class"].isna().sum() == 0


# ── feature_tool ──────────────────────────────────────────────────────────────

def test_engineer_features_output_shape(tmp_path, sample_df):
    raw_path = tmp_path / "raw_data.csv"
    sample_df.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    clean_data(input_path=str(raw_path), output_dir=str(tmp_path))

    from src.tools.feature_tool import engineer_features
    features = engineer_features(
        input_path=str(tmp_path / "clean_data.csv"),
        contract_path=str(tmp_path / "dataset_contract.json"),
        output_dir=str(tmp_path),
    )
    assert "pathway_encoded" in features.columns
    assert "device_class" in features.columns
    assert len(features) > 0


# ── similar_devices ───────────────────────────────────────────────────────────

def test_find_similar_devices(tmp_path, sample_df):
    raw_path = tmp_path / "raw_data.csv"
    sample_df.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    clean_data(input_path=str(raw_path), output_dir=str(tmp_path))

    from src.tools.similar_devices import find_similar_devices
    result = find_similar_devices(
        device_class=2,
        advisory_committee="CV",
        data_path=str(tmp_path / "clean_data.csv"),
    )
    assert result is not None
    assert result["total_similar"] > 0
    assert "pathway_distribution" in result
    assert "success_rates" in result
    assert "examples" in result


def test_find_similar_devices_no_match(tmp_path, sample_df):
    raw_path = tmp_path / "raw_data.csv"
    sample_df.to_csv(raw_path, index=False)

    from src.tools.cleaning_tool import clean_data
    clean_data(input_path=str(raw_path), output_dir=str(tmp_path))

    from src.tools.similar_devices import find_similar_devices
    result = find_similar_devices(
        device_class=99,  # no such class
        data_path=str(tmp_path / "clean_data.csv"),
    )
    assert result is None
