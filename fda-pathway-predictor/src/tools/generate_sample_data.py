"""Generate realistic sample FDA data for development and testing."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

SPECIALTIES = {
    "AN": ("Anesthesiology", ["anesthesia machines","ventilators","breathing circuits"]),
    "CV": ("Cardiovascular", ["pacemakers","stents","heart valves","catheters"]),
    "CH": ("Clinical Chemistry", ["blood analyzers","glucose monitors","test strips"]),
    "DE": ("Dental", ["dental implants","orthodontic brackets","dental chairs"]),
    "EN": ("Ear, Nose, Throat", ["hearing aids","cochlear implants","tympanostomy tubes"]),
    "GU": ("Gastroenterology", ["endoscopes","capsule cameras","biopsy forceps"]),
    "HO": ("General Hospital", ["hospital beds","IV pumps","surgical lights"]),
    "IM": ("Immunology", ["allergy test kits","immunoassay systems"]),
    "MI": ("Microbiology", ["culture media","susceptibility test systems"]),
    "NE": ("Neurology", ["EEG systems","nerve stimulators","brain monitors"]),
    "OB": ("Obstetrics/Gynecology", ["fetal monitors","ultrasound probes"]),
    "OP": ("Ophthalmic", ["intraocular lenses","retinal cameras","tonometers"]),
    "OR": ("Orthopedic", ["hip implants","knee replacements","bone screws"]),
    "PA": ("Pathology", ["microscopes","tissue processors","staining systems"]),
    "PM": ("Physical Medicine", ["prosthetics","wheelchairs","rehab devices"]),
    "RA": ("Radiology", ["X-ray machines","MRI systems","CT scanners","ultrasound"]),
    "SU": ("General Surgery", ["surgical robots","electrosurgical units","staplers"]),
    "TX": ("Clinical Toxicology", ["drug test kits","toxicology analyzers"]),
}

COMPANIES = ["Medtronic Inc.","Johnson & Johnson","Abbott Laboratories","Boston Scientific Corp.",
    "Stryker Corporation","Becton Dickinson","Baxter International","Edwards Lifesciences",
    "Zimmer Biomet","Hologic Inc.","Smith & Nephew","Teleflex Inc.","Intuitive Surgical",
    "Philips Healthcare","GE Healthcare","Siemens Healthineers","Olympus Corp.","Cook Medical",
    "Cardinal Health","Danaher Corporation","3M Healthcare","Roche Diagnostics",
    "Small Device Startup A","Small Device Startup B","MedTech Innovation Ltd."]

COUNTRIES = ["US"]*70 + ["DE","JP","GB","FR","IL","CA","CH","IE","NL","AU","CN","KR","SE","DK",
    "IT","BE","AT","SG","IN","BR","TW","FI","NO","ES","CZ","HU","PL","NZ","MX","ZA"]
US_STATES = ["CA","MA","MN","NJ","IN","PA","IL","TX","FL","NY","CT","OH","MI","WI","CO","MD"]

DECISION_CODES_510K = {"SESE":0.85,"SEKD":0.03,"SESD":0.02,"SESI":0.04,"SESK":0.02,"SESP":0.02,"SEKN":0.02}
DECISION_CODES_PMA = {"APPR":0.80,"APWD":0.05,"APCV":0.03,"WTDR":0.07,"DENY":0.05}

def generate_product_code():
    return "".join(np.random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 3))

def _specialty_weights(pathway):
    keys = list(SPECIALTIES.keys())
    weights = np.ones(len(keys))
    idx = {k: i for i, k in enumerate(keys)}
    if pathway == "510k":
        for k in ["RA","OR","SU","CV","CH","GU"]:
            if k in idx: weights[idx[k]] = 3.0
    elif pathway == "PMA":
        for k in ["CV","OR","NE","OP"]:
            if k in idx: weights[idx[k]] = 5.0
        for k in ["CH","MI","TX","PA"]:
            if k in idx: weights[idx[k]] = 0.3
    elif pathway == "De_Novo":
        for k in ["CH","RA","NE","CV","GU"]:
            if k in idx: weights[idx[k]] = 2.0
    return weights / weights.sum()

def _year_weights(start, end):
    years = list(range(start, end + 1))
    weights = np.array([1.0 + 0.05 * i for i in range(len(years))])
    return weights / weights.sum()

def generate_records(n_510k=8000, n_pma=1500, n_denovo=500):
    records = []
    for i in range(n_510k):
        spec = np.random.choice(list(SPECIALTIES.keys()), p=_specialty_weights("510k"))
        spec_name, devices = SPECIALTIES[spec]
        dc = np.random.choice([1,2,3], p=[0.15,0.75,0.10])
        year = np.random.choice(range(1980, 2026), p=_year_weights(1980, 2025))
        month, day = np.random.randint(1,13), np.random.randint(1,29)
        country = np.random.choice(COUNTRIES)
        records.append({
            "submission_id": f"K{year%100:02d}{np.random.randint(1000,9999)}",
            "pathway": "510k", "device_name": np.random.choice(devices)+f" model {np.random.randint(100,999)}",
            "product_code": generate_product_code(), "advisory_committee": spec,
            "advisory_committee_description": spec_name,
            "decision_code": np.random.choice(list(DECISION_CODES_510K.keys()), p=list(DECISION_CODES_510K.values())),
            "decision_date": f"{year}-{month:02d}-{day:02d}",
            "applicant": np.random.choice(COMPANIES), "country_code": country,
            "state": np.random.choice(US_STATES) if country=="US" else None,
            "city": None, "device_class": dc, "medical_specialty": spec,
            "medical_specialty_description": spec_name,
            "regulation_number": f"8{np.random.randint(62,92)}.{np.random.randint(1000,9999)}",
            "date_received": f"{year}-{max(1,month-3):02d}-{day:02d}",
            "clearance_type": np.random.choice(["Traditional","Special","Abbreviated"], p=[0.75,0.15,0.10]),
            "third_party_flag": np.random.choice(["Y","N"], p=[0.15,0.85]),
        })
    for i in range(n_pma):
        spec = np.random.choice(list(SPECIALTIES.keys()), p=_specialty_weights("PMA"))
        spec_name, devices = SPECIALTIES[spec]
        year = np.random.choice(range(1980, 2026), p=_year_weights(1980, 2025))
        month, day = np.random.randint(1,13), np.random.randint(1,29)
        country = np.random.choice(COUNTRIES)
        records.append({
            "submission_id": f"P{year%100:02d}{np.random.randint(1000,9999):04d}",
            "pathway": "PMA", "device_name": np.random.choice(devices)+f" system {np.random.randint(100,999)}",
            "product_code": generate_product_code(), "advisory_committee": spec,
            "advisory_committee_description": spec_name,
            "decision_code": np.random.choice(list(DECISION_CODES_PMA.keys()), p=list(DECISION_CODES_PMA.values())),
            "decision_date": f"{year}-{month:02d}-{day:02d}",
            "applicant": np.random.choice(COMPANIES[:20]), "country_code": country,
            "state": np.random.choice(US_STATES) if country=="US" else None,
            "city": None, "device_class": 3, "medical_specialty": spec,
            "medical_specialty_description": spec_name,
            "regulation_number": f"8{np.random.randint(62,92)}.{np.random.randint(1000,9999)}",
            "date_received": None, "clearance_type": None, "third_party_flag": None,
        })
    for i in range(n_denovo):
        spec = np.random.choice(list(SPECIALTIES.keys()), p=_specialty_weights("De_Novo"))
        spec_name, devices = SPECIALTIES[spec]
        dc = np.random.choice([1,2], p=[0.30,0.70])
        year = np.random.choice(range(1997, 2026), p=_year_weights(1997, 2025))
        month, day = np.random.randint(1,13), np.random.randint(1,29)
        country = np.random.choice(COUNTRIES)
        records.append({
            "submission_id": f"DEN{year%100:02d}{np.random.randint(1000,9999)}",
            "pathway": "De_Novo", "device_name": np.random.choice(devices)+f" gen {np.random.randint(1,9)}",
            "product_code": generate_product_code(), "advisory_committee": spec,
            "advisory_committee_description": spec_name,
            "decision_code": np.random.choice(["DENG","DENY"], p=[0.88,0.12]),
            "decision_date": f"{year}-{month:02d}-{day:02d}",
            "applicant": np.random.choice(COMPANIES), "country_code": country,
            "state": np.random.choice(US_STATES) if country=="US" else None,
            "city": None, "device_class": dc, "medical_specialty": spec,
            "medical_specialty_description": spec_name,
            "regulation_number": f"8{np.random.randint(62,92)}.{np.random.randint(1000,9999)}",
            "date_received": None, "clearance_type": None, "third_party_flag": None,
        })
    df = pd.DataFrame(records)
    # Add noise: 5% missing product_code, 3% missing advisory_committee, 1% near-dupes
    n = len(df)
    df.loc[np.random.random(n)<0.05, "product_code"] = None
    df.loc[np.random.random(n)<0.03, "advisory_committee"] = None
    df.loc[np.random.random(n)<0.02, "applicant"] = None
    # Some missing device_class but with SESE decision code (to test imputation)
    mask_sese = (df["decision_code"]=="SESE") & (np.random.random(n)<0.03)
    df.loc[mask_sese, "device_class"] = None
    n_dupes = int(n * 0.01)
    dupes = df.iloc[np.random.choice(n, n_dupes, replace=False)].copy()
    dupes["submission_id"] = dupes["submission_id"].apply(lambda x: x+"A" if x else x)
    df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    df = generate_records()
    df.to_csv("artifacts/raw_data.csv", index=False)
    print(f"Generated {len(df)} records\nPathways:\n{df['pathway'].value_counts()}")
