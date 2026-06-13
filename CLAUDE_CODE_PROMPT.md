# FDA Regulatory Pathway Predictor — Project Context for Claude Code

## COURSE REQUIREMENTS (from teacher's brief)

This is a **final project for an AI Developer course**. The deliverables are:
1. Repository with source code, artifacts, and documentation
2. Business presentation (10-12 slides) — DONE ✅
3. Short demo video (≤5 min) — TODO

### What the teacher expects:
- A **CrewAI Flow** coordinating two separate multi-agent crews
- **Crew 1 — Data Analyst** (at least 3 agents): ingest, clean, EDA, produce clean_data.csv, eda_report.html, insights.md, dataset_contract.json
- **Crew 2 — Data Scientist** (at least 3 agents): feature engineering, train 2+ model variations, produce features.csv, model.pkl, evaluation_report.md, model_card.md
- Flow must automate handoff, include validation steps, support reproducibility, fail gracefully
- **Required stack:** CrewAI, Python, Git+GitHub (with PRs), Streamlit or Flask, Pandas, scikit-learn, Matplotlib/Seaborn
- **Git:** Must use branches and Pull Requests (teacher checks PR history)
- Optional deployment to Streamlit Cloud or Railway

---

## PROJECT OVERVIEW

**FDA Regulatory Pathway Predictor** — an ML pipeline that predicts which FDA regulatory pathway (510(k), PMA, or De Novo) a medical device should go through, based on device characteristics.

**User persona:** Sarah, 35, RA Manager at a 50-person medtech startup. She needs to quickly determine the likely FDA pathway before committing months of work and budget.

**Data source:** openFDA API (api.fda.gov) — public government data:
- /device/510k.json — ~174,000 510(k) clearances (De Novo identified by "DEN" prefix)
- /device/pma.json — ~30,000 PMA approvals  
- /device/classification.json — ~6,000 device classifications (enrichment)
- Merge key: product_code (3-letter code shared across endpoints)
- API limit: 1,000 records per query — pagination by date range needed

---

## ARCHITECTURE

```
openFDA API → [Data Ingestor] → [Data Cleaner] → [EDA Analyst]
                                                        │
                                                 Validation Gate
                                                        │
              [Model Evaluator] ← [Model Trainer] ← [Feature Engineer]
                      │
                [Streamlit App]
```

### Crew 1 — Data Analyst (3 agents):
1. **Data Ingestor** — fetches from 3 openFDA endpoints with pagination, merges into unified dataset → raw_data.csv
2. **Data Cleaner** — deduplicates, handles missing values, standardizes types, generates dataset contract → clean_data.csv + dataset_contract.json
3. **EDA Analyst** — generates 7 chart types, produces HTML report and insights → eda_report.html + insights.md

### Validation Gate:
- Checks dataset_contract.json against clean_data.csv
- Verifies target variable exists, classes match, minimum record count, no-null constraints

### Crew 2 — Data Scientist (3 agents):
4. **Feature Engineer** — frequency encoding, label encoding, temporal features, boolean flags (20 features total) → features.csv + label_encoders.json
5. **Model Trainer** — Random Forest (200 trees, balanced) vs Gradient Boosting (200 estimators). 5-fold CV → model.pkl
6. **Model Evaluator** — accuracy, F1-macro, precision, recall, ROC-AUC, confusion matrices, feature importance → evaluation_report.md + model_card.md

### Streamlit App (3 pages):
- Page 1: Pathway Predictor — enter device details, get recommendation with confidence scores
- Page 2: EDA Dashboard — embedded charts and insights
- Page 3: Model Performance — evaluation metrics, confusion matrices, model card

---

## PROJECT STRUCTURE

```
fda-pathway-predictor/
├── src/
│   ├── crews/
│   │   ├── analyst_crew.py      # Crew 1: 3 data analyst agents with custom tools
│   │   └── scientist_crew.py    # Crew 2: 3 data scientist agents with custom tools
│   ├── flow/
│   │   └── main_flow.py         # CrewAI Flow orchestration + direct-mode runner
│   ├── tools/
│   │   ├── fda_api_tool.py      # openFDA API ingestion with pagination
│   │   ├── generate_sample_data.py  # Realistic sample data when API unavailable
│   │   ├── cleaning_tool.py     # Data cleaning + dataset contract generation
│   │   ├── eda_tool.py          # EDA charts + insights
│   │   ├── feature_tool.py      # Feature engineering (20 features)
│   │   └── model_tool.py        # Model training + evaluation + model card
│   └── app/
│       └── streamlit_app.py     # 3-page Streamlit dashboard
├── artifacts/                   # All pipeline outputs (auto-generated)
├── tests/
├── .cursorrules
├── .gitignore
├── requirements.txt
└── README.md
```

---

## KEY TECHNICAL DECISIONS

1. **decision_code_encoded is EXCLUDED from model features** — it leaks the target variable (SESE is 510k-specific, APPR is PMA-specific)
2. Class imbalance handled with `class_weight="balanced"` in Random Forest
3. Frequency encoding for high-cardinality categoricals (product_code, applicant)
4. Cyclical sin/cos encoding for month
5. Pipeline has two modes: CrewAI Flow (needs LLM API key) and direct mode (no LLM)
6. When FDA API is unreachable, falls back to generate_sample_data.py

---

## GIT STRATEGY (7 feature branches with PRs)

```
main
 ├── feature/project-setup        → PR #1: .gitignore, README, requirements.txt, __init__.py files
 ├── feature/data-ingestion       → PR #2: fda_api_tool.py, generate_sample_data.py
 ├── feature/data-cleaning-eda    → PR #3: cleaning_tool.py, eda_tool.py
 ├── feature/feature-engineering  → PR #4: feature_tool.py
 ├── feature/model-training       → PR #5: model_tool.py
 ├── feature/crewai-flow          → PR #6: analyst_crew.py, scientist_crew.py, main_flow.py
 └── feature/streamlit-app        → PR #7: streamlit_app.py
```

Each PR should have a description: What was added, Why, What artifacts it produces, How to test.

---

## PROJECT PATH & ENVIRONMENT

- **Project folder:** `C:\Users\30010\V2_FDA_PW_TR\`
- **GitHub repo:** `https://github.com/300106-gif/fda-pathway-predictor-v2`
- **Virtual environment:** `.venv` exists in project root — already set up ✅
- **All dependencies installed ✅:** requests, pandas, scikit-learn, matplotlib, seaborn, joblib, streamlit, numpy, uv, crewai, python-dotenv, jupyter

### Activate venv (run this every time you open a new terminal):
```powershell
cd C:\Users\30010\V2_FDA_PW_TR
.\.venv\Scripts\Activate.ps1
```

---

## IMMEDIATE TASKS

The project files are in this folder. Environment is ready. Next steps:

1. **Test FDA API** — verify `https://api.fda.gov/device/510k.json?limit=1` returns data
2. **Run the full pipeline** — `python -m src.flow.main_flow` (will use sample data if API fails)
3. **Create test.ipynb** — Jupyter notebook for interactive data exploration and model testing
4. **Set up Git & GitHub** — initialize repo, create GitHub remote, start making PRs per the branch strategy below
5. **Update Streamlit app** — implement the two-stage prediction, similar device lookup, and year filter per the UI/UX spec below
6. **Launch Streamlit** — `streamlit run src/app/streamlit_app.py`
7. **Deploy to Streamlit Cloud** — connect GitHub repo, deploy

---

## RUNNING THE PROJECT

```powershell
# Activate venv first
cd C:\Users\30010\V2_FDA_PW_TR
.\.venv\Scripts\Activate.ps1

# Run full pipeline (direct mode, no LLM key needed)
python -m src.flow.main_flow

# Run with CrewAI agents (needs LLM key)
$env:OPENAI_API_KEY="your-key-here"
python -m src.flow.main_flow

# Launch Streamlit app
streamlit run src/app/streamlit_app.py

# Launch Jupyter notebook
jupyter notebook
```

## ALL REQUIRED ARTIFACTS (13 files in artifacts/)

Crew 1: raw_data.csv, clean_data.csv, dataset_contract.json, eda_report.html, insights.md
Crew 2: features.csv, label_encoders.json, model_meta.json, model.pkl, evaluation_report.md, model_card.md, confusion_matrices.png, feature_importance.png

---

## STREAMLIT APP — DETAILED UI/UX SPECIFICATION

The Streamlit app (`src/app/streamlit_app.py`) must have 3 pages. Below is the exact UI specification for each. Implement these using Streamlit components (st.columns, st.metric, st.bar_chart, st.markdown with HTML, st.selectbox, st.text_input, st.text_area, st.checkbox, st.button, st.expander, etc.).

### PAGE 1 — Device Input (Pathway Predictor)

**Layout:** Two-column layout. Left = input form (wider), Right = info cards (narrower).

**Left column — Input Form:**
- Title: "FDA Pathway Advisor"
- Subtitle: "Enter your device details to get a regulatory pathway recommendation."
- "Device Name" — text input, placeholder "e.g. Wireless cardiac rhythm monitor"
- "Device Description" — large text area, placeholder "Describe your device's intended use, technology, and how it works"
- "Medical Specialty" — dropdown with options: Cardiovascular, Radiology, Orthopedic, Neurology, General Surgery, Dental, Ophthalmic, Clinical Chemistry, Ear/Nose/Throat, Gastroenterology, Anesthesiology, General Hospital, Immunology, Microbiology, Obstetrics/Gynecology, Pathology, Physical Medicine, Clinical Toxicology
- "Device Risk Profile" — three selectable cards in a row: "Class I - Low Risk" (green dot), "Class II - Moderate Risk" (yellow dot), "Class III - High Risk" (red dot). Default: Class II selected.
- "Country of Origin" — dropdown: United States, Germany, Japan, United Kingdom, France, Israel, Canada, Switzerland, Ireland, Other
- Checkbox: "This device has a substantially equivalent predicate device already on the market"
- Large blue button: "Analyze Pathway"

**Right column — Info Cards:**
- Card 1 heading "What is this tool?" with text: "This tool uses machine learning trained on 200,000+ historical FDA submissions to predict the most likely regulatory pathway for your medical device."
- Card 2 heading "Supported Pathways" listing: blue dot "510(k) - Premarket Notification", orange dot "PMA - Premarket Approval", green dot "De Novo - Classification Request"

**Sidebar navigation:**
- Small logo/icon at top
- Nav items: "New Analysis" (active), "How It Works", "About the Model"

### PAGE 2 — Results (shown after clicking Analyze Pathway)

**Top section:**
- Back arrow + breadcrumb: "New Analysis > Results"
- Large banner card with blue background:
  - "Recommended Pathway: 510(k)" in large white text
  - "87% confidence" in a white badge
  - Small text: "Based on analysis of 200,000+ historical FDA submissions"

**Three-column layout below banner:**

- **Left column — "Confidence Breakdown":**
  Three horizontal bars: 510(k) at 87% (blue), PMA at 8% (orange), De Novo at 5% (green)

- **Middle column — "Why This Pathway?":**
  Top 5 factors with relative importance bars:
  1. Device Class II
  2. Cardiovascular specialty
  3. US-based manufacturer
  4. Has predicate device
  5. Traditional clearance type
  (These come from model.feature_importances_ mapped to the user's input values)

- **Right column — "What This Means":**
  Info card with pathway explanation. Content varies by predicted pathway:
  - 510(k): "Requires demonstrating substantial equivalence to a legally marketed predicate device. Typical review: 3-6 months. Clinical trials usually not required. FDA user fee: ~$22K."
  - PMA: "Requires valid scientific evidence of safety and effectiveness. Typical review: 12-18 months. Clinical trials usually required. FDA user fee: ~$420K."
  - De Novo: "For novel low/moderate risk devices with no predicate. Creates new classification. Typical review: 6-12 months. FDA user fee: ~$130K."

**Bottom action buttons (row of 3):**
- "View Preparation Checklist" → navigates to Page 3
- "Download Report" → generates a summary PDF or markdown
- "Start New Analysis" → clears form, returns to Page 1

### PAGE 3 — Preparation Checklist (stretch goal / nice-to-have)

**Title:** "[Pathway] Preparation Checklist" (dynamic based on predicted pathway)
**Subtitle:** "Steps to prepare your premarket submission."

**Vertical stepper layout with 6 steps** (content varies by pathway, below is 510(k)):

1. **"Classify Your Device"** — green checkmark (auto-completed by our tool)
   Description: "Product classification identified via FDA Product Classification Database"
   Link: "View classification database" → https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm

2. **"Identify Predicate Device"** — empty circle
   Description: "Find a legally marketed device with the same intended use and similar technological characteristics."

3. **"Prepare Comparison"** — empty circle
   Description: "Document a comparison of your device to the predicate including intended use, technology, and performance."

4. **"Conduct Performance Testing"** — empty circle
   Description: "Complete bench testing, biocompatibility testing, and other non-clinical tests as applicable. Follow Good Laboratory Practices (21 CFR 58)."

5. **"Prepare Labeling"** — empty circle
   Description: "Draft device labeling per 21 CFR 801 including indications for use, warnings, and instructions."

6. **"Submit via eSTAR"** — empty circle
   Description: "Submit your 510(k) electronically through the FDA CDRH eSTAR portal."

**Bottom notices:**
- Yellow warning card: "Even if your device does not require a premarket submission, you must comply with applicable regulatory controls, establishment registration, and device listing."
- Two side-by-side notice cards:
  - "Radiation-emitting product? Also comply with 21 CFR parts 1000-1050" → link to https://www.fda.gov/radiation-emitting-products
  - "Combination product? Contact FDA's Office of Combination Products at combination@fda.gov" → link to https://www.fda.gov/combination-products/about-combination-products/frequently-asked-questions-about-combination-products

### DESIGN SYSTEM
- Clean white background
- Subtle card shadows (box-shadow)
- Color scheme: Blue (#378ADD) for 510(k), Orange (#D85A30) for PMA, Green (#1D9E75) for De Novo
- Professional medical/healthcare aesthetic
- Consistent card styling across all pages
- Gray disclaimer text at bottom of results: "This prediction is based on historical FDA data patterns. It should not replace professional regulatory advice."

---

## TWO-STAGE PREDICTION ARCHITECTURE

The prediction system uses a two-stage approach. This is critical to implement correctly.

### Stage 1: Rule-Based Exemption Check (runs FIRST)

Before the ML model runs, query the openFDA classification endpoint to check if the device is exempt from premarket submission. Many Class I and some Class II devices are 510(k)-exempt and don't need any premarket submission.

```python
import requests

def check_exemption(product_code=None, device_name=None):
    """
    Check FDA classification database for exemption status.
    Returns classification info including whether device is exempt.
    """
    base_url = "https://api.fda.gov/device/classification.json"
    
    if product_code:
        search = f"product_code:{product_code}"
    elif device_name:
        search = f"device_name:{device_name}"
    else:
        return None
    
    try:
        r = requests.get(f"{base_url}?search={search}&limit=5", timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0]  # Best match
    except Exception:
        pass
    return None


def determine_pathway(device_info, model, feature_cols):
    """
    Two-stage pathway determination:
    Stage 1: Check if device is exempt (rule-based)
    Stage 2: ML prediction for non-exempt devices
    """
    
    # ── STAGE 1: Exemption check ──
    classification = check_exemption(
        product_code=device_info.get("product_code"),
        device_name=device_info.get("device_name"),
    )
    
    if classification:
        device_class = str(classification.get("device_class", ""))
        # Check submission_type_id — if empty or exempt, no submission needed
        submission_type = classification.get("submission_type_id", "")
        
        # GMP exempt flag is also informative
        gmp_exempt = classification.get("gmp_exempt_flag", "")
        
        # Class I devices with no submission type = exempt
        if device_class == "1" and submission_type.strip() == "":
            return {
                "pathway": "Exempt",
                "confidence": 1.0,
                "device_class": 1,
                "product_code": classification.get("product_code", ""),
                "regulation_number": classification.get("regulation_number", ""),
                "message": (
                    "This is a Class I exempt device. No premarket submission "
                    "(510(k), PMA, or De Novo) is required. However, you must "
                    "still comply with:\n"
                    "• General controls (21 CFR Part 820 — Quality System Regulation)\n"
                    "• Establishment registration (21 CFR Part 807)\n"
                    "• Device listing (21 CFR Part 807)\n"
                    "• Labeling requirements (21 CFR Part 801)\n"
                    "• Medical Device Reporting (21 CFR Part 803)\n\n"
                    "Note: If your device exceeds the limitations of exemption "
                    "stated in 21 CFR xxx.9, a 510(k) may still be required."
                ),
            }
        
        # Class II exempt (less common but exists)
        if device_class == "2" and submission_type.strip() == "":
            return {
                "pathway": "Exempt",
                "confidence": 1.0,
                "device_class": 2,
                "product_code": classification.get("product_code", ""),
                "message": (
                    "This is a Class II exempt device. No 510(k) is required "
                    "unless your device exceeds the limitations of exemption "
                    "stated in 21 CFR xxx.9. You must still comply with general "
                    "controls AND any applicable special controls."
                ),
            }
        
        # If classification found but not exempt, enrich device_info with
        # the actual device class and specialty for better ML prediction
        device_info["device_class"] = int(device_class) if device_class.isdigit() else 2
        device_info["medical_specialty"] = classification.get("medical_specialty", "")
        device_info["product_code"] = classification.get("product_code", "")
    
    # ── STAGE 2: ML prediction for non-exempt devices ──
    features = build_feature_vector(device_info)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    pathway_map = {0: "510k", 1: "De_Novo", 2: "PMA"}  # from label_encoders.json
    predicted_pathway = pathway_map.get(prediction, "510k")
    
    return {
        "pathway": predicted_pathway,
        "confidence": float(max(probabilities)),
        "probabilities": {
            "510k": float(probabilities[0]) if len(probabilities) > 0 else 0,
            "De_Novo": float(probabilities[1]) if len(probabilities) > 1 else 0,
            "PMA": float(probabilities[2]) if len(probabilities) > 2 else 0,
        },
        "device_class": device_info.get("device_class"),
        "message": None,  # Streamlit app fills in pathway-specific guidance
    }
```

### Stage 2: ML Model (runs only for non-exempt devices)

The ML model predicts 510(k) vs PMA vs De Novo based on 18 features learned from 200K+ historical submissions. See FEATURE_COLS in model_tool.py.

### How This Changes the Streamlit App

In `streamlit_app.py` Page 2 (Results), add a fourth result type for "Exempt":

- If pathway == "Exempt": show a GREEN banner saying "No Premarket Submission Required"
  - Display the exemption message with compliance requirements
  - Show a checklist for general controls, establishment registration, device listing
  - Link to https://www.fda.gov/medical-devices/classify-your-medical-device/class-i-and-class-ii-device-exemptions
  
- If pathway == "510k" / "PMA" / "De_Novo": show the normal results with confidence scores

### Additional Notices in Results Page

After showing any result (including Exempt), check and display these notices:

**Radiation-emitting products:**
If the device could be radiation-emitting, show: "If your device is also a radiation-emitting electronic product, it must comply with 21 CFR parts 1000-1050."
Link: https://www.fda.gov/radiation-emitting-products

**Combination products:**
Show a notice: "If your product combines a medical device with a drug, biologic, or other FDA-regulated product, contact FDA's Office of Combination Products at combination@fda.gov"
Link: https://www.fda.gov/combination-products/about-combination-products/frequently-asked-questions-about-combination-products

**General compliance reminder (always shown):**
"Even if your device does not require a premarket submission, you must identify the correct classification and comply with applicable regulatory controls."
Link: https://www.fda.gov/medical-devices/overview-device-regulation/regulatory-controls

---

## SIMILAR DEVICE LOOKUP (Evidence Feature)

After the ML model predicts the pathway, show the user historical evidence from similar devices. This uses decision_code as USER-FACING EVIDENCE, not as a model feature.

### Implementation

Add this function to `src/tools/similar_devices.py`:

```python
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
```

### Streamlit Integration (Page 2 — Results)

After showing the ML prediction and confidence scores, add a section:

**"Similar Devices in FDA History"**

Display:
- "We found {total_similar} similar devices in the FDA database"
- Pathway breakdown: "94% went through 510(k), 4% through PMA, 2% through De Novo"
- Success rate: "Of those that went through 510(k), 97% received SESE (Substantially Equivalent) clearance"
- Expandable table showing the top 10 most recent similar devices with their submission_id, device_name, pathway, decision_code, applicant, and decision_date

Example UI text for a Class II Cardiovascular device:

```
📊 Similar Devices in FDA History
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Found 847 similar devices (Class II, Cardiovascular)

Pathway breakdown:
  510(k)  ████████████████████████ 94.2%  (798 devices)
  PMA     ██                       3.8%  (32 devices)
  De Novo █                        2.0%  (17 devices)

Historical clearance rate:
  ✅ 97.1% of similar 510(k) submissions received SESE clearance

Recent examples:
  K241234  Cardiac Monitor Pro    510(k)  SESE  Medtronic       2024-08-15
  K240987  HeartSync Wireless     510(k)  SESE  Abbott Labs     2024-07-22
  K240543  CardioTrack System     510(k)  SESE  Boston Sci      2024-06-10
  ...
```

### Why This Matters

This feature transforms the tool from "the model says 510(k)" into "the model says 510(k), AND here's the evidence: 847 similar devices went through 510(k) with a 97% clearance rate." The decision_code (SESE, APPR, DENG) is shown as evidence to the user, not fed to the model. This is the proper way to use decision_code — transparency, not leakage.

---

## JUPYTER NOTEBOOK FOR TESTING (test.ipynb)

Create a `test.ipynb` notebook in the project root for interactive data exploration and model testing. This notebook lets the student inspect DataFrames, run predictions, and validate the pipeline step by step.

### Notebook Structure

**Cell 1 — Setup & Imports:**
```python
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 120)

ARTIFACTS = Path("artifacts")
print("Artifacts available:", [f.name for f in ARTIFACTS.glob("*")] if ARTIFACTS.exists() else "Run pipeline first!")
```

**Cell 2 — Load & Explore Raw Data:**
```python
df_raw = pd.read_csv(ARTIFACTS / "raw_data.csv")
print(f"Raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"\nPathway distribution:\n{df_raw['pathway'].value_counts()}")
print(f"\nColumns: {list(df_raw.columns)}")
df_raw.head(10)
```

**Cell 3 — Inspect Missing Values & Device Class Issues:**
```python
# Find records where device_class is missing
missing_class = df_raw[df_raw["device_class"].isna()]
print(f"Records with missing device_class: {len(missing_class)}")

# Cross-tab: missing device_class vs decision_code
if len(missing_class) > 0:
    print(f"\nDecision codes for missing device_class:")
    print(missing_class["decision_code"].value_counts().head(10))
    
    # How many are SESE (can be repaired)?
    sese_missing = missing_class[missing_class["decision_code"] == "SESE"]
    print(f"\nSESE with missing class: {len(sese_missing)} (can be repaired to Class II)")
```

**Cell 4 — Load & Explore Clean Data:**
```python
df_clean = pd.read_csv(ARTIFACTS / "clean_data.csv")
print(f"Clean data: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"\nNull counts:\n{df_clean.isnull().sum()}")
print(f"\nDevice class distribution:\n{df_clean['device_class'].value_counts()}")
df_clean.head(10)
```

**Cell 5 — Inspect Dataset Contract:**
```python
with open(ARTIFACTS / "dataset_contract.json") as f:
    contract = json.load(f)

print(f"Target variable: {contract['target_variable']}")
print(f"Target classes: {contract['target_classes']}")
print(f"Total records: {contract['total_records']}")
print(f"\nRecommended features: {contract['recommended_features']}")
print(f"\nClass distribution: {contract['constraints']['class_distribution']}")
```

**Cell 6 — Load & Explore Features:**
```python
df_features = pd.read_csv(ARTIFACTS / "features.csv")
print(f"Features: {df_features.shape[0]} rows × {df_features.shape[1]} columns")

feature_cols = [c for c in df_features.columns if c not in ["submission_id", "pathway", "pathway_encoded"]]
print(f"\n{len(feature_cols)} model features: {feature_cols}")
print(f"\nFeature statistics:")
df_features[feature_cols].describe()
```

**Cell 7 — Load Model & Encoders:**
```python
model = joblib.load(ARTIFACTS / "model.pkl")
print(f"Model type: {type(model).__name__}")
print(f"Model parameters: {model.get_params()}")

with open(ARTIFACTS / "label_encoders.json") as f:
    encoders = json.load(f)
print(f"\nPathway encoding: {encoders['pathway']}")
print(f"Pathway inverse: {encoders['pathway_inverse']}")

with open(ARTIFACTS / "model_meta.json") as f:
    meta = json.load(f)
print(f"\nBest model: {meta['best_model']}")
print(f"Feature columns: {meta['feature_columns']}")
```

**Cell 8 — Test a Single Prediction:**
```python
# Simulate a user entering a device
test_device = {
    "device_class": 2,
    "advisory_committee_freq": 0.12,
    "advisory_committee_encoded": 3,
    "medical_specialty_freq": 0.12,
    "medical_specialty_encoded": 3,
    "is_us": 1,
    "country_freq": 0.70,
    "decision_year": 2025,
    "decision_month": 6,
    "month_sin": np.sin(2 * np.pi * 6 / 12),
    "month_cos": np.cos(2 * np.pi * 6 / 12),
    "review_days": df_features["review_days"].median(),
    "has_review_days": 1,
    "clearance_type_encoded": 0,
    "third_party": 0,
    "product_code_freq": 0.01,
    "applicant_freq": 0.02,
    "applicant_submission_count": 50,
}

X_test = pd.DataFrame([test_device])
X_test = X_test.reindex(columns=meta["feature_columns"], fill_value=0)

prediction = model.predict(X_test)[0]
probabilities = model.predict_proba(X_test)[0]

pathway_name = encoders["pathway_inverse"][str(prediction)]
print(f"Predicted pathway: {pathway_name}")
print(f"Confidence: {max(probabilities)*100:.1f}%")
print(f"\nProbabilities:")
for class_name, idx in encoders["pathway"].items():
    print(f"  {class_name}: {probabilities[idx]*100:.1f}%")
```

**Cell 9 — Test Similar Device Lookup:**
```python
# Find similar devices to our test case
mask = (df_clean["device_class"] == 2) & (df_clean["advisory_committee"] == "CV")
similar = df_clean[mask]
print(f"Found {len(similar)} similar devices (Class II, Cardiovascular)")

# Pathway breakdown
print(f"\nPathway distribution:")
pathway_pct = (similar["pathway"].value_counts() / len(similar) * 100).round(1)
for pathway, pct in pathway_pct.items():
    print(f"  {pathway}: {pct}%")

# Decision code breakdown
print(f"\nDecision codes:")
decision_pct = (similar["decision_code"].value_counts() / len(similar) * 100).round(1)
for code, pct in decision_pct.head(5).items():
    print(f"  {code}: {pct}%")

# SESE success rate for 510(k) submissions
k510_similar = similar[similar["pathway"] == "510k"]
if len(k510_similar) > 0:
    sese_rate = (k510_similar["decision_code"] == "SESE").mean() * 100
    print(f"\nSESE clearance rate for 510(k): {sese_rate:.1f}%")
```

**Cell 10 — Test Exemption Check (requires internet):**
```python
import requests

def check_exemption(device_name):
    """Query openFDA classification for exemption status."""
    url = f"https://api.fda.gov/device/classification.json?search=device_name:{device_name}&limit=3"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            for result in results:
                print(f"\nDevice: {result.get('device_name', 'N/A')}")
                print(f"  Product code: {result.get('product_code', 'N/A')}")
                print(f"  Device class: {result.get('device_class', 'N/A')}")
                print(f"  Submission type: '{result.get('submission_type_id', '')}'")
                print(f"  GMP exempt: {result.get('gmp_exempt_flag', 'N/A')}")
                
                if result.get("device_class") == "1" and result.get("submission_type_id", "").strip() == "":
                    print(f"  ✅ EXEMPT — No premarket submission required")
                else:
                    print(f"  ❌ NOT EXEMPT — Premarket submission required")
            return results
        else:
            print(f"API error: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    return None

# Test with a Class I exempt device
print("=== Testing Class I Exempt Device ===")
check_exemption("bandage")

print("\n=== Testing Class III Device ===")
check_exemption("pacemaker")
```

**Cell 11 — Feature Importance Analysis:**
```python
import matplotlib.pyplot as plt

if hasattr(model, "feature_importances_"):
    importance = pd.Series(
        model.feature_importances_,
        index=meta["feature_columns"]
    ).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title("Feature Importance — What Drives the Prediction?")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    print("\nTop 5 features:")
    for feat, imp in importance.tail(5).iloc[::-1].items():
        print(f"  {feat}: {imp:.4f}")
```

**Cell 12 — Full Pipeline Test (run all steps):**
```python
# Run the complete pipeline and verify all artifacts
print("Running full pipeline...")
from src.flow.main_flow import run_pipeline_without_llm
model, results = run_pipeline_without_llm()

print("\n\nFinal Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    
print(f"\nAll artifacts:")
for f in sorted(ARTIFACTS.glob("*")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
```

---

## UX/UI MOCKUPS — STITCH PROMPTS (stitch.withgoogle.com)

Generate these mockups in Google Stitch to visualize the app before coding. Paste each prompt separately into Stitch.

### Stitch Prompt 1 — Device Input Page:

A clean web app page titled "FDA Pathway Advisor". Subtitle text "Enter your device details to get a regulatory pathway recommendation." Left side has a form with these fields stacked vertically: "Device Name" text input with placeholder "e.g. Wireless cardiac rhythm monitor", "Device Description" large text area with placeholder "Describe your device's intended use, technology, and how it works", "Medical Specialty" dropdown showing options like Cardiovascular, Radiology, Orthopedic, Neurology, "Device Risk Profile" three selectable cards in a row showing "Class I - Low Risk" with a green dot, "Class II - Moderate Risk" with a yellow dot, "Class III - High Risk" with a red dot, with Class II selected. Below that "Country of Origin" dropdown showing "United States", and a checkbox "This device has a substantially equivalent predicate device already on the market". At the bottom a large blue button "Analyze Pathway". Right side has a small info card with the heading "What is this tool?" and text "This tool uses machine learning trained on 200,000+ historical FDA submissions to predict the most likely regulatory pathway for your medical device." Below that another info card "Supported Pathways" listing three items with colored dots: blue "510(k) - Premarket Notification", orange "PMA - Premarket Approval", green "De Novo - Classification Request". Minimal white design with light gray sidebar containing a small logo and nav items: "New Analysis", "How It Works", "About the Model".

### Stitch Prompt 2 — Results Page:

A web app results page for an FDA regulatory pathway analysis tool. Top section has a back arrow and breadcrumb "New Analysis > Results". Below that a large banner card with blue background showing "Recommended Pathway: 510(k)" in large white text, with "87% confidence" in a white badge, and small text "Based on analysis of 200,000+ historical FDA submissions". Below the banner, three columns. Left column titled "Confidence Breakdown" shows three horizontal bars: 510(k) at 87% in blue, PMA at 8% in orange, De Novo at 5% in green. Middle column titled "Why This Pathway?" shows a vertical list of the top 5 factors that influenced the prediction, each with a small bar showing relative importance: "Device Class II", "Cardiovascular specialty", "US-based manufacturer", "Has predicate device", "Traditional clearance type". Right column titled "What This Means" shows an info card explaining "Your device most closely matches the 510(k) Premarket Notification pathway. This requires demonstrating substantial equivalence to a legally marketed predicate device." Below that key facts: "Typical review time: 3-6 months", "Clinical trials: Usually not required", "FDA user fee: $21,760 (2024)". Below that a section "Similar Devices in FDA History" showing "Found 847 similar devices (Class II, Cardiovascular)" with pathway breakdown bars and text "97% of similar 510(k) submissions received SESE clearance". At the very bottom a row of three action buttons: "View Preparation Checklist", "Download Report", "Start New Analysis". Clean white background with subtle card shadows.

### Stitch Prompt 3 — Preparation Checklist:

A web app checklist page titled "510(k) Preparation Checklist". Subtitle "Steps to prepare your premarket notification submission." A vertical stepper layout with 6 steps, each step is a card with a circle number, title, description, and status. Step 1 "Classify Your Device" with green checkmark, description "Product classification identified via FDA Product Classification Database", a small link "View classification database". Step 2 "Identify Predicate Device" with empty circle, description "Find a legally marketed device with the same intended use and similar technological characteristics." Step 3 "Prepare Comparison" with empty circle, description "Document a comparison of your device to the predicate including intended use, technology, and performance." Step 4 "Conduct Performance Testing" with empty circle, description "Complete bench testing, biocompatibility testing, and other non-clinical tests as applicable. Follow Good Laboratory Practices (21 CFR 58)." Step 5 "Prepare Labeling" with empty circle, description "Draft device labeling per 21 CFR 801 including indications for use, warnings, and instructions." Step 6 "Submit via eSTAR" with empty circle, description "Submit your 510(k) electronically through the FDA CDRH eSTAR portal." At the bottom a yellow notice card with warning icon: "Even if your device does not require a premarket submission, you must comply with applicable regulatory controls, establishment registration, and device listing." Below that two small notice cards side by side: one saying "Radiation-emitting product? Also comply with 21 CFR parts 1000-1050" with a link, and another saying "Combination product? Contact FDA's Office of Combination Products at combination@fda.gov" with a link. Clean white design matching the previous screens.

### Stitch Prompt 4 — Exempt Device Result (new screen):

A web app results page for an FDA regulatory pathway tool showing an exempt device result. Top section has a back arrow and breadcrumb "New Analysis > Results". Below that a large banner card with green background showing "No Premarket Submission Required" in large white text, with "Class I Exempt" in a white badge, and small text "This device is exempt from 510(k) premarket notification requirements." Below the banner, two columns. Left column titled "What This Means" shows an info card explaining "Your device is classified as Class I Exempt. You do not need to submit a 510(k), PMA, or De Novo request to the FDA. However, you must still comply with the following regulatory requirements." Below that a checklist with 5 items each with a checkbox: "General Controls (21 CFR Part 820 — Quality System Regulation)", "Establishment Registration (21 CFR Part 807)", "Device Listing (21 CFR Part 807)", "Labeling Requirements (21 CFR Part 801)", "Medical Device Reporting (21 CFR Part 803)". Right column titled "Device Classification" shows the device details: product code, device class, regulation number, and a link "View in FDA Classification Database". Below that a yellow warning card: "If your device exceeds the limitations of exemption stated in 21 CFR xxx.9, a 510(k) may still be required. Review the specific exemption limitations for your device classification." At the bottom two action buttons: "View Regulatory Controls" and "Start New Analysis". Clean white background matching previous screens with green accent color for exempt status.

---

## DATA RANGE: 1980-2025

### IMPORTANT: Change all references from 2015-2025 to 1980-2025.

The pipeline must fetch FDA records starting from **1980**, not 2015. This gives us 45 years of historical submissions — the full depth of the FDA's digital records.

**Why this matters:**
- 510(k) records exist from the late 1970s
- PMA records exist from the 1980s
- De Novo pathway was created in 1997 (FDAMA) but became more widely used after 2012 (FDASIA)
- More data = better model + richer EDA showing long-term regulatory trends

### Files to update:

**1. `src/tools/fda_api_tool.py`** — change default start_year:
```python
def ingest_fda_data(output_path="artifacts/raw_data.csv", start_year=1980, end_year=2025):
```

**2. `src/tools/generate_sample_data.py`** — extend sample data range:
```python
year = np.random.choice(range(1980, 2026), p=_year_weights(1980, 2025))
```

**3. `src/flow/main_flow.py`** — update both modes:
```python
# In run_pipeline_without_llm and IngestionTool
start_year=1980, end_year=2025
```

**4. `.env`** — update defaults:
```
DATA_START_YEAR=1980
DATA_END_YEAR=2025
```

### EDA Year Filter

**5. `src/tools/eda_tool.py`** — the "Submissions Over Time" chart will now span 1980-2025, showing the growth of FDA submissions over 45 years.

**6. `src/app/streamlit_app.py`** — add a year range slider on the EDA Dashboard page:

```python
# On the EDA Dashboard page, add a year filter
st.subheader("Filter by Year Range")
min_year = int(df_clean["decision_year"].min())
max_year = int(df_clean["decision_year"].max())
year_range = st.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

# Filter data
df_filtered = df_clean[
    (df_clean["decision_year"] >= year_range[0]) &
    (df_clean["decision_year"] <= year_range[1])
]

st.metric("Submissions in selected range", f"{len(df_filtered):,}")

# Also add a single-year selector for detailed view
selected_year = st.selectbox(
    "View detailed stats for year",
    options=list(range(max_year, min_year - 1, -1)),  # Most recent first
)
year_data = df_clean[df_clean["decision_year"] == selected_year]
st.write(f"**{selected_year}:** {len(year_data):,} submissions")
st.write(year_data["pathway"].value_counts())
```

### Key EDA Insights from Extended Range

The 1980-2025 data will reveal:
- The dramatic growth of 510(k) submissions from the 1980s onward
- The introduction of De Novo pathway (1997) and its growth after FDASIA (2012)
- Shifts in which medical specialties dominate submissions over decades
- How the ratio of PMA vs 510(k) has changed over time
- Impact of regulatory reforms on submission volumes

### Note on Data Volume

Fetching 1980-2025 will pull significantly more records (~200K+ total). The pagination in `fda_api_tool.py` already handles this by chunking queries by year and splitting busy years into quarters. The API fetch will take longer (10-20 minutes) but only runs once.
