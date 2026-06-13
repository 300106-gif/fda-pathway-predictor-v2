# FDA Pathway Predictor — Step-by-Step Build Guide

Follow these steps in order. Each step has the exact commands to run.
Your project folder: `C:\Users\30010\V2_FDA_PW_TR\`

---

## PHASE 1: FIX PYTHON ENVIRONMENT (do this first)

### Step 1.1 — Check your Python

Open PowerShell and run:

```powershell
where python
python --version
where pip
pip --version
```

Write down the paths. If you see multiple Python installations, that's your problem.

### Step 1.2 — Create a virtual environment

This isolates your project from other Python installations:

```powershell
cd "C:\Users\30010\V2_FDA_PW_TR"
python -m venv venv
```

### Step 1.3 — Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If you get a permissions error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your command line. This means you're in the virtual environment.

**IMPORTANT:** Every time you open a new PowerShell window, you need to activate the venv again:
```powershell
cd "C:\Users\30010\V2_FDA_PW_TR"
.\venv\Scripts\Activate.ps1
```

### Step 1.4 — Install all dependencies

```powershell
pip install requests pandas scikit-learn matplotlib seaborn joblib streamlit numpy
pip install crewai crewai-tools
pip install python-dotenv
pip install jupyter
```

### Step 1.5 — Test that everything works

```powershell
python -c "import requests; print('requests OK')"
python -c "import pandas; print('pandas OK')"
python -c "import sklearn; print('sklearn OK')"
python -c "import matplotlib; print('matplotlib OK')"
python -c "import streamlit; print('streamlit OK')"
python -c "import crewai; print('crewai OK')"
```

All should print "OK". If any fail, run `pip install <package-name>` for that package.

### Step 1.6 — Test the FDA API

```powershell
epython -c "import requsts; r = requests.get('https://api.fda.gov/device/510k.json?limit=1'); print(f'Status: {r.status_code}, Records: {r.json()[\"meta\"][\"results\"][\"total\"]}')"
```

You should see: `Status: 200, Records: 174XXX`

---

## PHASE 2: SET UP GIT AND GITHUB

### Step 2.1 — Create GitHub repository

1. Go to https://github.com/new
2. Name: `fda-pathway-predictor`
3. Public
4. Do NOT check "Add a README" (you already have one)
5. Click Create repository
6. Copy the URL (looks like `https://github.com/YOURUSERNAME/fda-pathway-predictor.git`)

### Step 2.2 — Initialize Git locally

```powershell
cd "C:\Users\30010\V2_FDA_PW_TR"
git init
git branch -M main
git remote add origin https://github.com/300106-gif/fda-pathway-predictor-v2.git
```

### Step 2.3 — PR #1: Project Setup

```powershell
git checkout -b feature/project-setup

git add .gitignore
git add .env.example
git add .cursorrules
git add README.md
git add requirements.txt
git add src/__init__.py
git add src/crews/__init__.py
git add src/flow/__init__.py
git add src/tools/__init__.py
git add src/app/__init__.py

git commit -m "project setup: structure, dependencies, README, env config"
git push -u origin feature/project-setup
```

Now go to GitHub:
1. You'll see a yellow banner "feature/project-setup had recent pushes"
2. Click "Compare & pull request"
3. Title: "Project Setup"
4. Description: "Initial project structure with .gitignore, README, requirements.txt, and package init files"
5. Click "Create pull request"
6. Click "Merge pull request"
7. Click "Confirm merge"

### Step 2.4 — After each merge, sync main locally:

```powershell
git checkout main
git pull origin main
```

---

## PHASE 3: BUILD THE PIPELINE (one branch per PR)

### PR #2 — Data Ingestion

```powershell
git checkout main
git pull
git checkout -b feature/data-ingestion
git add src/tools/fda_api_tool.py
git add src/tools/generate_sample_data.py
git commit -m "feat: FDA API ingestion tool with pagination and sample data generator"
git push -u origin feature/data-ingestion
```

Go to GitHub → Create PR → Title: "Data Ingestion" → Description:
```
Added FDA API ingestion tool that fetches from 3 openFDA endpoints:
- /device/510k.json (510(k) + De Novo records)
- /device/pma.json (PMA records)
- /device/classification.json (enrichment data)

Handles API pagination (1000 record limit) by year/quarter partitioning.
Includes sample data generator for development when API is unavailable.

Output: artifacts/raw_data.csv
```
→ Merge

```powershell
git checkout main
git pull
```

### PR #3 — Data Cleaning + EDA

```powershell
git checkout -b feature/data-cleaning-eda
git add src/tools/cleaning_tool.py
git add src/tools/eda_tool.py
git commit -m "feat: data cleaning with dataset contract and EDA report generator"
git push -u origin feature/data-cleaning-eda
```

GitHub → Create PR → Title: "Data Cleaning & EDA" → Description:
```
Data Cleaner:
- Removes duplicates, handles missing values
- Imputes missing device_class using decision_code (SESE → Class II)
- Generates dataset_contract.json with schema and constraints
- Output: clean_data.csv, dataset_contract.json

EDA Analyst:
- Generates 7 chart types (pathway distribution, time trends, device class breakdown, etc.)
- Output: eda_report.html, insights.md
```
→ Merge → `git checkout main && git pull`

### PR #4 — Feature Engineering

```powershell
git checkout -b feature/feature-engineering
git add src/tools/feature_tool.py
git commit -m "feat: 18 engineered features with contract validation"
git push -u origin feature/feature-engineering
```

GitHub → Create PR → Title: "Feature Engineering" → Description:
```
Engineers 18 features from clean data:
- device_class, advisory_committee (freq + label encoded)
- medical_specialty, country (freq encoded), is_us boolean
- Temporal: year, month, cyclical sin/cos encoding
- review_days, clearance_type, third_party flag
- product_code_freq, applicant_freq, applicant_submission_count

Note: decision_code_encoded is EXCLUDED from model features (target leakage).
Validates features against dataset_contract.json before saving.

Output: features.csv, label_encoders.json
```
→ Merge → `git checkout main && git pull`

### PR #5 — Model Training

```powershell
git checkout -b feature/model-training
git add src/tools/model_tool.py
git commit -m "feat: Random Forest and Gradient Boosting with evaluation and model card"
git push -u origin feature/model-training
```

GitHub → Create PR → Title: "Model Training & Evaluation" → Description:
```
Trains two models:
1. Random Forest (200 trees, balanced class weights, max_depth=15)
2. Gradient Boosting (200 estimators, lr=0.1, max_depth=6)

Evaluation:
- 5-fold stratified cross-validation
- Metrics: accuracy, F1-macro, precision, recall, ROC-AUC
- Confusion matrices and feature importance plots
- Model card with limitations and ethical considerations

Output: model.pkl, evaluation_report.md, model_card.md, confusion_matrices.png, feature_importance.png
```
→ Merge → `git checkout main && git pull`

### PR #6 — CrewAI Flow

```powershell
git checkout -b feature/crewai-flow
git add src/crews/analyst_crew.py
git add src/crews/scientist_crew.py
git add src/flow/main_flow.py
git commit -m "feat: CrewAI crews (6 agents) and Flow with validation gate"
git push -u origin feature/crewai-flow
```

GitHub → Create PR → Title: "CrewAI Orchestration" → Description:
```
Crew 1 — Data Analyst (3 agents):
- Data Ingestor, Data Cleaner, EDA Analyst
- Each agent wraps a pipeline tool as a CrewAI BaseTool

Crew 2 — Data Scientist (3 agents):
- Feature Engineer, Model Trainer, Model Evaluator

Flow:
- CrewAI Flow orchestrates both crews with automated handoff
- Validation gate between crews checks dataset_contract.json
- Supports two modes: CrewAI (with LLM key) and direct (no LLM)
- Fail-safe: falls back to sample data if FDA API is unavailable
```
→ Merge → `git checkout main && git pull`

### PR #7 — Streamlit App

```powershell
git checkout -b feature/streamlit-app
git add src/app/streamlit_app.py
git commit -m "feat: 3-page Streamlit dashboard with pathway predictor"
git push -u origin feature/streamlit-app
```

GitHub → Create PR → Title: "Streamlit Dashboard" → Description:
```
Three-page Streamlit app:
1. Pathway Predictor — device input form → ML prediction with confidence scores
2. EDA Dashboard — embedded charts, insights, year filter slider
3. Model Performance — evaluation report, confusion matrices, model card

Features two-stage prediction:
- Stage 1: Rule-based exemption check via openFDA classification API
- Stage 2: ML model for non-exempt devices (510(k) vs PMA vs De Novo)

Includes similar device lookup showing historical evidence from FDA data.
```
→ Merge → `git checkout main && git pull`

---

## PHASE 4: RUN THE PIPELINE

### Step 4.1 — Make sure venv is active

```powershell
cd "C:\Users\30010\V2_FDA_PW_TR"
.\venv\Scripts\Activate.ps1
```

### Step 4.2 — Create artifacts folder

```powershell
mkdir artifacts
```

### Step 4.3 — Run the full pipeline

```powershell
python -m src.flow.main_flow
```

This will:
1. Try to fetch real FDA data (if API works, takes 10-20 minutes for 1980-2025)
2. If API fails, generate sample data (takes 5 seconds)
3. Clean the data
4. Generate EDA report
5. Engineer features
6. Train both models
7. Generate all 13 artifacts

### Step 4.4 — Verify artifacts were created

```powershell
dir artifacts
```

You should see 13+ files: raw_data.csv, clean_data.csv, dataset_contract.json, eda_report.html, insights.md, features.csv, label_encoders.json, model.pkl, model_meta.json, evaluation_report.md, model_card.md, confusion_matrices.png, feature_importance.png

### Step 4.5 — Launch Streamlit

```powershell
streamlit run src/app/streamlit_app.py
```

This opens a browser at http://localhost:8501. Test all 3 pages.
Press Ctrl+C in the terminal to stop.

---

## PHASE 5: CREATE THE TEST NOTEBOOK

### Step 5.1 — Launch Jupyter

```powershell
jupyter notebook
```

### Step 5.2 — Create new notebook

Click "New" → "Python 3" → Rename to "test"

### Step 5.3 — Add cells

Copy the cells from the CLAUDE_CODE_PROMPT.md file (the "JUPYTER NOTEBOOK FOR TESTING" section). Each `Cell N` block is one notebook cell. Paste each block into a separate cell and run them with Shift+Enter.

---

## PHASE 6: DEPLOY TO STREAMLIT CLOUD

### Step 6.1 — Commit artifacts needed for deployment

```powershell
git checkout main
git pull
git checkout -b feature/deployment
git add artifacts/model.pkl
git add artifacts/label_encoders.json
git add artifacts/model_meta.json
git add artifacts/clean_data.csv
git add artifacts/dataset_contract.json
git add artifacts/evaluation_report.md
git add artifacts/model_card.md
git add artifacts/insights.md
git add artifacts/eda_report.html
git add artifacts/confusion_matrices.png
git add artifacts/feature_importance.png
git commit -m "add pipeline artifacts for Streamlit Cloud deployment"
git push -u origin feature/deployment
```

→ Create PR → Merge

### Step 6.2 — Deploy

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo: `fda-pathway-predictor`
5. Branch: `main`
6. Main file path: `src/app/streamlit_app.py`
7. Click "Deploy"

Wait 2-3 minutes. You'll get a public URL like `https://yourusername-fda-pathway-predictor.streamlit.app`

---

## PHASE 7: RECORD DEMO VIDEO (≤5 min)

Your teacher requires a short demo video. Record these:

1. (30 sec) Open the Streamlit app, explain what it does
2. (1 min) Enter a Class II cardiovascular device → show 510(k) prediction with confidence
3. (30 sec) Show the similar devices evidence section
4. (1 min) Enter a Class I exempt device → show the exemption result
5. (30 sec) Show the EDA Dashboard with year filter
6. (30 sec) Show the Model Performance page
7. (30 sec) Show GitHub repo with PR history
8. (30 sec) Show the terminal running the pipeline

Use Windows built-in screen recorder: Win+G → Record

---

## TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'X'"
```powershell
pip install X
```

### "git is not recognized"
Install Git from https://git-scm.com/download/win

### Virtual environment won't activate
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Pipeline runs but model accuracy is 100%
This is expected with sample data. Real FDA data will give 75-90% accuracy.

### Streamlit won't start
Make sure venv is active and you're in the project root folder.

### FDA API returns 403
Your network may be blocking it. The pipeline will automatically fall back to sample data.

### "crewai" import fails
```powershell
pip install crewai crewai-tools --force-reinstall
```

### Git push fails with "remote rejected"
Make sure you created the GitHub repo first (Step 2.1) and the URL is correct.
