# FDA Regulatory Pathway Predictor

A CrewAI-powered ML pipeline that predicts FDA regulatory pathways (510(k), PMA, or De Novo) for medical devices.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python -m src.flow.main_flow  # Run pipeline
streamlit run src/app/streamlit_app.py  # Launch app
```

## Architecture

```
openFDA API → [Data Ingestor] → [Data Cleaner] → [EDA Analyst]
                                                       │
                                                Validation Gate
                                                       │
             [Model Evaluator] ← [Model Trainer] ← [Feature Engineer]
                     │
               [Streamlit App]
```

## Two-Stage Prediction

1. **Rule-based exemption check** — queries FDA classification database. If Class I/II exempt → no submission needed.
2. **ML prediction** — for non-exempt devices, predicts 510(k) vs PMA vs De Novo with confidence scores.
3. **Historical evidence** — shows similar devices and their clearance rates (SESE decision code).

## Tech Stack

CrewAI, Python, Pandas, scikit-learn, Matplotlib/Seaborn, Streamlit, Git+GitHub

## Data Source

openFDA API (api.fda.gov) — 200K+ public FDA submissions spanning 1980-2025.

## License

Educational project — AI Developer Course final project.
