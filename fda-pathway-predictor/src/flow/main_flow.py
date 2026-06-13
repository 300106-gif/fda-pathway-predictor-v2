"""FDA Pathway Predictor — CrewAI Flow orchestration."""
import os, sys, json, logging, pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from crewai.flow.flow import Flow, start, listen
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Flow = object
    def start(): return lambda f: f
    def listen(*a): return lambda f: f

logger = logging.getLogger(__name__)

class FDAPathwayFlow(Flow):
    def __init__(self):
        super().__init__()
        self.artifacts_dir = Path("artifacts"); self.artifacts_dir.mkdir(exist_ok=True)
        self.run_log = []
    def _log(self, step, status, details=""):
        self.run_log.append({"timestamp": datetime.now().isoformat(), "step": step, "status": status, "details": details})
        logger.info(f"[{status}] {step}: {details}")

    @start()
    def run_analyst_crew(self):
        self._log("analyst_crew","STARTED")
        from src.crews.analyst_crew import create_analyst_crew
        result = create_analyst_crew().kickoff()
        self._log("analyst_crew","COMPLETED",str(result)[:500])
        return {"status":"success"}

    @listen(run_analyst_crew)
    def validate_handoff(self, r):
        self._log("validation","STARTED")
        errors = []
        for f in ["clean_data.csv","dataset_contract.json","eda_report.html","insights.md"]:
            if not (self.artifacts_dir/f).exists(): errors.append(f"Missing: {f}")
        if errors: raise RuntimeError(f"Validation failed: {errors}")
        with open(self.artifacts_dir/"dataset_contract.json") as f: contract = json.load(f)
        df = pd.read_csv(self.artifacts_dir/"clean_data.csv")
        if contract["target_variable"] not in df.columns: errors.append("Target missing")
        if len(df) < 100: errors.append(f"Too few records: {len(df)}")
        if errors: raise RuntimeError(f"Validation failed: {errors}")
        self._log("validation","PASSED",f"{len(df)} records")
        return {"status":"validated","records":len(df)}

    @listen(validate_handoff)
    def run_scientist_crew(self, r):
        self._log("scientist_crew","STARTED")
        from src.crews.scientist_crew import create_scientist_crew
        result = create_scientist_crew().kickoff()
        self._log("scientist_crew","COMPLETED",str(result)[:500])
        return {"status":"success"}

    @listen(run_scientist_crew)
    def finalize(self, r):
        all_files = ["raw_data.csv","clean_data.csv","dataset_contract.json","eda_report.html","insights.md","features.csv","model.pkl","evaluation_report.md","model_card.md","confusion_matrices.png","feature_importance.png","label_encoders.json","model_meta.json"]
        present = [f for f in all_files if (self.artifacts_dir/f).exists()]
        with open(self.artifacts_dir/"pipeline_run_log.json","w") as f: json.dump(self.run_log, f, indent=2)
        print(f"\n{'='*60}\n  Pipeline Complete: {len(present)}/{len(all_files)} artifacts\n{'='*60}")

def run_pipeline_without_llm():
    """Run pipeline directly without LLM."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path("artifacts").mkdir(exist_ok=True)

    logger.info("="*50+"\nSTEP 1: Data Ingestion\n"+"="*50)
    try:
        from src.tools.fda_api_tool import ingest_fda_data
        df_raw = ingest_fda_data(start_year=1980, end_year=2025)
        if len(df_raw) < 100: raise ValueError(f"Only {len(df_raw)} records")
    except Exception as e:
        logger.warning(f"API failed ({e}), using sample data")
        from src.tools.generate_sample_data import generate_records
        df_raw = generate_records(); df_raw.to_csv("artifacts/raw_data.csv", index=False)
    logger.info(f"Raw data: {df_raw.shape}")

    logger.info("="*50+"\nSTEP 2: Data Cleaning\n"+"="*50)
    from src.tools.cleaning_tool import clean_data
    df_clean = clean_data()

    logger.info("="*50+"\nSTEP 3: EDA\n"+"="*50)
    from src.tools.eda_tool import generate_eda
    generate_eda()

    logger.info("="*50+"\nSTEP 4: Validation\n"+"="*50)
    with open("artifacts/dataset_contract.json") as f: contract = json.load(f)
    assert contract["target_variable"] in df_clean.columns
    assert len(df_clean) >= 100
    logger.info("Validation PASSED")

    logger.info("="*50+"\nSTEP 5: Feature Engineering\n"+"="*50)
    from src.tools.feature_tool import engineer_features
    features = engineer_features()

    logger.info("="*50+"\nSTEP 6: Model Training\n"+"="*50)
    from src.tools.model_tool import train_and_evaluate
    model, results = train_and_evaluate()
    for n,m in results.items(): logger.info(f"  {n}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    logger.info(f"\n{'='*50}\nPIPELINE COMPLETE\n{'='*50}")
    for f in sorted(Path("artifacts").glob("*")): logger.info(f"  {f.name}: {f.stat().st_size/1024:.1f} KB")
    return model, results

if __name__ == "__main__":
    has_llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if has_llm_key and CREWAI_AVAILABLE:
        logging.basicConfig(level=logging.INFO)
        FDAPathwayFlow().kickoff()
    else:
        if has_llm_key and not CREWAI_AVAILABLE:
            print("CrewAI not available (Python 3.14+ not yet supported). Running direct mode...\n")
        else:
            print("No LLM key found. Running direct mode...\n")
        run_pipeline_without_llm()
