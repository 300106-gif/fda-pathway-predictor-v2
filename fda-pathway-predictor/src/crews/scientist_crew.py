"""Data Scientist Crew — 3 agents wrapping pipeline tools."""
import os, logging
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
logger = logging.getLogger(__name__)

class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering"
    description: str = "Transforms clean data into model-ready features."
    def _run(self, **kwargs) -> str:
        from src.tools.feature_tool import engineer_features
        f = engineer_features()
        cols = [c for c in f.columns if c not in ["submission_id","pathway","pathway_encoded"]]
        return f"Feature engineering complete. {len(f)} records, {len(cols)} features."

class ModelTrainingTool(BaseTool):
    name: str = "model_training"
    description: str = "Trains RF and Gradient Boosting, saves best model."
    def _run(self, **kwargs) -> str:
        from src.tools.model_tool import train_and_evaluate
        model, results = train_and_evaluate()
        lines = ["Training complete."]
        for name, m in results.items():
            lines.append(f"{name}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
        best = max(results.items(), key=lambda x: x[1]["f1_macro"])
        lines.append(f"Best: {best[0]}")
        return "\n".join(lines)

class ModelEvaluationTool(BaseTool):
    name: str = "model_evaluation"
    description: str = "Reads evaluation artifacts and summarizes."
    def _run(self, **kwargs) -> str:
        artifacts = Path("artifacts")
        files = {f: "exists" if (artifacts/f).exists() else "MISSING" for f in ["evaluation_report.md","model_card.md","confusion_matrices.png","feature_importance.png"]}
        return f"Evaluation artifacts: {files}"

def create_scientist_crew() -> Crew:
    fe = Agent(role="Feature Engineer", goal="Transform clean data into ML-ready features.", backstory="You specialize in feature engineering for classification.", tools=[FeatureEngineeringTool()], verbose=True, allow_delegation=False)
    trainer = Agent(role="Model Trainer", goal="Train and compare RF and Gradient Boosting models.", backstory="You are a data scientist with multiclass classification expertise.", tools=[ModelTrainingTool()], verbose=True, allow_delegation=False)
    evaluator = Agent(role="Model Evaluator", goal="Evaluate models and generate reports.", backstory="You believe in thorough model evaluation and documentation.", tools=[ModelEvaluationTool()], verbose=True, allow_delegation=False)

    t1 = Task(description="Engineer features from clean_data.csv. Validate against dataset_contract.json. Save features.csv.", expected_output="Feature count and validation status.", agent=fe)
    t2 = Task(description="Train RF (200 trees, balanced) and GB (200 estimators). 5-fold CV. Save model.pkl.", expected_output="Model comparison results.", agent=trainer)
    t3 = Task(description="Verify evaluation artifacts exist. Summarize model performance.", expected_output="Artifact status and performance summary.", agent=evaluator)

    return Crew(agents=[fe, trainer, evaluator], tasks=[t1, t2, t3], process=Process.sequential, verbose=True)

if __name__ == "__main__":
    crew = create_scientist_crew()
    print(f"Scientist crew: {[a.role for a in crew.agents]}")
