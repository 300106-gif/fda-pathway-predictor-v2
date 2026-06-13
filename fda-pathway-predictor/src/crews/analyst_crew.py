"""Data Analyst Crew — 3 agents wrapping pipeline tools."""
import os, logging
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
logger = logging.getLogger(__name__)

class IngestionTool(BaseTool):
    name: str = "fda_data_ingestion"
    description: str = "Fetches FDA device data from openFDA API, merges into unified dataset."
    def _run(self, **kwargs) -> str:
        from src.tools.fda_api_tool import ingest_fda_data
        from src.tools.generate_sample_data import generate_records
        raw_path = Path("artifacts/raw_data.csv")
        try:
            df = ingest_fda_data(output_path=str(raw_path), start_year=1980, end_year=2025)
            if len(df) < 100: raise ValueError(f"Only {len(df)} records")
        except Exception as e:
            logger.warning(f"API failed ({e}), using sample data")
            df = generate_records(n_510k=8000, n_pma=1500, n_denovo=500)
            raw_path.parent.mkdir(exist_ok=True)
            df.to_csv(raw_path, index=False)
        return f"Ingestion complete. {len(df)} records saved. Pathways: {df['pathway'].value_counts().to_dict()}"

class CleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = "Cleans raw FDA data, produces clean_data.csv and dataset_contract.json."
    def _run(self, **kwargs) -> str:
        from src.tools.cleaning_tool import clean_data
        df = clean_data()
        return f"Cleaning complete. {len(df)} clean records."

class EDATool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = "Generates EDA report with charts and insights."
    def _run(self, **kwargs) -> str:
        from src.tools.eda_tool import generate_eda
        generate_eda()
        return "EDA complete. eda_report.html and insights.md saved."

def create_analyst_crew() -> Crew:
    ingestor = Agent(role="Data Ingestor", goal="Fetch FDA data from openFDA API and produce unified raw dataset.", backstory="You are a data engineer specializing in healthcare regulatory data.", tools=[IngestionTool()], verbose=True, allow_delegation=False)
    cleaner = Agent(role="Data Cleaner", goal="Clean raw data, handle missing values, generate dataset contract.", backstory="You are a meticulous data quality analyst.", tools=[CleaningTool()], verbose=True, allow_delegation=False)
    eda = Agent(role="EDA Analyst", goal="Perform exploratory data analysis and produce visual insights.", backstory="You are a senior data analyst with regulatory expertise.", tools=[EDATool()], verbose=True, allow_delegation=False)

    t1 = Task(description="Fetch data from openFDA API (510k, PMA, classification). Save as artifacts/raw_data.csv.", expected_output="Record count and pathway distribution.", agent=ingestor)
    t2 = Task(description="Clean raw data: dedup, handle missing values, standardize types. Save clean_data.csv and dataset_contract.json.", expected_output="Clean record count and null summary.", agent=cleaner)
    t3 = Task(description="Generate EDA with charts covering pathway distribution, time trends, device class breakdown. Save eda_report.html and insights.md.", expected_output="Summary of key findings.", agent=eda)

    return Crew(agents=[ingestor, cleaner, eda], tasks=[t1, t2, t3], process=Process.sequential, verbose=True)

if __name__ == "__main__":
    crew = create_analyst_crew()
    print(f"Analyst crew: {[a.role for a in crew.agents]}")
