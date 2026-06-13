"""EDA Tool — generates charts and insights."""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def generate_eda(input_path="artifacts/clean_data.csv", output_dir="artifacts"):
    output_dir = Path(output_dir)
    df = pd.read_csv(input_path)
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    logger.info(f"Running EDA on {len(df)} records")
    charts, insights = [], []

    # 1. Pathway distribution
    fig, ax = plt.subplots(figsize=(8,5))
    pc = df["pathway"].value_counts()
    colors = {"510k":"#378ADD","PMA":"#D85A30","De_Novo":"#1D9E75"}
    bars = ax.bar(pc.index, pc.values, color=[colors.get(p,"#888") for p in pc.index])
    ax.set_title("Distribution of FDA Regulatory Pathways", fontsize=14); ax.set_ylabel("Submissions")
    for b,v in zip(bars, pc.values): ax.text(b.get_x()+b.get_width()/2, b.get_height()+50, f"{v:,}", ha="center")
    charts.append(("Pathway Distribution", _fig_to_b64(fig)))
    insights.append(f"Dataset contains {len(df):,} submissions: {pc.get('510k',0):,} via 510(k), {pc.get('PMA',0):,} via PMA, {pc.get('De_Novo',0):,} via De Novo.")

    # 2. Time trends
    if "decision_year" in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        yearly = df.groupby(["decision_year","pathway"]).size().unstack(fill_value=0)
        for p in ["510k","PMA","De_Novo"]:
            if p in yearly.columns: ax.plot(yearly.index, yearly[p], marker="o", lw=2, color=colors.get(p,"#888"), label=p)
        ax.set_title("Submissions Over Time by Pathway"); ax.set_xlabel("Year"); ax.set_ylabel("Count"); ax.legend(); ax.grid(True, alpha=0.3)
        charts.append(("Submissions Over Time", _fig_to_b64(fig)))

    # 3. Specialty distribution (all categories including SaMD/IVD/IMPLANT/ANALYZER_POC)
    fig, ax = plt.subplots(figsize=(12, 6))
    spec_counts = df["advisory_committee"].value_counts().head(15)
    special_colors = {
        "SAMD": "#7C3AED", "IVD": "#0891B2", "IMPLANT": "#D85A30",
        "ANALYZER_POC": "#F59E0B", "UNKNOWN": "#94A3B8",
    }
    bar_colors = [special_colors.get(s, "#378ADD") for s in spec_counts.index]
    ax.barh(spec_counts.index[::-1], spec_counts.values[::-1], color=bar_colors[::-1], alpha=0.85)
    ax.set_title("Medical Specialty / Category Distribution", fontsize=13)
    ax.set_xlabel("Submissions")
    for i, v in enumerate(spec_counts.values[::-1]):
        ax.text(v + spec_counts.values.max() * 0.01, i, f"{v:,}", va="center", fontsize=9)
    charts.append(("Specialty Distribution", _fig_to_b64(fig)))

    # 4. Device class breakdown per specialty (stacked bar)
    if "device_class" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_specs = df["advisory_committee"].value_counts().head(12).index
        ct = pd.crosstab(
            df[df["advisory_committee"].isin(top_specs)]["advisory_committee"],
            df[df["advisory_committee"].isin(top_specs)]["device_class"],
        )
        ct = ct.reindex(top_specs)
        class_colors = {1: "#85B7EB", 2: "#378ADD", 3: "#0C447C"}
        bottom = np.zeros(len(ct))
        for cls in sorted(ct.columns):
            vals = ct[cls].fillna(0).values
            ax.barh(ct.index, vals, left=bottom,
                    color=class_colors.get(int(cls), "#aaa"), label=f"Class {int(cls)}", alpha=0.85)
            bottom += vals
        ax.set_title("Device Class Breakdown by Specialty", fontsize=13)
        ax.set_xlabel("Submissions")
        ax.legend(title="Device Class", loc="lower right")
        charts.append(("Device Class by Specialty", _fig_to_b64(fig)))

    # 5. Specialty vs Pathway heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    top_specs = df["advisory_committee"].value_counts().head(12).index
    hm = pd.crosstab(
        df[df["advisory_committee"].isin(top_specs)]["advisory_committee"],
        df[df["advisory_committee"].isin(top_specs)]["pathway"],
    )
    sns.heatmap(hm, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title("Specialty vs Regulatory Pathway", fontsize=13)
    ax.set_ylabel("Specialty / Category"); ax.set_xlabel("Pathway")
    charts.append(("Specialty vs Pathway Heatmap", _fig_to_b64(fig)))

    # 6. Country
    fig, ax = plt.subplots(figsize=(8, 5))
    us_pct = (df["is_us"] == 1).mean() * 100
    ax.pie([us_pct, 100 - us_pct], labels=["US", "International"],
           autopct="%1.1f%%", colors=["#378ADD", "#F0997B"])
    ax.set_title("Domestic vs International")
    charts.append(("Country", _fig_to_b64(fig)))

    # 7. Decision codes
    fig, ax = plt.subplots(figsize=(10, 5))
    dc = df["decision_code"].value_counts().head(10)
    ax.barh(dc.index[::-1], dc.values[::-1], color="#1D9E75", alpha=0.8)
    ax.set_title("Top 10 Decision Codes"); ax.set_xlabel("Count")
    charts.append(("Decision Codes", _fig_to_b64(fig)))

    # Build HTML
    html = '<!DOCTYPE html><html><head><title>FDA EDA Report</title><style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px}h1{border-bottom:2px solid #378ADD}h2{color:#378ADD;margin-top:40px}.chart{margin:20px 0;text-align:center}.chart img{max-width:100%;border-radius:8px}.summary{background:#f5f8fc;padding:16px;border-radius:8px;border-left:4px solid #378ADD;margin:20px 0}</style></head><body><h1>FDA Medical Device EDA Report</h1>'
    for title, b64 in charts:
        html += f'<h2>{title}</h2><div class="chart"><img src="data:image/png;base64,{b64}"></div>'
    html += '</body></html>'
    (output_dir/"eda_report.html").write_text(html)

    insights_md = "# FDA Medical Device — Key Insights\n\n" + "\n\n".join(f"## Insight {i+1}\n{ins}" for i,ins in enumerate(insights))
    (output_dir/"insights.md").write_text(insights_md)
    logger.info("EDA complete")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); generate_eda()
