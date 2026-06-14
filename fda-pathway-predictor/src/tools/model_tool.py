"""Model Training & Evaluation — RF vs Gradient Boosting."""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
import joblib, json, logging
from pathlib import Path
from datetime import datetime
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # Device identity — legitimate signals present before submission decision
    "device_class", "device_class_unknown",
    "advisory_committee_freq", "advisory_committee_encoded",
    "medical_specialty_freq", "medical_specialty_encoded",
    "product_code_freq",
    # Geography
    "is_us", "country_freq",
    # Temporal — submission volume trends over time
    "decision_year", "decision_month", "month_sin", "month_cos",
    # Device-level risk flags (from foiclass join) — key signals for 510(k) exempt
    # These reflect device TYPE characteristics, not submission outcomes → no leakage
    "implant_flag",       # implants → PMA/510k, not exempt
    "life_sustain_flag",  # life-sustaining → higher oversight, not exempt
    "gmp_exempt",         # GMP-exempt → often correlates with low-risk / 510k exempt
    # EXCLUDED — submission-type-specific fields that leak the pathway label:
    #   clearance_type_encoded  : Traditional/Special/Abbreviated are 510(k)-only terms
    #   third_party             : Third-party review only exists for 510(k)
    #   review_days / has_review_days : unavailable for PMA/De Novo
    # EXCLUDED — not available at inference time + create synthetic signal in exempt records:
    #   applicant_freq          : all synthetic exempt rows share one fake applicant
    #   applicant_submission_count : same issue
]

def train_and_evaluate(input_path="artifacts/features.csv", output_dir="artifacts", test_size=0.2, random_state=42):
    output_dir = Path(output_dir)
    df = pd.read_csv(input_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Using {len(available)} features")
    X = df[available].fillna(0); y = df["pathway_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    enc_path = output_dir/"label_encoders.json"
    if enc_path.exists():
        with open(enc_path) as f: encoders = json.load(f)
        pathway_names = [encoders["pathway_inverse"][str(i)] for i in range(len(encoders["pathway_inverse"]))]
    else: pathway_names = [str(i) for i in sorted(y.unique())]

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, class_weight="balanced", random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train); rf_pred = rf.predict(X_test); rf_proba = rf.predict_proba(X_test)

    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state)
    gb_sample_weights = compute_sample_weight("balanced", y_train)
    gb.fit(X_train, y_train, sample_weight=gb_sample_weights)
    gb_pred = gb.predict(X_test); gb_proba = gb.predict_proba(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
    gb_cv = cross_val_score(gb, X, y, cv=cv, scoring="f1_macro")

    def _metrics(y_t, y_p, y_pr, cv_s):
        m = {"accuracy": accuracy_score(y_t,y_p), "f1_macro": f1_score(y_t,y_p,average="macro"),
             "f1_weighted": f1_score(y_t,y_p,average="weighted"), "precision_macro": precision_score(y_t,y_p,average="macro"),
             "recall_macro": recall_score(y_t,y_p,average="macro"), "cv_f1_mean": cv_s.mean(), "cv_f1_std": cv_s.std(),
             "classification_report": classification_report(y_t,y_p,target_names=pathway_names),
             "confusion_matrix": confusion_matrix(y_t,y_p).tolist()}
        try:
            y_bin = label_binarize(y_t, classes=list(range(len(pathway_names))))
            m["roc_auc_macro"] = roc_auc_score(y_bin, y_pr, average="macro", multi_class="ovr")
        except: m["roc_auc_macro"] = None
        return m

    results = {"Random Forest": _metrics(y_test,rf_pred,rf_proba,rf_cv), "Gradient Boosting": _metrics(y_test,gb_pred,gb_proba,gb_cv)}
    rf_imp = pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)
    gb_imp = pd.Series(gb.feature_importances_, index=available).sort_values(ascending=False)

    best_name = "Random Forest" if results["Random Forest"]["f1_macro"] >= results["Gradient Boosting"]["f1_macro"] else "Gradient Boosting"
    best_model = rf if best_name == "Random Forest" else gb
    best_imp = rf_imp if best_name == "Random Forest" else gb_imp
    logger.info(f"Best model: {best_name}")

    joblib.dump(best_model, output_dir/"model.pkl")
    with open(output_dir/"model_meta.json","w") as f:
        json.dump({"feature_columns": available, "best_model": best_name, "feature_importance": best_imp.to_dict()}, f, indent=2)

    # Confusion matrices
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for ax,pred,title in [(axes[0],rf_pred,"Random Forest"),(axes[1],gb_pred,"Gradient Boosting")]:
        sns.heatmap(confusion_matrix(y_test,pred), annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=pathway_names, yticklabels=pathway_names)
        ax.set_title(f"{title}"); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig(output_dir/"confusion_matrices.png", dpi=100); plt.close()

    # Feature importance
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax,imp,title in [(axes[0],rf_imp.head(10),"Random Forest"),(axes[1],gb_imp.head(10),"Gradient Boosting")]:
        ax.barh(imp.index[::-1], imp.values[::-1], color="#378ADD", alpha=0.8)
        ax.set_title(f"{title} — Top Features"); ax.set_xlabel("Importance")
    plt.tight_layout(); plt.savefig(output_dir/"feature_importance.png", dpi=100); plt.close()

    # Reports
    eval_lines = [f"# Model Evaluation Report\n\n**Date:** {datetime.now().strftime('%Y-%m-%d')}\n**Best Model:** {best_name}\n\n"]
    eval_lines.append("| Metric | Random Forest | Gradient Boosting |\n|---|:---:|:---:|\n")
    for label,key in [("Accuracy","accuracy"),("F1-macro","f1_macro"),("Precision","precision_macro"),("Recall","recall_macro"),("ROC-AUC","roc_auc_macro"),("CV F1","cv_f1_mean")]:
        rv = results["Random Forest"].get(key); gv = results["Gradient Boosting"].get(key)
        rv_s = f"{rv:.4f}" if rv is not None else "N/A"
        gv_s = f"{gv:.4f}" if gv is not None else "N/A"
        eval_lines.append(f"| {label} | {rv_s} | {gv_s} |\n")
    eval_lines.append(f"\n## Classification Report — {best_name}\n```\n{results[best_name]['classification_report']}```\n")
    (output_dir/"evaluation_report.md").write_text("".join(eval_lines))

    model_card = f"""# Model Card — FDA Pathway Predictor\n\n## Purpose\nPredicts FDA regulatory pathway (510(k) Exempt, 510(k), PMA, De Novo) for medical devices.\n510(k) Exempt: low-risk Class I/II devices that may be marketed without premarket notification.\n\n## Model: {best_name}\n## Features: {len(available)}\n## Training Records: {len(df):,}\n\n## Metrics\n| Metric | Value |\n|---|---|\n| Accuracy | {results[best_name]['accuracy']:.4f} |\n| F1-macro | {results[best_name]['f1_macro']:.4f} |\n| CV F1 | {results[best_name]['cv_f1_mean']:.4f} ± {results[best_name]['cv_f1_std']:.4f} |\n\n## Limitations\n- Trained on historical data; regulatory criteria can change\n- Class imbalance (510(k) dominates)\n- Does not analyze submission narratives or clinical evidence\n- Decision support only — not regulatory advice\n\n## Ethical Considerations\n- May reflect historical biases in FDA decisions\n- Must be used by qualified regulatory professionals\n- All data is publicly available via openFDA API\n"""
    (output_dir/"model_card.md").write_text(model_card)
    logger.info("Reports saved")
    return best_model, results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, results = train_and_evaluate()
    for n,m in results.items(): print(f"{n}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
