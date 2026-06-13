"""FDA Pathway Predictor — Streamlit App (redesigned to match HTML mockups)."""
import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, requests
from pathlib import Path

ARTIFACTS = Path("artifacts")
st.set_page_config(
    page_title="FDA Pathway Advisor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in [
    ("device_class", 2),
    ("show_results", False),
    ("last_pathway", None),
    ("last_proba", []),
    ("last_confidence", 0.0),
    ("last_device_class", 2),
    ("last_advisory", "CV"),
    ("last_feature_cols", []),
    ("nav_page", "Pathway Predictor"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CSS Design System ──────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<style>
html, body, [class*="css"], h1, h2, h3, h4, p, span, div, button,
.stMarkdown, .stText, label { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #002046 !important; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stRadio label { color: rgba(255,255,255,0.8) !important; font-size: 14px; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: white !important; }

/* Cards */
.fda-card {
    background: white; border-radius: 12px; padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.05), 0 2px 4px -1px rgba(0,0,0,.03);
    border: 1px solid #f1f5f9; margin-bottom: 16px;
}

/* Device class selector cards */
.cls-card {
    border: 2px solid #e2e8f0; border-radius: 12px; padding: 18px 12px;
    background: white; text-align: center; margin-bottom: 8px; min-height: 108px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 7px;
    transition: border-color 0.15s;
}
.cls-card.sel-1 { border-color: #27AE60; background: #f0fdf4; }
.cls-card.sel-2 { border-color: #378ADD; background: #eff6ff; }
.cls-card.sel-3 { border-color: #D85A30; background: #fff7ed; }
.cls-dot { width: 13px; height: 13px; border-radius: 50%; display: inline-block; }

/* Results banner */
.res-banner {
    background: linear-gradient(135deg, #002046 0%, #1b365d 100%);
    border-radius: 16px; padding: 36px 44px; color: white;
    margin-bottom: 24px; display: flex; align-items: center;
    justify-content: space-between; gap: 20px;
}
.pw-badge {
    display: inline-block; background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3); border-radius: 20px;
    padding: 4px 14px; font-size: 13px; font-weight: 500; margin-bottom: 10px;
}
.pw-name { font-size: 40px; font-weight: 700; margin: 0 0 6px; letter-spacing: -.5px; }
.pw-sub  { font-size: 14px; color: rgba(255,255,255,.75); }

/* Info cards */
.ic {
    background: white; border-radius: 12px; padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.05); border: 1px solid #f1f5f9; height: 100%;
}
.ic-title {
    font-size: 12px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: .05em; margin: 0 0 14px;
}

/* Dot importance bars */
.feat-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 9px; }
.feat-nm  { font-size: 13px; color: #374151; font-weight: 500; flex: 1; margin-right: 8px; }
.feat-dots { display: flex; gap: 4px; }
.d-f { width: 10px; height: 10px; border-radius: 50%; background: #002046; display: inline-block; }
.d-e { width: 10px; height: 10px; border-radius: 50%; background: #e2e8f0; display: inline-block; }

/* Icon rows */
.ir  { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.iw  {
    width: 34px; height: 34px; border-radius: 8px; background: #eff6ff;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.iw .material-icons-round { font-size: 17px; color: #002046; }
.it-lbl { font-size: 11px; color: #64748b; font-weight: 500; }
.it-val { font-size: 14px; color: #1e293b; font-weight: 600; margin-top: 1px; }

/* Pathway bars */
.pw-bar-row { margin-bottom: 9px; }
.pw-bar-hd  { display: flex; justify-content: space-between; font-size: 13px; color: #374151; margin-bottom: 4px; font-weight: 500; }
.pw-bar-t   { background: #f1f5f9; border-radius: 4px; height: 7px; overflow: hidden; }
.pw-bar-f   { height: 7px; border-radius: 4px; }

/* Stepper */
.step-wrap  { display: flex; gap: 14px; }
.step-left  { display: flex; flex-direction: column; align-items: center; width: 38px; flex-shrink: 0; }
.s-circle   {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700; flex-shrink: 0;
}
.s-done    { background: #002046; color: white; }
.s-pending { background: white; color: #94a3b8; border: 2px solid #e2e8f0; font-size: 13px; }
.s-line     { width: 2px; background: #e2e8f0; flex: 1; min-height: 28px; margin: 4px 0; }
.s-line.done-l { background: #002046; }
.s-content  { padding-bottom: 28px; flex: 1; }
.s-lbl      { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 3px; }
.s-lbl-d   { color: #002046; }
.s-lbl-p   { color: #94a3b8; }
.s-ttl      { font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 5px; }
.s-desc     { font-size: 13px; color: #64748b; line-height: 1.55; }
.s-tip      { font-size: 12px; color: #94a3b8; font-style: italic; margin-top: 5px; line-height: 1.5; }

/* Notice cards */
.ntc     { border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; display: flex; gap: 12px; align-items: flex-start; }
.ntc-w   { background: #fffbeb; border: 1px solid #fcd34d; }
.ntc-i   { background: #eff6ff; border: 1px solid #93c5fd; }
.ntc-ttl { font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
.ntc-txt { font-size: 13px; color: #64748b; }

/* Sim section */
.sim-hd  { font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 3px; }
.sim-sub { font-size: 13px; color: #64748b; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = ARTIFACTS / "model.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_encoders():
    p = ARTIFACTS / "label_encoders.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_meta():
    p = ARTIFACTS / "model_meta.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_clean():
    p = ARTIFACTS / "clean_data.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_features():
    p = ARTIFACTS / "features.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_contract():
    p = ARTIFACTS / "dataset_contract.json"
    return json.load(open(p)) if p.exists() else {}

# ── Constants ──────────────────────────────────────────────────────────────────
COLORS = {"510k": "#378ADD", "PMA": "#D85A30", "De_Novo": "#1D9E75"}
COUNTRY_MAP = {
    "United States": "US", "Germany": "DE", "Japan": "JP", "United Kingdom": "GB",
    "France": "FR", "Israel": "IL", "Canada": "CA", "Switzerland": "CH",
    "Ireland": "IE", "Other": "OTHER",
}
FEAT_LABELS = {
    "device_class": "Device Risk Class", "advisory_committee_encoded": "Medical Specialty",
    "medical_specialty_encoded": "Medical Specialty", "advisory_committee_freq": "Committee Frequency",
    "is_us": "US Manufacturer", "review_days": "Review Duration",
    "decision_year": "Decision Year", "product_code_freq": "Product Code Frequency",
    "applicant_freq": "Applicant History", "applicant_submission_count": "Applicant Volume",
    "clearance_type_encoded": "Clearance Type", "third_party": "Third-Party Review",
    "country_freq": "Country Frequency", "month_sin": "Seasonal Pattern",
    "month_cos": "Seasonal Pattern", "has_review_days": "Review Data Present",
    "device_class_unknown": "Unknown Class",
}
PATHWAY_INFO = {
    "510k": {
        "full": "510(k) — Premarket Notification",
        "desc": "Demonstrates substantial equivalence to a legally marketed predicate device.",
        "review": "3–6 months", "trials": "Usually not required", "fee": "~$22,000",
        "icon_review": "schedule", "icon_trials": "device_hub", "icon_fee": "payments",
    },
    "PMA": {
        "full": "PMA — Premarket Approval",
        "desc": "Requires valid scientific evidence of safety and effectiveness.",
        "review": "12–18 months", "trials": "Required", "fee": "~$420,000",
        "icon_review": "schedule", "icon_trials": "biotech", "icon_fee": "payments",
    },
    "De_Novo": {
        "full": "De Novo — Classification Request",
        "desc": "For novel low/moderate risk devices with no predicate. Creates a new device type.",
        "review": "6–12 months", "trials": "Sometimes required", "fee": "~$130,000",
        "icon_review": "schedule", "icon_trials": "science", "icon_fee": "payments",
    },
}
CHECKLISTS = {
    "510k": [
        {
            "title": "Determine Device Classification",
            "content": "Your device has been classified and the 510(k) pathway has been identified as the appropriate regulatory route.",
            "tip": None, "done": True,
        },
        {
            "title": "Identify Predicate Device",
            "content": "Search the 510(k) database for cleared devices with similar intended use and technological characteristics.",
            "tip": "Use FDA's 510(k) database and product code search. You may cite multiple predicates.",
            "done": False,
        },
        {
            "title": "Prepare 510(k) Summary or Statement",
            "content": "Document the comparison between your device and the predicate, including substantial equivalence rationale.",
            "tip": "Prepare either a 510(k) Summary (publicly available) or a 510(k) Statement within 30 days of clearance.",
            "done": False,
        },
        {
            "title": "Compile Technical Documentation",
            "content": "Gather device description, labeling, performance testing data, biocompatibility, and software documentation.",
            "tip": "Follow FDA guidance: 'Recommended Content and Format of Non-Clinical Bench Performance Testing Information'.",
            "done": False,
        },
        {
            "title": "Complete Performance Testing",
            "content": "Conduct bench testing, biocompatibility (ISO 10993), sterilization validation, and EMC testing as applicable.",
            "tip": "Document all testing in a Device Master Record (DMR) per 21 CFR Part 820.",
            "done": False,
        },
        {
            "title": "Submit and Monitor Review",
            "content": "Submit the complete 510(k) package via FDA's CDRH eSubmitter or eCopy program.",
            "tip": "Track acceptance and substantive review status via FDA TPLC. Respond promptly to Additional Information (AI) requests.",
            "done": False,
        },
    ],
    "PMA": [
        {
            "title": "Determine Device Classification",
            "content": "Your device has been classified as Class III and the PMA pathway has been identified.",
            "tip": None, "done": True,
        },
        {
            "title": "Request Pre-Submission Meeting (Q-Sub)",
            "content": "Meet with FDA to align on clinical trial design, endpoints, and overall submission strategy.",
            "tip": "Submit a Q-Sub at least 90 days before you need the meeting. Strongly recommended for all PMA applicants.",
            "done": False,
        },
        {
            "title": "Design and Conduct Clinical Trials",
            "content": "Develop an IDE (Investigational Device Exemption) application for significant-risk devices.",
            "tip": "Ensure your clinical study meets FDA requirements for statistical power, endpoints, and IRB approval.",
            "done": False,
        },
        {
            "title": "Compile PMA Application",
            "content": "Prepare the full PMA including clinical data, technical documentation, labeling, and manufacturing information.",
            "tip": "Follow 21 CFR Part 814. The application typically exceeds 1,000 pages. Consider a modular PMA approach.",
            "done": False,
        },
        {
            "title": "Submit PMA and Respond to Panel/Requests",
            "content": "Submit via FDA's eSubmitter. FDA may convene an advisory panel meeting for complex devices.",
            "tip": "Total FDA review target is 180 days. Major Deficiency Letters (MDLs) pause the clock — respond within 180 days.",
            "done": False,
        },
        {
            "title": "Post-Approval Studies and Reporting",
            "content": "Comply with all post-approval conditions including MDRs, supplements, and annual reports.",
            "tip": "PMA holders must submit annual reports and any required post-approval studies per the approval order conditions.",
            "done": False,
        },
    ],
    "De_Novo": [
        {
            "title": "Determine Device Classification",
            "content": "Your device has been identified as novel and the De Novo pathway has been recommended.",
            "tip": None, "done": True,
        },
        {
            "title": "Confirm No Predicate Device Exists",
            "content": "Verify that no legally marketed device exists with the same intended use and technological characteristics.",
            "tip": "Search FDA 510(k), De Novo, and PMA databases. Also review published literature and international clearances.",
            "done": False,
        },
        {
            "title": "Develop Proposed Classification and Special Controls",
            "content": "Propose a device type name, classification (Class I or II), and specific special controls.",
            "tip": "Special controls are device-specific requirements that provide reasonable assurance of safety and effectiveness.",
            "done": False,
        },
        {
            "title": "Prepare De Novo Request Package",
            "content": "Compile device description, classification proposal, performance testing data, and proposed labeling.",
            "tip": "Follow FDA guidance: 'De Novo Classification Process (Evaluation of Automatic Class III Designation)'.",
            "done": False,
        },
        {
            "title": "Submit and Respond to FDA",
            "content": "Submit the De Novo request via FDA's CDRH eSubmitter.",
            "tip": "FDA has 150 days to respond. They may request additional information or convene an advisory panel.",
            "done": False,
        },
        {
            "title": "Leverage Grant Order for Future 510(k)s",
            "content": "Once De Novo is granted, your device becomes Class I or II and can serve as a predicate for future 510(k) submissions.",
            "tip": "The grant order establishes a new regulation for your device type. Monitor for 510(k)s citing your device as predicate.",
            "done": False,
        },
    ],
}

# ── Helper functions ───────────────────────────────────────────────────────────
def check_exemption(product_code=None, device_name=None):
    base = "https://api.fda.gov/device/classification.json"
    q = (f"product_code:{product_code}" if product_code
         else f"device_name:{device_name}" if device_name else None)
    if not q:
        return None
    try:
        r = requests.get(f"{base}?search={q}&limit=5", timeout=10)
        if r.status_code == 200:
            res = r.json().get("results", [])
            return res[0] if res else None
    except Exception:
        pass
    return None


def _conf_circle(confidence):
    return f"""
    <div style="position:relative;width:120px;height:120px;flex-shrink:0;">
        <svg width="120" height="120" viewBox="0 0 36 36"
             style="transform:rotate(-90deg);display:block;">
            <circle cx="18" cy="18" r="15.9155" fill="none"
                stroke="rgba(255,255,255,0.15)" stroke-width="2.8"/>
            <circle cx="18" cy="18" r="15.9155" fill="none"
                stroke="white" stroke-width="2.8"
                stroke-dasharray="{confidence:.1f} 100"
                stroke-linecap="round"/>
        </svg>
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                    text-align:center;pointer-events:none;">
            <div style="font-size:24px;font-weight:700;color:white;line-height:1.1;">{confidence:.0f}%</div>
            <div style="font-size:10px;color:rgba(255,255,255,.65);margin-top:2px;white-space:nowrap;">confidence</div>
        </div>
    </div>"""


def _dot_bar(imp, max_imp, label):
    filled = max(1, round(imp / max_imp * 4)) if max_imp > 0 else 0
    dots = "".join(
        f'<span class="d-f"></span>' if i < filled else f'<span class="d-e"></span>'
        for i in range(4)
    )
    return f'<div class="feat-row"><span class="feat-nm">{label}</span><div class="feat-dots">{dots}</div></div>'


def _icon_row(icon, label, value):
    return (
        f'<div class="ir">'
        f'<div class="iw"><span class="material-icons-round">{icon}</span></div>'
        f'<div><div class="it-lbl">{label}</div><div class="it-val">{value}</div></div>'
        f'</div>'
    )


def _pw_bar(name, pct, color):
    return (
        f'<div class="pw-bar-row">'
        f'<div class="pw-bar-hd"><span>{name}</span><span>{pct}%</span></div>'
        f'<div class="pw-bar-t"><div class="pw-bar-f" style="width:{min(pct,100)}%;background:{color};"></div></div>'
        f'</div>'
    )


def _compliance_notices():
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("""
        <div class="ntc ntc-w">
            <div style="font-size:18px;flex-shrink:0;margin-top:1px;">&#9762;</div>
            <div>
                <div class="ntc-ttl">Radiation-emitting product?</div>
                <div class="ntc-txt">If your device emits radiation, it must also comply with 21 CFR parts 1000–1050, regardless of pathway.</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with nc2:
        st.markdown("""
        <div class="ntc ntc-i">
            <div style="font-size:18px;flex-shrink:0;margin-top:1px;">&#128138;</div>
            <div>
                <div class="ntc-ttl">Combination product?</div>
                <div class="ntc-txt">If your product combines a device with a drug or biologic, contact FDA's Office of Combination Products at combination@fda.gov.</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:8px 0 16px;">
    <div style="font-size:20px;font-weight:700;color:white;letter-spacing:-.3px;">FDA Pathway</div>
    <div style="font-size:20px;font-weight:300;color:rgba(255,255,255,.6);">Advisor</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGES = ["Pathway Predictor", "Preparation Checklist", "EDA Dashboard", "Model Performance"]
nav_idx = PAGES.index(st.session_state.nav_page) if st.session_state.nav_page in PAGES else 0
page = st.sidebar.radio("Navigate", PAGES, index=nav_idx)
st.session_state.nav_page = page

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:12px;color:rgba(255,255,255,.45);line-height:1.7;">'
    'Data: openFDA API<br>Model: RF + Gradient Boosting<br>Built with CrewAI + scikit-learn</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PATHWAY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Pathway Predictor":
    model = load_model()
    encoders = load_encoders()
    meta = load_meta()
    features_df = load_features()

    st.markdown(
        '<h1 style="font-size:28px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'FDA Regulatory Pathway Advisor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;margin-bottom:28px;font-size:15px;">'
        'Enter your device details to get an AI-powered regulatory pathway recommendation.</p>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("No trained model found. Run the pipeline first: `python -m src.flow.main_flow`")
        st.stop()

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    if not st.session_state.show_results:
        col_form, col_info = st.columns([3, 1], gap="large")

        with col_form:
            st.markdown('<div class="fda-card">', unsafe_allow_html=True)
            st.markdown("#### Device Information")
            device_name = st.text_input(
                "Device Name",
                placeholder="e.g. Wireless cardiac rhythm monitor",
                key="inp_dn",
            )
            st.text_area(
                "Device Description",
                placeholder="Describe your device's intended use, technology, and mechanism of action",
                height=100,
                key="inp_dd",
            )

            st.markdown("<br>**Device Risk Profile**", unsafe_allow_html=True)
            cls_cols = st.columns(3)
            cls_defs = [
                (1, cls_cols[0], "#27AE60", "sel-1", "Low risk\nGeneral Controls only"),
                (2, cls_cols[1], "#378ADD", "sel-2", "Moderate risk\nGeneral + Special Controls"),
                (3, cls_cols[2], "#D85A30", "sel-3", "High risk\nPMA required"),
            ]
            for cls, col, color, sel_cls, sub in cls_defs:
                with col:
                    is_sel = st.session_state.device_class == cls
                    card_cls = sel_cls if is_sel else ""
                    lines = sub.split("\n")
                    st.markdown(
                        f'<div class="cls-card {card_cls}">'
                        f'<span class="cls-dot" style="background:{color};"></span>'
                        f'<div style="font-weight:700;font-size:14px;color:#1e293b;">Class {cls}</div>'
                        f'<div style="font-size:11px;color:#64748b;text-align:center;line-height:1.4;">'
                        f'{lines[0]}<br>{lines[1]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    label = "✓ Selected" if is_sel else f"Select Class {cls}"
                    btn_type = "primary" if is_sel else "secondary"
                    if st.button(label, key=f"cls_btn_{cls}", use_container_width=True, type=btn_type):
                        st.session_state.device_class = cls
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            acol, ccol = st.columns(2)
            with acol:
                advisory_committees = sorted(
                    k for k in encoders.get("advisory_committee", {}).keys() if k != "UNKNOWN"
                )
                if not advisory_committees:
                    advisory_committees = [
                        "AN", "CV", "CH", "DE", "EN", "GU", "HO", "IM",
                        "MI", "NE", "OB", "OP", "OR", "PA", "PM", "RA", "SU", "TX",
                    ]
                advisory_committee = st.selectbox(
                    "Advisory Committee / Medical Specialty",
                    advisory_committees, key="inp_ac",
                )
            with ccol:
                country_label = st.selectbox("Country of Origin", list(COUNTRY_MAP.keys()), key="inp_cn")
            country_code = COUNTRY_MAP[country_label]
            has_predicate = st.checkbox(
                "This device has a substantially equivalent predicate device on the market",
                key="inp_hp",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown("""
            <div class="fda-card">
                <div style="font-size:12px;font-weight:700;color:#002046;margin-bottom:12px;
                            text-transform:uppercase;letter-spacing:.06em;">About This Tool</div>
                <p style="font-size:13px;color:#374151;line-height:1.65;margin:0 0 14px;">
                Uses machine learning trained on 200,000+ historical FDA submissions to predict
                the most likely regulatory pathway for your device.</p>
                <hr style="border:none;border-top:1px solid #f1f5f9;margin:0 0 14px;">
                <div style="font-size:13px;font-weight:600;color:#1e293b;margin-bottom:10px;">
                Supported Pathways</div>
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">
                    <span style="width:9px;height:9px;border-radius:50%;background:#378ADD;flex-shrink:0;"></span>
                    <div><div style="font-size:13px;font-weight:600;color:#1e293b;">510(k)</div>
                    <div style="font-size:12px;color:#64748b;">Premarket Notification</div></div>
                </div>
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">
                    <span style="width:9px;height:9px;border-radius:50%;background:#D85A30;flex-shrink:0;"></span>
                    <div><div style="font-size:13px;font-weight:600;color:#1e293b;">PMA</div>
                    <div style="font-size:12px;color:#64748b;">Premarket Approval</div></div>
                </div>
                <div style="display:flex;align-items:center;gap:9px;">
                    <span style="width:9px;height:9px;border-radius:50%;background:#1D9E75;flex-shrink:0;"></span>
                    <div><div style="font-size:13px;font-weight:600;color:#1e293b;">De Novo</div>
                    <div style="font-size:12px;color:#64748b;">Classification Request</div></div>
                </div>
            </div>
            <div class="fda-card" style="margin-top:0;">
                <div style="font-size:12px;font-weight:700;color:#92400e;margin-bottom:7px;">Disclaimer</div>
                <p style="font-size:12px;color:#64748b;line-height:1.65;margin:0;">
                This tool provides decision support only. It is not regulatory advice.
                Always consult a qualified regulatory professional before submitting to FDA.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Regulatory Pathway →", type="primary", use_container_width=True, key="analyze_btn"):

            # Stage 1: Exemption check via FDA API
            if device_name:
                with st.spinner("Checking FDA classification database..."):
                    classification = check_exemption(device_name=device_name)
                if classification:
                    dev_cls = str(classification.get("device_class", ""))
                    sub_type = classification.get("submission_type_id", "").strip()
                    if dev_cls in ("1", "2") and sub_type == "":
                        pathway_label = "Class I Exempt" if dev_cls == "1" else "Class II Exempt"
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#065f46,#047857);
                            border-radius:16px;padding:36px 44px;color:white;margin-bottom:24px;">
                            <div style="font-size:12px;font-weight:600;letter-spacing:.08em;
                                text-transform:uppercase;color:rgba(255,255,255,.65);margin-bottom:8px;">
                                FDA Classification Result</div>
                            <div style="font-size:36px;font-weight:700;margin-bottom:6px;">
                                No Premarket Submission Required</div>
                            <div style="font-size:15px;color:rgba(255,255,255,.75);">
                                {pathway_label} — Exempt from 510(k) requirements</div>
                        </div>
                        """, unsafe_allow_html=True)
                        el, er = st.columns(2)
                        with el:
                            st.subheader("What This Means")
                            st.success(
                                "Your device is **exempt** from premarket notification. "
                                "You must still comply with General Controls:"
                            )
                            st.markdown(
                                "- Quality System Regulation (21 CFR Part 820)\n"
                                "- Establishment Registration\n"
                                "- Device Listing\n"
                                "- Labeling Requirements (21 CFR Part 801)\n"
                                "- Medical Device Reporting (21 CFR Part 803)"
                            )
                            st.warning(
                                "If your device exceeds the limitations of exemption stated in "
                                "21 CFR xxx.9, a 510(k) may still be required."
                            )
                        with er:
                            st.subheader("Device Classification")
                            st.markdown(f"**Product Code:** `{classification.get('product_code', 'N/A')}`")
                            st.markdown(f"**Device Class:** {dev_cls}")
                            st.markdown(f"**Regulation Number:** {classification.get('regulation_number', 'N/A')}")
                            if classification.get("device_name"):
                                st.markdown(f"**Matched Device:** {classification['device_name']}")
                        st.markdown("---")
                        _compliance_notices()
                        st.stop()

            # Stage 2: ML prediction
            device_class = st.session_state.device_class
            feature_cols = meta.get("feature_columns", [])
            if not feature_cols:
                st.error("Model metadata missing. Run the pipeline first.")
                st.stop()

            def _fmed(col):
                return (
                    features_df[col].median()
                    if features_df is not None and col in features_df.columns
                    else 0
                )

            ac_enc = encoders.get("advisory_committee", {}).get(advisory_committee, 0)
            ms_enc = encoders.get("medical_specialty", {}).get(advisory_committee, 0)
            ct_enc = encoders.get("clearance_type", {}).get(
                "Traditional" if has_predicate else "UNKNOWN", 0
            )
            input_dict = {
                "device_class": device_class, "device_class_unknown": 0,
                "advisory_committee_freq": _fmed("advisory_committee_freq"),
                "advisory_committee_encoded": ac_enc,
                "medical_specialty_freq": _fmed("medical_specialty_freq"),
                "medical_specialty_encoded": ms_enc,
                "is_us": 1 if country_code == "US" else 0,
                "country_freq": _fmed("country_freq"),
                "decision_year": 2025, "decision_month": 6,
                "month_sin": np.sin(2 * np.pi * 6 / 12),
                "month_cos": np.cos(2 * np.pi * 6 / 12),
                "review_days": _fmed("review_days"), "has_review_days": 1,
                "clearance_type_encoded": ct_enc, "third_party": 0,
                "product_code_freq": _fmed("product_code_freq"),
                "applicant_freq": _fmed("applicant_freq"),
                "applicant_submission_count": _fmed("applicant_submission_count"),
            }
            X = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            pathway = encoders.get("pathway_inverse", {}).get(str(pred), "510k")

            st.session_state.last_pathway = pathway
            st.session_state.last_proba = proba.tolist()
            st.session_state.last_confidence = float(max(proba) * 100)
            st.session_state.last_device_class = device_class
            st.session_state.last_advisory = advisory_committee
            st.session_state.last_feature_cols = feature_cols
            st.session_state.show_results = True
            st.rerun()

    # ── RESULTS ────────────────────────────────────────────────────────────────
    else:
        pathway = st.session_state.last_pathway
        proba = st.session_state.last_proba
        confidence = st.session_state.last_confidence
        device_class = st.session_state.last_device_class
        advisory_committee = st.session_state.last_advisory
        feature_cols = st.session_state.last_feature_cols
        info = PATHWAY_INFO.get(pathway, PATHWAY_INFO["510k"])
        encoders = load_encoders()
        model = load_model()

        # Results banner
        st.markdown(
            f'<div class="res-banner">'
            f'<div style="flex:1;">'
            f'<div class="pw-badge">Recommended Pathway</div>'
            f'<div class="pw-name">{pathway.replace("_", " ")}</div>'
            f'<div class="pw-sub">{info["full"]}</div>'
            f'<div style="margin-top:10px;font-size:12px;color:rgba(255,255,255,.5);">'
            f'Based on 200,000+ historical FDA submissions</div>'
            f'</div>'
            f'{_conf_circle(confidence)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Three info cards
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="ic"><div class="ic-title">Confidence Breakdown</div>', unsafe_allow_html=True)
            pathway_order = sorted(encoders.get("pathway", {}).items(), key=lambda x: x[1])
            for cn, ci in pathway_order:
                if ci < len(proba):
                    pct = proba[ci] * 100
                    bar_color = COLORS.get(cn, "#94a3b8")
                    opacity = "1" if cn == pathway else ".45"
                    weight = "font-weight:700;" if cn == pathway else ""
                    st.markdown(
                        f'<div class="pw-bar-row">'
                        f'<div class="pw-bar-hd"><span style="{weight}">{cn.replace("_", " ")}</span>'
                        f'<span style="{weight}">{pct:.1f}%</span></div>'
                        f'<div class="pw-bar-t"><div class="pw-bar-f" '
                        f'style="width:{min(pct,100):.1f}%;background:{bar_color};opacity:{opacity};"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="ic"><div class="ic-title">Key Decision Factors</div>', unsafe_allow_html=True)
            if hasattr(model, "feature_importances_") and feature_cols:
                imp_series = (
                    pd.Series(model.feature_importances_, index=feature_cols)
                    .sort_values(ascending=False)
                    .head(5)
                )
                max_imp = imp_series.max()
                dots_html = "".join(
                    _dot_bar(v, max_imp, FEAT_LABELS.get(k, k.replace("_", " ").title()))
                    for k, v in imp_series.items()
                )
                st.markdown(dots_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="font-size:13px;color:#64748b;">Feature importance not available.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown(
                f'<div class="ic">'
                f'<div class="ic-title">Pathway Details</div>'
                f'<p style="font-size:13px;color:#64748b;margin-bottom:18px;line-height:1.55;">{info["desc"]}</p>'
                f'{_icon_row(info["icon_review"], "Typical Review Time", info["review"])}'
                f'{_icon_row(info["icon_trials"], "Clinical Trials", info["trials"])}'
                f'{_icon_row(info["icon_fee"],    "FDA User Fee",      info["fee"])}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Similar devices
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            from src.tools.similar_devices import find_similar_devices
            sim = find_similar_devices(
                device_class=device_class,
                advisory_committee=advisory_committee,
                data_path=str(ARTIFACTS / "clean_data.csv"),
            )
        except Exception:
            sim = None

        if sim:
            st.markdown('<div class="fda-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sim-hd">Historical Precedent</div>'
                f'<div class="sim-sub">Found <strong>{sim["total_similar"]:,}</strong> similar devices '
                f'(Class {device_class}, {advisory_committee} committee)</div>',
                unsafe_allow_html=True,
            )
            sc1, sc2 = st.columns(2, gap="large")
            with sc1:
                st.markdown("**Pathway Distribution**")
                for pname, pct in sim["pathway_distribution"].items():
                    st.markdown(_pw_bar(pname, pct, COLORS.get(pname, "#94a3b8")), unsafe_allow_html=True)
            with sc2:
                st.markdown("**Historical Clearance Rates**")
                for pname, sr in sim["success_rates"].items():
                    rate_color = (
                        "#1D9E75" if sr["rate"] >= 70
                        else "#D85A30" if sr["rate"] < 40
                        else "#F59E0B"
                    )
                    st.markdown(
                        f'<div style="margin-bottom:12px;">'
                        f'<div style="font-size:13px;font-weight:600;color:#1e293b;">{pname}</div>'
                        f'<div style="font-size:26px;font-weight:700;color:{rate_color};line-height:1.1;">{sr["rate"]}%</div>'
                        f'<div style="font-size:12px;color:#64748b;">{sr["cleared"]:,} of {sr["total"]:,} cleared</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("View example devices"):
                st.dataframe(pd.DataFrame(sim["examples"]), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        ab1, ab2, ab3 = st.columns(3, gap="medium")
        with ab1:
            report_lines = [
                f"# FDA Pathway Analysis Report\n",
                f"**Recommended Pathway:** {pathway}\n",
                f"**Confidence:** {confidence:.1f}%\n\n",
                f"## Pathway Details\n",
                f"**{info['full']}**\n\n{info['desc']}\n\n",
                f"- Review Time: {info['review']}\n",
                f"- Clinical Trials: {info['trials']}\n",
                f"- FDA User Fee: {info['fee']}\n\n",
                f"## Probability Breakdown\n",
            ]
            for cn, ci in sorted(encoders.get("pathway", {}).items(), key=lambda x: x[1]):
                if ci < len(proba):
                    report_lines.append(f"- {cn}: {proba[ci]*100:.1f}%\n")
            report_lines.append(
                "\n---\n*This report is for informational purposes only "
                "and does not constitute regulatory advice.*"
            )
            st.download_button(
                "Download Report",
                data="".join(report_lines),
                file_name="fda_pathway_analysis.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with ab2:
            if st.button("Start New Analysis", use_container_width=True, key="new_analysis_btn"):
                st.session_state.show_results = False
                st.rerun()
        with ab3:
            checklist_label = f"View {pathway.replace('_', ' ')} Checklist →"
            if st.button(checklist_label, type="primary", use_container_width=True, key="view_checklist_btn"):
                st.session_state.nav_page = "Preparation Checklist"
                st.rerun()

        # Compliance notices
        st.markdown("---")
        _compliance_notices()
        st.caption(
            "This prediction is based on historical FDA data patterns. "
            "It should not replace professional regulatory advice."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREPARATION CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Preparation Checklist":
    pathway = st.session_state.get("last_pathway") or "510k"
    steps = CHECKLISTS.get(pathway, CHECKLISTS["510k"])
    pw_display = pathway.replace("_", " ")
    color = COLORS.get(pathway, "#378ADD")

    st.markdown(
        f'<h1 style="font-size:26px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        f'{pw_display} Preparation Checklist</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#64748b;margin-bottom:20px;font-size:15px;">'
        f'Step-by-step guide to prepare your {pw_display} submission.</p>',
        unsafe_allow_html=True,
    )

    # Pathway switcher if no prediction exists yet
    if not st.session_state.get("last_pathway"):
        sw1, sw2, sw3 = st.columns(3)
        for col, pw in [(sw1, "510k"), (sw2, "PMA"), (sw3, "De_Novo")]:
            with col:
                if st.button(
                    f"View {pw.replace('_', ' ')} Checklist",
                    key=f"sw_{pw}", use_container_width=True,
                    type="primary" if pw == pathway else "secondary",
                ):
                    st.session_state.last_pathway = pw
                    st.rerun()
        st.markdown("---")

    cl_left, cl_right = st.columns([2, 1], gap="large")

    with cl_left:
        # Build stepper HTML
        html = '<div style="padding:4px 0;">'
        for i, step in enumerate(steps):
            is_done = step["done"]
            is_last = i == len(steps) - 1
            circle_cls = "s-done" if is_done else "s-pending"
            lbl_cls = "s-lbl-d" if is_done else "s-lbl-p"
            lbl_text = "Completed" if is_done else f"Step {i + 1} of {len(steps)}"
            circle_icon = "&#10003;" if is_done else str(i + 1)
            line_cls = "done-l" if is_done and not is_last else ""
            tip_html = (
                f'<div class="s-tip">{step["tip"]}</div>'
                if step.get("tip") else ""
            )
            html += (
                f'<div class="step-wrap">'
                f'  <div class="step-left">'
                f'    <div class="s-circle {circle_cls}">{circle_icon}</div>'
                f'    {"" if is_last else f\'<div class="s-line {line_cls}"></div>\'}'
                f'  </div>'
                f'  <div class="s-content">'
                f'    <div class="s-lbl {lbl_cls}">{lbl_text}</div>'
                f'    <div class="s-ttl">{step["title"]}</div>'
                f'    <div class="s-desc">{step["content"]}</div>'
                f'    {tip_html}'
                f'  </div>'
                f'</div>'
            )
        html += "</div>"

        st.markdown(f'<div class="fda-card">{html}</div>', unsafe_allow_html=True)

        if st.button("← Back to Pathway Predictor", key="back_to_pred"):
            st.session_state.nav_page = "Pathway Predictor"
            st.rerun()

    with cl_right:
        # Key resources
        resources = {
            "510k": [
                ("510(k) Guidance Overview",
                 "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k"),
                ("Search 510(k) Database",
                 "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"),
                ("Special 510(k) Program",
                 "https://www.fda.gov/medical-devices/premarket-notification-510k/special-510k-program"),
                ("eSubmitter Tool",
                 "https://www.fda.gov/industry/fda-esubmitter"),
            ],
            "PMA": [
                ("PMA Guidance Overview",
                 "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-approval-pma"),
                ("Search PMA Database",
                 "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm"),
                ("IDE Guidance",
                 "https://www.fda.gov/medical-devices/how-study-and-market-your-device/investigational-device-exemption-ide"),
                ("Q-Submission Program",
                 "https://www.fda.gov/medical-devices/how-study-and-market-your-device/q-submission-program"),
            ],
            "De_Novo": [
                ("De Novo Guidance",
                 "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request"),
                ("Search De Novo Database",
                 "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm"),
                ("Q-Submission Program",
                 "https://www.fda.gov/medical-devices/how-study-and-market-your-device/q-submission-program"),
            ],
        }
        timelines = {
            "510k": [
                ("Months 1–3", "Predicate search & bench testing"),
                ("Months 3–6", "Technical documentation"),
                ("Month 6", "Submission to FDA"),
                ("Months 7–12", "FDA substantive review"),
            ],
            "PMA": [
                ("Months 1–6", "Q-Sub meeting & IDE prep"),
                ("Months 6–30", "Clinical trial conduct"),
                ("Months 30–36", "PMA compilation & submission"),
                ("Months 36–54", "FDA review & approval"),
            ],
            "De_Novo": [
                ("Months 1–6", "Novelty confirmation & special controls"),
                ("Months 6–12", "Package preparation & submission"),
                ("Months 12–24", "FDA review & grant order"),
            ],
        }
        res_links = "".join(
            f'<a href="{url}" target="_blank" style="display:block;font-size:13px;'
            f'color:#378ADD;text-decoration:none;margin-bottom:6px;">{label}</a>'
            for label, url in resources.get(pathway, resources["510k"])
        )
        tl_rows = "".join(
            f'<div style="margin-bottom:7px;">'
            f'<span style="font-size:12px;font-weight:700;color:#002046;">{period}:</span>'
            f'<span style="font-size:13px;color:#374151;"> {desc}</span></div>'
            for period, desc in timelines.get(pathway, timelines["510k"])
        )
        st.markdown(
            f'<div class="fda-card">'
            f'<div class="ic-title">Key Resources</div>'
            f'{res_links}'
            f'</div>'
            f'<div class="fda-card" style="margin-top:0;">'
            f'<div class="ic-title">Typical Timeline</div>'
            f'{tl_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div class="ntc ntc-w">
            <div style="font-size:17px;flex-shrink:0;margin-top:1px;">&#9888;</div>
            <div>
                <div class="ntc-ttl">Not Regulatory Advice</div>
                <div class="ntc-txt">Requirements vary by device type. Consult a qualified regulatory professional.</div>
            </div>
        </div>
        <div class="ntc ntc-i">
            <div style="font-size:17px;flex-shrink:0;margin-top:1px;">&#8505;</div>
            <div>
                <div class="ntc-ttl">FDA Pre-Submission Programs</div>
                <div class="ntc-txt">For complex submissions, consider requesting a Q-Sub meeting with FDA before you file.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    st.markdown(
        '<h1 style="font-size:26px;font-weight:700;color:#0f172a;">Exploratory Data Analysis</h1>',
        unsafe_allow_html=True,
    )
    df = load_clean()
    if df is not None and "decision_year" in df.columns:
        min_y = int(df["decision_year"].min())
        max_y = int(df["decision_year"].max())
        yr = st.slider("Year Range", min_y, max_y, (min_y, max_y))
        df_f = df[(df["decision_year"] >= yr[0]) & (df["decision_year"] <= yr[1])]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Submissions", f"{len(df_f):,}")
        m2.metric("510(k)", f"{(df_f['pathway'] == '510k').sum():,}")
        m3.metric("PMA", f"{(df_f['pathway'] == 'PMA').sum():,}")
        m4.metric("De Novo", f"{(df_f['pathway'] == 'De_Novo').sum():,}")

    ins = ARTIFACTS / "insights.md"
    if ins.exists():
        st.markdown(ins.read_text())

    eda = ARTIFACTS / "eda_report.html"
    if eda.exists():
        st.subheader("Full EDA Report")
        st.components.v1.html(eda.read_text(), height=3000, scrolling=True)

    cont = load_contract()
    if cont:
        with st.expander("Dataset Contract"):
            st.json(cont)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    import re as _re

    st.markdown(
        '<h1 style="font-size:26px;font-weight:700;color:#0f172a;">Model Performance</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;margin-bottom:24px;font-size:15px;">'
        'Comparison of Random Forest vs. Gradient Boosting trained on 200,000+ FDA submissions.</p>',
        unsafe_allow_html=True,
    )

    meta = load_meta()
    best = meta.get("best_model", "Random Forest") if meta else "Random Forest"
    feat_count = len(meta.get("feature_columns", [])) if meta else 0

    # ── Summary chips ─────────────────────────────────────────────────────────
    def _chip(label, value):
        return (
            f'<div style="display:inline-flex;flex-direction:column;gap:2px;'
            f'padding:12px 20px;border-right:1px solid #f1f5f9;">'
            f'<span style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:.05em;">{label}</span>'
            f'<span style="font-size:16px;font-weight:700;color:#0f172a;">{value}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="fda-card" style="padding:0;display:flex;flex-wrap:wrap;margin-bottom:24px;">'
        f'{_chip("Best Model", best)}'
        f'{_chip("Features", feat_count)}'
        f'{_chip("Training Records", "200,000+")}'
        f'{_chip("Classes", "510(k) · PMA · De Novo")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Metrics comparison table ───────────────────────────────────────────────
    ev = ARTIFACTS / "evaluation_report.md"
    if ev.exists():
        ev_text = ev.read_text()
        # Parse markdown table rows: lines starting with | that aren't separators
        table_rows = [
            [c.strip() for c in line.split("|")[1:-1]]
            for line in ev_text.split("\n")
            if line.startswith("|") and "---" not in line
        ]
        # table_rows[0] = header, table_rows[1:] = data
        if len(table_rows) >= 2:
            headers = table_rows[0]   # ["Metric", "Random Forest", "Gradient Boosting"]
            data_rows = table_rows[1:]
            rf_col = "Random Forest"
            gb_col = "Gradient Boosting"
            rf_is_best = best == "Random Forest"

            def _val_color(val_str):
                try:
                    v = float(val_str)
                    if v >= 0.90: return "#1D9E75"
                    if v >= 0.80: return "#378ADD"
                    if v >= 0.70: return "#F59E0B"
                    return "#D85A30"
                except (ValueError, TypeError):
                    return "#64748b"

            rows_html = ""
            for i, row in enumerate(data_rows):
                if len(row) < 3:
                    continue
                metric, rf_val, gb_val = row[0], row[1], row[2]
                rf_color = _val_color(rf_val)
                gb_color = _val_color(gb_val)
                bg = "white" if i % 2 == 0 else "#f8fafc"
                rf_badge = (
                    f'&nbsp;<span style="background:#002046;color:white;font-size:10px;'
                    f'font-weight:700;padding:2px 7px;border-radius:20px;">Best</span>'
                    if rf_is_best else ""
                )
                gb_badge = (
                    f'&nbsp;<span style="background:#002046;color:white;font-size:10px;'
                    f'font-weight:700;padding:2px 7px;border-radius:20px;">Best</span>'
                    if not rf_is_best else ""
                )
                rows_html += (
                    f'<tr style="background:{bg};">'
                    f'<td style="padding:12px 16px;font-size:13px;font-weight:600;color:#374151;">{metric}</td>'
                    f'<td style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:{rf_color};">'
                    f'{rf_val}{rf_badge}</td>'
                    f'<td style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:{gb_color};">'
                    f'{gb_val}{gb_badge}</td>'
                    f'</tr>'
                )

            table_html = (
                f'<div class="fda-card" style="padding:0;overflow:hidden;margin-bottom:24px;">'
                f'<div style="padding:16px 20px;border-bottom:1px solid #f1f5f9;">'
                f'<div class="ic-title" style="margin:0;">Model Comparison</div></div>'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr style="background:#f8fafc;">'
                f'<th style="padding:10px 16px;text-align:left;font-size:11px;font-weight:700;'
                f'color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Metric</th>'
                f'<th style="padding:10px 16px;text-align:center;font-size:11px;font-weight:700;'
                f'color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Random Forest</th>'
                f'<th style="padding:10px 16px;text-align:center;font-size:11px;font-weight:700;'
                f'color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Gradient Boosting</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table></div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

        # Show classification report in expander
        cr_match = _re.search(r"## Classification Report.*?```\n(.*?)```", ev_text, _re.DOTALL)
        if cr_match:
            with st.expander(f"Classification Report — {best}"):
                st.code(cr_match.group(1), language=None)

    # ── Confusion matrices + feature importance ────────────────────────────────
    img1 = ARTIFACTS / "confusion_matrices.png"
    img2 = ARTIFACTS / "feature_importance.png"
    if img1.exists() or img2.exists():
        ic1, ic2 = st.columns(2, gap="large")
        if img1.exists():
            with ic1:
                st.markdown(
                    '<div class="fda-card" style="padding:16px 20px 20px;">'
                    '<div class="ic-title">Confusion Matrices</div>',
                    unsafe_allow_html=True,
                )
                st.image(str(img1), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        if img2.exists():
            with ic2:
                st.markdown(
                    '<div class="fda-card" style="padding:16px 20px 20px;">'
                    '<div class="ic-title">Feature Importance</div>',
                    unsafe_allow_html=True,
                )
                st.image(str(img2), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Model card ────────────────────────────────────────────────────────────
    mc = ARTIFACTS / "model_card.md"
    if mc.exists():
        st.markdown('<div class="fda-card" style="margin-top:8px;">', unsafe_allow_html=True)
        st.markdown(mc.read_text())
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Export button ─────────────────────────────────────────────────────────
    if ev.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        export_col, _ = st.columns([1, 3])
        with export_col:
            report_parts = [ev.read_text()]
            if mc.exists():
                report_parts.append("\n\n---\n\n" + mc.read_text())
            st.download_button(
                "Export Full Technical Report",
                data="\n".join(report_parts),
                file_name="fda_model_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
