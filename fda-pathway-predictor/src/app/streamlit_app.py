"""FDA Pathway Predictor — Streamlit App (redesigned to match HTML mockups)."""
import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, requests, base64, logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
<style>
@font-face {
    font-family: 'Asenine';
    src: url('/app/static/ASENINE_.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}
@font-face {
    font-family: 'Asenine';
    src: url('/app/static/ASENW___.ttf') format('truetype');
    font-weight: bold;
    font-style: normal;
}
html, body, [class*="css"], h1, h2, h3, h4, p, div, button,
.stMarkdown, .stText, label { font-family: 'Inter', sans-serif !important; }
/* Headlines use Asenine */
h1, h2, h3,
[data-testid="stHeading"],
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Asenine', sans-serif !important;
    letter-spacing: 0.02em !important;
}
/* Exclude Material Icon spans from Inter override */
span:not([data-testid="stIconMaterial"]):not(.material-icons-round):not(.material-icons) { font-family: 'Inter', sans-serif !important; }
.material-icons-round, .material-icons {
    font-family: 'Material Icons Round', 'Material Symbols Rounded', sans-serif !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons Round', sans-serif !important;
    font-size: 20px !important;
    line-height: 1 !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #FAF8F5 !important; border-right: 1px solid #E8E2D9 !important; }
section[data-testid="stSidebar"] * { color: #1B365D !important; }
section[data-testid="stSidebar"] .stRadio label { color: #334155 !important; font-size: 14px; }
section[data-testid="stSidebar"] hr { border-color: #E8E2D9 !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #334155 !important; }

/* Sidebar collapse button icon */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons Round', sans-serif !important;
    font-size: 22px !important;
    line-height: 1 !important;
    color: #64748b !important;
}

/* Sidebar nav — smooth clickable rows */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    border-radius: 8px !important;
    padding: 8px 10px !important;
    margin: 2px 0 !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
    background: rgba(0,32,70,0.07) !important;
}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:active {
    background: rgba(0,32,70,0.14) !important;
}
/* Hide the radio dot — use row highlight instead */
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] div:first-child {
    display: none !important;
}
/* Selected row highlight */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(0,32,70,0.10) !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) span {
    color: #002046 !important;
    font-weight: 600 !important;
}
/* Left accent bar on selected item */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    border-left: 3px solid #002046 !important;
    padding-left: 7px !important;
}

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

/* Expander — hide default SVG arrow, inject minimalist chevron */
[data-testid="stExpander"] details > summary svg { display: none !important; }
[data-testid="stExpander"] details > summary > svg { display: none !important; }
[data-testid="stExpander"] details > summary span svg { display: none !important; }
[data-testid="stExpander"] details > summary::-webkit-details-marker { display: none; }
[data-testid="stExpander"] details > summary { list-style: none; position: relative; padding-right: 28px !important; }
[data-testid="stExpander"] details > summary::after {
    content: '';
    position: absolute;
    right: 10px; top: 50%;
    width: 7px; height: 7px;
    border-right: 1.5px solid #94a3b8;
    border-bottom: 1.5px solid #94a3b8;
    transform: translateY(-70%) rotate(-45deg);
    transition: transform 0.18s ease;
    pointer-events: none;
}
[data-testid="stExpander"] details[open] > summary::after {
    transform: translateY(-30%) rotate(45deg);
}
</style>
""")

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
    "Not Specified": None, "United States": "US", "Germany": "DE", "Japan": "JP", "United Kingdom": "GB",
    "France": "FR", "Israel": "IL", "Canada": "CA", "Switzerland": "CH",
    "Ireland": "IE", "Other": "OTHER",
}
FEAT_LABELS = {
    "device_class": "Device Risk Class", "advisory_committee_encoded": "Medical Specialty",
    "medical_specialty_encoded": "Medical Specialty (Alt)", "advisory_committee_freq": "Committee Frequency",
    "is_us": "US Manufacturer", "review_days": "Review Duration",
    "decision_year": "Decision Year", "product_code_freq": "Product Code Frequency",
    "applicant_freq": "Applicant History", "applicant_submission_count": "Applicant Volume",
    "clearance_type_encoded": "Clearance Type", "third_party": "Third-Party Review",
    "country_freq": "Country Frequency", "month_sin": "Seasonal Pattern (sin)",
    "month_cos": "Seasonal Pattern (cos)", "has_review_days": "Review Data Present",
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
def _sidebar_logo() -> str:
    """Return base64-encoded logo, or empty string if not found."""
    for candidate in [
        Path(__file__).parent.parent.parent.parent / "logo_regula_BW.png",
        Path(__file__).parent.parent.parent / "logo_regula_BW.png",
        Path("logo_regula_BW.png"),
        Path(__file__).parent / "logo_regula_BW.png",
    ]:
        if candidate.exists():
            return base64.b64encode(candidate.read_bytes()).decode()
    return ""

_logo_b64 = _sidebar_logo()
if _logo_b64:
    st.sidebar.markdown(
        f'<div style="padding:16px 8px 12px;text-align:center;">'
        f'<img src="data:image/png;base64,{_logo_b64}" '
        f'style="width:90%;max-width:210px;opacity:0.9;"></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown("""
<div style="padding:8px 0 16px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:22px;line-height:1;color:rgba(255,255,255,.85);">&#9672;</span>
        <div style="font-size:20px;font-weight:700;color:white;letter-spacing:-.3px;">FDA Pathway Advisor</div>
    </div>
    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
                color:rgba(255,255,255,.45);padding-left:36px;">Regulatory Intelligence</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGES = ["Device Classification", "Pathway Predictor", "Preparation Checklist", "EDA Dashboard", "Model Performance"]
# Resolve any programmatic navigation requests before the radio renders
if "_nav_target" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_nav_target")
if st.session_state.get("nav_page") not in PAGES:
    st.session_state["nav_page"] = "Pathway Predictor"
page = st.sidebar.radio("Navigate", PAGES, key="nav_page")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:12px;color:#94a3b8;line-height:1.7;">'
    'Data: openFDA API<br>Model: RF + Gradient Boosting<br>Built with CrewAI + scikit-learn</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: DEVICE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Device Classification":
    import math

    def _last_sunday():
        """Return the most recent Sunday (date) at or before today."""
        from datetime import date, timedelta
        today = date.today()
        return today - timedelta(days=(today.weekday() + 1) % 7)

    def _refresh_foiclass():
        import zipfile, io, requests as _req
        resp = _req.get("https://www.accessdata.fda.gov/premarket/ftparea/foiclass.zip", timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(('.txt', '.lis')))
            raw = z.read(name).decode("latin-1")
        lines = raw.splitlines()
        cols = [c.strip().lower() for c in lines[0].split("|")]
        rows = [dict(zip(cols, l.split("|"))) for l in lines[1:] if l.strip()]
        df = pd.DataFrame(rows)
        p = ARTIFACTS / "foiclass.csv"
        df.to_csv(p, index=False)
        # record download date
        (ARTIFACTS / "foiclass_updated.txt").write_text(str(_last_sunday()))
        st.cache_data.clear()
        return df

    @st.cache_data(show_spinner=False)
    def load_foiclass():
        from datetime import date
        p = ARTIFACTS / "foiclass.csv"
        stamp_p = ARTIFACTS / "foiclass_updated.txt"
        last_sunday = _last_sunday()
        # check if we need a refresh
        needs_refresh = (
            not p.exists() or
            not stamp_p.exists() or
            date.fromisoformat(stamp_p.read_text().strip()) < last_sunday
        )
        if needs_refresh:
            return _refresh_foiclass()
        df = pd.read_csv(p, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="margin:0;font-size:24px;font-weight:700;color:#0b1c30;">FDA Device Classification</h2>
        <p style="margin:6px 0 0;color:#44474e;font-size:14px;">
            Search the full FDA 21 CFR product code classification database (7,070 device types)
            to identify device class, regulation number, and submission pathway requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading FDA classification database…"):
        foiclass_df = load_foiclass()

    # ── DB status bar ─────────────────────────────────────────────────────────
    stamp_p = ARTIFACTS / "foiclass_updated.txt"
    stamp = stamp_p.read_text().strip() if stamp_p.exists() else "unknown"
    info_col, btn_col2 = st.columns([6, 1])
    with info_col:
        st.markdown(
            f'<div style="font-size:12px;color:#64748b;padding:4px 0 12px;">'
            f'Database: <b>{len(foiclass_df):,} devices</b> &nbsp;|&nbsp; '
            f'Last synced: <b>{stamp}</b> &nbsp;|&nbsp; FDA updates every Sunday</div>',
            unsafe_allow_html=True
        )
    with btn_col2:
        if st.button("↻ Refresh", help="Re-download latest FDA classification file"):
            with st.spinner("Downloading latest foiclass.zip from FDA…"):
                try:
                    _refresh_foiclass()
                    st.success("Updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")

    # ── Search bar ────────────────────────────────────────────────────────────
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search device name, product code, or keyword",
            placeholder="e.g. tag, label, pacemaker, QMX, glucose…",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("Search", width='stretch', type="primary")

    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        filter_class = st.selectbox("Filter by Class", ["All", "Class I", "Class II", "Class III"])
    with adv_col2:
        filter_specialty = st.text_input("Filter by Specialty (code)", placeholder="e.g. CV, SU, MI…")
    with adv_col3:
        per_page = st.selectbox("Results per page", [10, 25, 50], index=0)

    # ── Local search ──────────────────────────────────────────────────────────
    if search_clicked or (query and "dc_results" not in st.session_state):
        df_search = foiclass_df.copy()
        q = query.strip()

        if q:
            # Product code exact match only when user types ALL CAPS 3-char code (e.g. QMX)
            if len(q) == 3 and q.isupper():
                mask = df_search["productcode"].str.upper() == q
            else:
                # Keyword search across device name and definition
                words = [w.strip() for w in q.split() if w.strip()]
                mask = pd.Series([True] * len(df_search), index=df_search.index)
                for w in words:
                    mask &= (
                        df_search["devicename"].str.contains(w, case=False, na=False) |
                        df_search["definition"].str.contains(w, case=False, na=False)
                    )
            df_search = df_search[mask]

        if filter_class != "All":
            cls_num = {"Class I": "1", "Class II": "2", "Class III": "3"}[filter_class]
            df_search = df_search[df_search["deviceclass"] == cls_num]

        if filter_specialty.strip():
            df_search = df_search[
                df_search["medicalspecialty"].str.upper() == filter_specialty.strip().upper()
            ]

        st.session_state["dc_results"] = df_search.to_dict("records")
        st.session_state["dc_page"] = 0

    # ── Results ───────────────────────────────────────────────────────────────
    results = st.session_state.get("dc_results", [])

    if results is not None and len(results) == 0 and "dc_results" in st.session_state:
        st.info("No matching devices found. Try broader search terms.")
    elif results:
        pg = st.session_state.get("dc_page", 0)
        start = pg * per_page
        page_results = results[start: start + per_page]

        st.markdown(
            f'<div style="font-size:13px;color:#44474e;margin-bottom:12px;">'
            f'Showing <b>{start+1}–{min(start+per_page, len(results))}</b> of <b>{len(results)}</b> results'
            f'</div>',
            unsafe_allow_html=True
        )

        CLASS_COLORS = {
            "1": ("#d1fae5", "#065f46", "Class I",   "Low Risk"),
            "2": ("#fef3c7", "#92400e", "Class II",  "Moderate Risk"),
            "3": ("#fee2e2", "#991b1b", "Class III", "High Risk"),
        }
        PATHWAY_MAP = {
            "1": "510(k) / Exempt",
            "2": "510(k)",
            "3": "PMA",
        }

        import html as _html
        for _ri, rec in enumerate(page_results):
            # Normalize all fields — guard against NaN (float) coming from CSV
            def _s(v, fallback="—"):
                if v is None: return fallback
                try:
                    import math
                    if isinstance(v, float) and math.isnan(v): return fallback
                except Exception: pass
                return str(v).strip() or fallback

            cls    = _s(rec.get("deviceclass"), "?")
            # submission_type_id may be stored as float (e.g. 4.0) — normalise to int-string
            _raw_sub = rec.get("submission_type_id")
            try:
                import math
                sub_id = "" if (_raw_sub is None or (isinstance(_raw_sub, float) and math.isnan(_raw_sub))) \
                         else str(int(float(_raw_sub)))
            except Exception:
                sub_id = ""

            bg, fg, cls_label, risk = CLASS_COLORS.get(cls, ("#f1f5f9","#334155","Unknown",""))
            # Determine accurate pathway hint from submission_type_id
            if sub_id in ("", "4", "7"):
                pathway_hint = "510(k) Exempt"
            elif sub_id == "2":
                pathway_hint = "PMA"
            elif sub_id == "6":
                pathway_hint = "De Novo"
            elif cls == "1":
                pathway_hint = "510(k) Exempt"
            elif cls == "3":
                pathway_hint = "PMA"
            else:
                pathway_hint = "510(k)"

            prod_code  = _s(rec.get("productcode"))
            dev_name   = _s(rec.get("devicename")).title()
            specialty  = _s(rec.get("medicalspecialty"))
            reg_num    = _s(rec.get("regulationnumber"))
            definition = _s(rec.get("definition"), fallback="")
            implant    = _s(rec.get("implant_flag"), "N")
            life_sus   = _s(rec.get("life_sustain_support_flag"), "N")
            gmp_ex     = _s(rec.get("gmpexemptflag"), "N")
            tp_flag    = _s(rec.get("thirdpartyflag"), "N")

            flag_html = ""
            if implant == "Y":
                flag_html += '<span style="background:#ede9fe;color:#5b21b6;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;margin-right:4px;">Implant</span>'
            if life_sus == "Y":
                flag_html += '<span style="background:#fee2e2;color:#991b1b;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;margin-right:4px;">Life-Sustaining</span>'
            if gmp_ex == "Y":
                flag_html += '<span style="background:#d1fae5;color:#065f46;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;margin-right:4px;">GMP Exempt</span>'
            if tp_flag == "Y":
                flag_html += '<span style="background:#dbeafe;color:#1e40af;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;margin-right:4px;">Third-Party</span>'

            def_snippet = _html.escape((definition[:200] + "…") if len(definition) > 200 else definition)
            dev_name_safe = _html.escape(dev_name)
            prod_code_safe = _html.escape(prod_code)
            reg_num_safe = _html.escape(reg_num)
            specialty_safe = _html.escape(specialty)

            with st.container():
                st.markdown(f"""
<div style="border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;
            margin-bottom:4px;background:#fff;
            box-shadow:0 1px 4px rgba(0,0,0,.04);">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;">
    <div style="flex:1;min-width:200px;">
      <div style="font-size:15px;font-weight:700;color:#0b1c30;margin-bottom:2px;">{dev_name_safe}</div>
      <div style="font-size:12px;color:#44474e;">
        <b>Product Code:</b> {prod_code_safe} &nbsp;|&nbsp;
        <b>Regulation:</b> {reg_num_safe} &nbsp;|&nbsp;
        <b>Specialty:</b> {specialty_safe}
      </div>
      {f'<div style="font-size:12px;color:#64748b;margin-top:6px;">{def_snippet}</div>' if def_snippet else ''}
      <div style="margin-top:8px;">{flag_html}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;min-width:120px;">
      <span style="background:{bg};color:{fg};font-size:12px;font-weight:700;
                   padding:3px 12px;border-radius:999px;white-space:nowrap;">
        {cls_label} — {risk}
      </span>
      <span style="font-size:11px;color:#64748b;">Typical pathway: <b>{pathway_hint}</b></span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                use_col, _ = st.columns([2, 8])
                with use_col:
                    if st.button(f"Use Class {cls} in Predictor", key=f"use_{_ri}_{prod_code}_{cls}"):
                        st.session_state["prefill_class"] = int(cls) if cls.isdigit() else 2
                        st.session_state["_nav_target"] = "Pathway Predictor"
                        st.rerun()

        # Pagination
        n_pages = math.ceil(len(results) / per_page)
        if n_pages > 1:
            pcols = st.columns([1, 3, 1])
            with pcols[0]:
                if pg > 0 and st.button("← Prev"):
                    st.session_state["dc_page"] = pg - 1; st.rerun()
            with pcols[1]:
                st.markdown(
                    f'<div style="text-align:center;font-size:13px;color:#44474e;padding-top:8px;">Page {pg+1} of {n_pages}</div>',
                    unsafe_allow_html=True
                )
            with pcols[2]:
                if pg < n_pages - 1 and st.button("Next →"):
                    st.session_state["dc_page"] = pg + 1; st.rerun()

    else:
        # Landing state — show quick-access class cards
        st.markdown('<div style="margin-top:32px;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        card_style = "border-radius:8px;padding:20px;text-align:center;border:1px solid"
        with c1:
            st.html(f"""<div style="{card_style} #a7f3d0;background:#ecfdf5;">
              <div style="font-size:28px;font-weight:800;color:#065f46;">Class I</div>
              <div style="font-size:13px;font-weight:600;color:#065f46;margin:4px 0;">Low Risk</div>
              <div style="font-size:12px;color:#047857;">General Controls only.<br>Often 510(k) exempt.</div>
              <div style="font-size:11px;color:#6b7280;margin-top:8px;">Examples: bandages, tongue depressors, elastic bandages</div>
            </div>""")
        with c2:
            st.html(f"""<div style="{card_style} #fcd34d;background:#fffbeb;">
              <div style="font-size:28px;font-weight:800;color:#92400e;">Class II</div>
              <div style="font-size:13px;font-weight:600;color:#92400e;margin:4px 0;">Moderate Risk</div>
              <div style="font-size:12px;color:#b45309;">General + Special Controls.<br>Usually requires 510(k).</div>
              <div style="font-size:11px;color:#6b7280;margin-top:8px;">Examples: infusion pumps, surgical drapes, x-ray systems</div>
            </div>""")
        with c3:
            st.html(f"""<div style="{card_style} #fca5a5;background:#fef2f2;">
              <div style="font-size:28px;font-weight:800;color:#991b1b;">Class III</div>
              <div style="font-size:13px;font-weight:600;color:#991b1b;margin:4px 0;">High Risk</div>
              <div style="font-size:12px;color:#b91c1c;">General + Special Controls<br>+ Premarket Approval (PMA).</div>
              <div style="font-size:11px;color:#6b7280;margin-top:8px;">Examples: pacemakers, cochlear implants, heart valves</div>
            </div>""")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PATHWAY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pathway Predictor":
    # Apply pre-fill from Device Classification page
    if "prefill_class" in st.session_state:
        st.session_state["device_class"] = st.session_state.pop("prefill_class")

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

    # ── Load foiclass for autocomplete & auto-detect ───────────────────────────
    @st.cache_data(show_spinner=False)
    def _load_foi():
        p = ARTIFACTS / "foiclass.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    foi_df = _load_foi()
    _foi_names = sorted(foi_df["devicename"].str.strip().str.title().unique()) if not foi_df.empty else []

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    if not st.session_state.show_results:
        st.markdown('<div class="fda-card">', unsafe_allow_html=True)
        di_col, sd_col = st.columns(2, gap="large")

        with di_col:
            st.markdown(
                '<div style="font-size:11px;font-weight:700;color:#002046;text-transform:uppercase;'
                'letter-spacing:.06em;margin-bottom:16px;">Device Information</div>',
                unsafe_allow_html=True,
            )

            # Device name — free text + autocomplete suggestions from foiclass
            device_name_typed = st.text_input(
                "Device Name",
                placeholder="e.g. pacemaker, label, glucose monitor…",
                key="inp_dn",
            )
            foi_match = None
            # Clear saved suggestion if user clears/changes the text input
            if "inp_dn_last" not in st.session_state:
                st.session_state["inp_dn_last"] = ""
            if device_name_typed.strip() != st.session_state["inp_dn_last"]:
                st.session_state.pop("inp_dn_foi_match", None)
                st.session_state["inp_dn_last"] = device_name_typed.strip()

            if len(device_name_typed.strip()) >= 2 and not foi_df.empty:
                _hits = foi_df[foi_df["devicename"].str.contains(
                    device_name_typed.strip(), case=False, na=False, regex=False)]
                if not _hits.empty:
                    _names = _hits["devicename"].str.strip().str.title().tolist()[:20]
                    _options = ["— select to auto-fill fields —"] + _names
                    _pick = st.selectbox(
                        f"FDA database suggestions ({len(_hits)} match{'es' if len(_hits)!=1 else ''})",
                        _options, key="inp_dn_suggest",
                    )
                    if _pick != "— select to auto-fill fields —":
                        _row = _hits[_hits["devicename"].str.strip().str.title() == _pick]
                        if not _row.empty:
                            st.session_state["inp_dn_foi_match"] = _row.iloc[0].to_dict()

            if "inp_dn_foi_match" in st.session_state:
                foi_match = st.session_state["inp_dn_foi_match"]
            device_name = foi_match["devicename"].title() if foi_match else device_name_typed

            # Advisory Committee — Auto-detect or manual
            advisory_committees = sorted(
                k for k in encoders.get("advisory_committee", {}).keys() if k != "UNKNOWN"
            )
            if not advisory_committees:
                advisory_committees = [
                    "AN", "CV", "CH", "DE", "EN", "GU", "HO", "IM",
                    "MI", "NE", "OB", "OP", "OR", "PA", "PM", "RA", "SU", "TX",
                ]
            _ac_options = ["Not Specified"] + advisory_committees
            _ac_hint = ""
            if foi_match is not None and foi_match.get("medicalspecialty", "") in advisory_committees:
                _ac_hint = foi_match["medicalspecialty"]
                _ac_idx = _ac_options.index(_ac_hint)
            else:
                _ac_idx = 0
            advisory_committee_sel = st.selectbox(
                "Advisory Committee / Medical Specialty",
                _ac_options,
                index=_ac_idx,
                key="inp_ac",
                help="Select 'Not Specified' to let the model infer from device class and name",
            )

            # Device Class — Auto-detect or manual
            _dc_options = ["Not Specified", "Class I — Low Risk", "Class II — Moderate Risk", "Class III — High Risk"]
            _dc_idx = 0
            if foi_match is not None and foi_match.get("deviceclass", "") in ("1", "2", "3"):
                _dc_idx = int(foi_match["deviceclass"])  # 1→idx1, 2→idx2, 3→idx3
            elif st.session_state.get("device_class") in (1, 2, 3):
                _dc_idx = st.session_state["device_class"]
            device_class_sel = st.selectbox(
                "Device Class",
                _dc_options,
                index=_dc_idx,
                key="inp_dc",
                help="Select 'Not Specified' to use dataset median",
            )

        with sd_col:
            st.markdown(
                '<div style="font-size:11px;font-weight:700;color:#002046;text-transform:uppercase;'
                'letter-spacing:.06em;margin-bottom:16px;">Submission Details</div>',
                unsafe_allow_html=True,
            )
            country_label = st.selectbox("Country of Origin", list(COUNTRY_MAP.keys()), index=0, key="inp_cn")
            country_code = COUNTRY_MAP[country_label]
            third_party = st.toggle(
                "Third-party review",
                value=False,
                key="inp_tp",
                help="Accelerated review by an accredited organization",
            )
            # Clearance Type — Auto-detect or manual
            _ct_options = ["Not Specified", "Traditional", "Abbreviated", "Special"]
            _ct_idx = 0
            if foi_match is not None:
                _sub = foi_match.get("submission_type_id", "")
                if _sub == "1":
                    _ct_idx = 1  # Traditional 510(k)
                elif _sub == "2":
                    _ct_idx = 1  # PMA → default Traditional
            clearance_type_sel = st.selectbox(
                "Clearance Type",
                _ct_options,
                index=_ct_idx,
                key="inp_ct",
                help="Applicable for 510(k) pathway. 'Not Specified' uses dataset most common.",
            )
            _pc_default = foi_match.get("productcode", "") if foi_match else ""
            st.text_input(
                "Product Code (optional)",
                value=_pc_default,
                placeholder="e.g. DQK — auto-filled from FDA suggestion",
                key="inp_pc",
                help="3-letter FDA product code. Used for predicate lookup and regulatory evidence.",
            )

        # ── Resolve auto-detect values before prediction ──────────────────────
        # Advisory committee
        if advisory_committee_sel == "Not Specified":
            # use mode from training data
            advisory_committee = (
                features_df["advisory_committee_encoded"].mode()[0]
                if features_df is not None and "advisory_committee_encoded" in features_df.columns
                else 0
            )
            advisory_committee_str = "Auto"
        else:
            advisory_committee_str = advisory_committee_sel
            advisory_committee = advisory_committee_sel

        # Device class
        _dc_map = {"Class I — Low Risk": 1, "Class II — Moderate Risk": 2, "Class III — High Risk": 3}
        if device_class_sel == "Not Specified":
            device_class = int(
                features_df["device_class"].median()
                if features_df is not None and "device_class" in features_df.columns
                else 2
            )
            st.session_state["device_class"] = device_class
        else:
            device_class = _dc_map[device_class_sel]
            st.session_state["device_class"] = device_class

        # Clearance type
        clearance_type = "Traditional" if clearance_type_sel == "Not Specified" else clearance_type_sel
        has_predicate = clearance_type == "Traditional"
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Pathway", type="primary", width='stretch', key="analyze_btn"):

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

            # Resolve encoded values; fall back to training median when "Auto"
            if advisory_committee_str == "Auto":
                ac_enc = _fmed("advisory_committee_encoded")
                ms_enc = _fmed("medical_specialty_encoded")
            else:
                ac_enc = encoders.get("advisory_committee", {}).get(advisory_committee_str, 0)
                ms_enc = encoders.get("medical_specialty", {}).get(advisory_committee_str, 0)
            ct_enc = encoders.get("clearance_type", {}).get(clearance_type, 0)
            is_us = (1 if country_code == "US" else 0) if country_code is not None else int(_fmed("is_us"))
            input_dict = {
                "device_class": device_class, "device_class_unknown": 0 if device_class_sel != "Not Specified" else 1,
                "advisory_committee_freq": _fmed("advisory_committee_freq"),
                "advisory_committee_encoded": ac_enc,
                "medical_specialty_freq": _fmed("medical_specialty_freq"),
                "medical_specialty_encoded": ms_enc,
                "is_us": is_us,
                "country_freq": _fmed("country_freq"),
                "decision_year": 2025, "decision_month": 6,
                "month_sin": np.sin(2 * np.pi * 6 / 12),
                "month_cos": np.cos(2 * np.pi * 6 / 12),
                "review_days": _fmed("review_days"), "has_review_days": 1,
                "clearance_type_encoded": ct_enc, "third_party": 1 if third_party else 0,
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
            st.session_state.last_advisory = advisory_committee_str
            st.session_state.last_feature_cols = feature_cols

            # Gather regulatory evidence (graceful — never blocks prediction)
            try:
                from src.tools.regulatory_evidence import build_regulatory_evidence
                _inp_pc = st.session_state.get("inp_pc", "").strip()
                with st.spinner("Gathering regulatory evidence..."):
                    evidence = build_regulatory_evidence(
                        device_name=device_name,
                        device_description="",
                        device_class=device_class,
                        advisory_committee=advisory_committee_str if advisory_committee_str != "Auto" else None,
                        product_code=_inp_pc or None,
                        pathway=pathway,
                        has_predicate=has_predicate,
                        data_path=str(ARTIFACTS / "clean_data.csv"),
                    )
                st.session_state.last_evidence = evidence
            except Exception as _ev_err:
                st.session_state.last_evidence = None
                logger.warning(f"Evidence gathering skipped: {_ev_err}")

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

        # ── RESULTS GRID (matches Stitch design) ───────────────────────────────
        PATHWAY_DISPLAY = {"510k": "510(k)", "PMA": "PMA", "De_Novo": "De Novo"}
        pathway_display = PATHWAY_DISPLAY.get(pathway, pathway)
        confidence_level = "High" if confidence >= 75 else "Moderate" if confidence >= 50 else "Low"

        hero_col, charts_col = st.columns([1, 2], gap="large")

        with hero_col:
            # Hero pathway card — dark navy
            st.markdown(
                f'<div style="background:#002046;border-radius:12px;padding:32px 24px;'
                f'color:white;margin-bottom:16px;">'
                f'<div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                f'color:rgba(255,255,255,.55);margin-bottom:16px;">Recommended Pathway</div>'
                f'<div style="font-size:56px;font-weight:700;line-height:1;margin-bottom:24px;">'
                f'{pathway_display}</div>'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="#4ede9b"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'
                f'<span style="font-size:14px;font-weight:600;">{confidence_level} confidence prediction</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            # Confidence Score metric
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;'
                f'display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">'
                f'<div><div style="font-size:12px;font-weight:600;color:#64748b;margin-bottom:4px;">'
                f'Confidence Score</div>'
                f'<div style="font-size:24px;font-weight:700;color:#002046;">{confidence:.1f}%</div></div>'
                f'<div style="width:40px;height:40px;border-radius:50%;background:#dcfce7;'
                f'display:flex;align-items:center;justify-content:center;">'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#16a34a"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            # Assigned Class metric
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;'
                f'display:flex;align-items:center;justify-content:space-between;">'
                f'<div><div style="font-size:12px;font-weight:600;color:#64748b;margin-bottom:4px;">'
                f'Assigned Class</div>'
                f'<div style="font-size:24px;font-weight:700;color:#002046;">Class {device_class}</div></div>'
                f'<div style="width:40px;height:40px;border-radius:50%;background:#dbeafe;'
                f'display:flex;align-items:center;justify-content:center;">'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 3c1.93 0 3.5 1.57 3.5 3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm7 13H5v-.23c0-.62.28-1.2.76-1.58C7.47 15.82 9.64 15 12 15s4.53.82 6.24 2.19c.48.38.76.97.76 1.58V19z"/></svg>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with charts_col:
            st.markdown('<div class="fda-card">', unsafe_allow_html=True)

            # Confidence by Pathway
            st.markdown(
                '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:16px;">'
                'Confidence by Pathway</div>',
                unsafe_allow_html=True,
            )
            pathway_order = sorted(encoders.get("pathway", {}).items(), key=lambda x: x[1])
            for cn, ci in pathway_order:
                if ci < len(proba):
                    pct = proba[ci] * 100
                    bar_color = "#002046" if cn == pathway else "#cbd5e1"
                    weight = "font-weight:700;" if cn == pathway else ""
                    pw_full = PATHWAY_INFO.get(cn, {}).get("full", cn.replace("_", " "))
                    st.markdown(
                        f'<div class="pw-bar-row">'
                        f'<div class="pw-bar-hd"><span style="{weight}">{pw_full}</span>'
                        f'<span style="{weight}">{pct:.0f}%</span></div>'
                        f'<div class="pw-bar-t"><div class="pw-bar-f" '
                        f'style="width:{min(pct,100):.1f}%;background:{bar_color};"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<hr style="border:none;border-top:1px solid #f1f5f9;margin:24px 0;">', unsafe_allow_html=True)

            # Key Factors in This Prediction
            st.markdown(
                '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:16px;">'
                'Key Factors in This Prediction</div>',
                unsafe_allow_html=True,
            )
            if hasattr(model, "feature_importances_") and feature_cols:
                imp_series = (
                    pd.Series(model.feature_importances_, index=feature_cols)
                    .sort_values(ascending=False)
                    .head(6)
                )
                max_imp = imp_series.max()
                rows_html = ""
                for k, v in imp_series.items():
                    pct = int(v / max_imp * 100) if max_imp > 0 else 0
                    label = FEAT_LABELS.get(k, k.replace("_", " ").title())
                    rows_html += (
                        f'<div style="margin-bottom:12px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:13px;margin-bottom:4px;">'
                        f'<span style="font-weight:600;color:#0b1c30;">{label}</span>'
                        f'<span style="color:#64748b;">{pct}%</span></div>'
                        f'<div style="background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{pct}%;height:100%;background:#002046;border-radius:4px;"></div>'
                        f'</div></div>'
                    )
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="font-size:13px;color:#64748b;">Feature importance not available.</p>',
                    unsafe_allow_html=True,
                )

            # Pathway info box
            st.markdown(
                f'<div style="background:#eff6ff;border-left:4px solid #002046;padding:16px 20px;'
                f'border-radius:0 8px 8px 0;margin-top:16px;">'
                f'<div style="font-size:14px;font-weight:600;color:#002046;margin-bottom:6px;">{info["full"]}</div>'
                f'<p style="font-size:13px;color:#475569;line-height:1.6;margin:0;">'
                f'{info["desc"]} Timeline: <strong>{info["review"]}</strong>. '
                f'Clinical trials: <strong>{info["trials"]}</strong>.</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

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
                st.dataframe(pd.DataFrame(sim["examples"]), width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── REGULATORY EVIDENCE SECTIONS ──────────────────────────────────────
        evidence = st.session_state.get("last_evidence")
        if evidence:
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Section 1: Evidence Strength ───────────────────────────────────
            strength = evidence.get("evidence_strength", "LOW")
            facts    = evidence.get("supporting_facts", [])
            strength_colors = {
                "HIGH":   ("#d1fae5", "#065f46", "#10b981"),
                "MEDIUM": ("#fef3c7", "#92400e", "#f59e0b"),
                "LOW":    ("#fee2e2", "#991b1b", "#ef4444"),
            }
            bg, fg, dot_c = strength_colors.get(strength, strength_colors["LOW"])
            facts_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:7px;">'
                f'<span style="color:{dot_c};font-size:16px;line-height:1.3;">&#10003;</span>'
                f'<span style="font-size:13px;color:#374151;">{f}</span></div>'
                for f in facts
            )
            st.markdown(
                f'<div class="fda-card">'
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:.05em;">Evidence Strength</div>'
                f'<span style="background:{bg};color:{fg};font-size:13px;font-weight:700;'
                f'padding:3px 14px;border-radius:999px;">{strength}</span></div>'
                f'{facts_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Section 2: Top Predicate Candidates ───────────────────────────
            predicates = evidence.get("predicate_candidates", [])
            if predicates:
                st.markdown(
                    '<div class="fda-card">'
                    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;">'
                    'Top Predicate Candidates</div>',
                    unsafe_allow_html=True,
                )
                pred_df = pd.DataFrame([
                    {
                        "Submission ID": p.get("submission_id", ""),
                        "Device Name":   p.get("device_name", ""),
                        "Applicant":     p.get("applicant", ""),
                        "Decision Date": p.get("decision_date", ""),
                        "Decision":      p.get("decision_code", ""),
                        "Similarity":    f'{p.get("similarity_score", 0):.2f}',
                        "Source":        p.get("source", ""),
                    }
                    for p in predicates
                ])
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="fda-card">'
                    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">'
                    'Top Predicate Candidates</div>'
                    '<p style="font-size:13px;color:#64748b;margin:0;">No predicate candidates found for this '
                    'device configuration. If you have a product code, enter it above to improve results.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── Section 3: Semantic Similar Devices (if available) ─────────────
            sem = evidence.get("similar_devices", {})
            sem_top = sem.get("top_similar_devices", [])
            sem_method = sem.get("similarity_method", "")
            if sem_top and sem_method:
                with st.expander(
                    f"Semantically Similar Devices — {sem_method.replace('_', ' ').title()} "
                    f"({sem.get('total_matches', 0):,} total)"
                ):
                    sem_df = pd.DataFrame([
                        {
                            "Submission ID":    d.get("submission_id", ""),
                            "Device Name":      d.get("device_name", ""),
                            "Pathway":          d.get("pathway", ""),
                            "Decision":         d.get("decision_code", ""),
                            "Similarity Score": f'{d.get("similarity_score", 0):.3f}',
                        }
                        for d in sem_top
                    ])
                    st.dataframe(sem_df, use_container_width=True, hide_index=True)

            # ── Section 4: Regulatory References ──────────────────────────────
            reg_refs = evidence.get("regulatory_references", [])
            if reg_refs:
                st.markdown(
                    '<div class="fda-card">'
                    '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;">'
                    'Regulatory References</div>',
                    unsafe_allow_html=True,
                )
                for ref in reg_refs:
                    label = ref.get("label", ref.get("title", ref.get("part", "")))
                    url   = ref.get("url", "")
                    if label and url:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                            f'<span style="color:#002046;font-size:16px;">&#8594;</span>'
                            f'<a href="{url}" target="_blank" style="font-size:13px;color:#378ADD;'
                            f'text-decoration:none;font-weight:500;">{label}</a></div>',
                            unsafe_allow_html=True,
                        )
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
                width='stretch',
            )
        with ab2:
            if st.button("Start New Analysis", width='stretch', key="new_analysis_btn"):
                st.session_state.show_results = False
                st.rerun()
        with ab3:
            checklist_label = f"View {pathway.replace('_', ' ')} Checklist →"
            if st.button(checklist_label, type="primary", width='stretch', key="view_checklist_btn"):
                st.session_state["_nav_target"] = "Preparation Checklist"
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
    predicted = st.session_state.get("last_pathway")

    st.markdown(
        '<h1 style="font-size:26px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'Submission Preparation Checklists</h1>',
        unsafe_allow_html=True,
    )
    if predicted:
        PW_DISPLAY = {"510k": "510(k)", "PMA": "PMA", "De_Novo": "De Novo"}
        st.markdown(
            f'<p style="color:#64748b;margin-bottom:8px;font-size:15px;">'
            f'Your predicted pathway is <strong style="color:#002046;">{PW_DISPLAY.get(predicted, predicted)}</strong>. '
            f'All three pathways are shown below for comparison.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#64748b;margin-bottom:8px;font-size:15px;">'
            'Browse all three FDA regulatory pathways — checklists, required forms, and key resources.</p>',
            unsafe_allow_html=True,
        )

    # ── Required forms per pathway ─────────────────────────────────────────────
    FORMS = {
        "510k": [
            ("FDA Form 3881 — Indications for Use",
             "https://www.fda.gov/media/72421/download",
             "Required for every 510(k). Describes the device's intended use and indications."),
            ("FDA Form 3514 — 510(k) Summary",
             "https://www.fda.gov/media/71706/download",
             "Public summary of the substantial equivalence determination. Alternative: 510(k) Statement."),
            ("CDRH eCopy Cover Sheet",
             "https://www.fda.gov/medical-devices/how-study-and-market-your-device/ecopy-program-medical-device-submissions",
             "Required for all 510(k) submissions. eCopy must be provided on CD/DVD or USB."),
            ("FDA User Fee Cover Sheet (Form 3601)",
             "https://www.fda.gov/media/72418/download",
             "Required for paying the 510(k) user fee. Submit before or with your application."),
            ("Substantial Equivalence (SE) Comparison Table",
             "https://www.fda.gov/medical-devices/premarket-notification-510k/how-prepare-510k",
             "Side-by-side comparison of your device vs. predicate device (not a form — part of the 510(k) package)."),
        ],
        "PMA": [
            ("FDA Form 3514A — PMA Cover Sheet",
             "https://www.fda.gov/media/72421/download",
             "Cover sheet for PMA applications. Identifies the applicant, device, and PMA type."),
            ("FDA Form 3674 — IDE Application Cover Sheet",
             "https://www.fda.gov/media/71683/download",
             "Required to begin significant-risk clinical studies under an Investigational Device Exemption (IDE)."),
            ("FDA Form 3454 — Certification: Financial Interests",
             "https://www.fda.gov/media/71681/download",
             "Financial disclosure certification required for clinical investigators in PMA studies."),
            ("FDA Form 3455 — Financial Disclosure",
             "https://www.fda.gov/media/71682/download",
             "Discloses financial interests of clinical investigators per 21 CFR Part 54."),
            ("CDRH eCopy Cover Sheet",
             "https://www.fda.gov/medical-devices/how-study-and-market-your-device/ecopy-program-medical-device-submissions",
             "Required for PMA submissions in eCopy format."),
            ("PMA User Fee Cover Sheet (Form 3602A)",
             "https://www.fda.gov/media/72419/download",
             "Required to pay the PMA application user fee prior to submission."),
        ],
        "De_Novo": [
            ("FDA Form 3881 — Indications for Use",
             "https://www.fda.gov/media/72421/download",
             "Required as part of the De Novo request package."),
            ("De Novo Request Cover Sheet",
             "https://www.fda.gov/media/109484/download",
             "Identifies the applicant, device name, product code (if known), and classification proposal."),
            ("CDRH eCopy Cover Sheet",
             "https://www.fda.gov/medical-devices/how-study-and-market-your-device/ecopy-program-medical-device-submissions",
             "Required for De Novo submissions in eCopy format."),
            ("De Novo User Fee Cover Sheet",
             "https://www.fda.gov/medical-devices/how-study-and-market-your-device/medical-device-user-fee-amendments-mdufa",
             "Required to pay the De Novo classification request user fee."),
            ("Proposed Special Controls Document",
             "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request",
             "Applicant-proposed special controls that provide reasonable assurance of safety and effectiveness for a Class II device."),
        ],
    }

    RESOURCES = {
        "510k": [
            ("510(k) Guidance Overview", "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k"),
            ("Search 510(k) Database", "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"),
            ("Special 510(k) Program", "https://www.fda.gov/medical-devices/premarket-notification-510k/special-510k-program"),
            ("Abbreviated 510(k) Program", "https://www.fda.gov/medical-devices/premarket-notification-510k/abbreviated-510k-program"),
            ("eSubmitter Tool", "https://www.fda.gov/industry/fda-esubmitter"),
            ("CDRH Learn: 510(k) Basics", "https://www.fda.gov/training-and-continuing-education/cdrh-learn"),
        ],
        "PMA": [
            ("PMA Guidance Overview", "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-approval-pma"),
            ("Search PMA Database", "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm"),
            ("IDE Guidance", "https://www.fda.gov/medical-devices/how-study-and-market-your-device/investigational-device-exemption-ide"),
            ("Q-Submission Program", "https://www.fda.gov/medical-devices/how-study-and-market-your-device/q-submission-program"),
            ("Modular PMA Guidance", "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/guidance-modular-pma-review-procedures"),
            ("Breakthrough Device Program", "https://www.fda.gov/patients/fast-track-breakthrough-therapy-accelerated-approval-priority-review/breakthrough-devices-program"),
        ],
        "De_Novo": [
            ("De Novo Guidance", "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request"),
            ("Search De Novo Database", "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm"),
            ("Q-Submission Program", "https://www.fda.gov/medical-devices/how-study-and-market-your-device/q-submission-program"),
            ("De Novo Acceptance Checklist", "https://www.fda.gov/media/108280/download"),
            ("Special Controls Guidance", "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"),
        ],
    }

    TIMELINES = {
        "510k": [
            ("Months 1–3", "Predicate search & bench testing"),
            ("Months 3–6", "Technical documentation & labeling"),
            ("Month 6", "Submission to FDA + fee payment"),
            ("Months 7–9", "FDA acceptance review (30 days)"),
            ("Months 9–12", "Substantive review + AI responses"),
        ],
        "PMA": [
            ("Months 1–6", "Q-Sub meeting & IDE preparation"),
            ("Months 6–30", "Clinical trial conduct under IDE"),
            ("Months 30–36", "PMA compilation & submission"),
            ("Months 36–42", "FDA filing & substantive review"),
            ("Months 42–54", "Advisory panel (if required) & approval"),
        ],
        "De_Novo": [
            ("Months 1–3", "Confirm no predicate & novelty analysis"),
            ("Months 3–6", "Special controls development & testing"),
            ("Months 6–9", "Package preparation & submission"),
            ("Months 9–21", "FDA review (150-day goal) & grant order"),
        ],
    }

    # ── Sub-checklist items per step ──────────────────────────────────────────
    STEP_ITEMS = {
        "510k": [
            [   # Step 1 – Determine Device Classification
                "Verify device meets FDA definition of a medical device (21 CFR 201(h))",
                "Confirm device is not a drug, biologic, or combination product",
                "Identify correct product code using FDA Device Classification database",
                "Confirm Class II classification (or Class I requiring 510(k))",
                "Document classification rationale in regulatory file",
            ],
            [   # Step 2 – Identify Predicate Device
                "Search FDA 510(k) database (PMNS) for cleared devices with same intended use",
                "Verify predicate's technological characteristics match or differences don't raise new questions",
                "Consider split-predicate approach if needed (intended use + tech characteristics)",
                "Download and save predicate 510(k) decision summary",
                "Document predicate selection rationale in technical file",
            ],
            [   # Step 3 – Prepare 510(k) Summary or Statement
                "Choose between 510(k) Summary (public disclosure) or 510(k) Statement",
                "Draft intended use comparison (your device vs. predicate)",
                "Draft technological characteristics comparison table",
                "Write substantial equivalence conclusion with supporting data references",
                "Have regulatory counsel review the SE determination",
            ],
            [   # Step 4 – Compile Technical Documentation
                "Device description with photographs and labeled diagrams",
                "Draft labeling (instructions for use, labels) per 21 CFR Part 801",
                "Biocompatibility assessment (ISO 10993-1 risk-based evaluation)",
                "Software documentation (IEC 62304 lifecycle, risk analysis) — if applicable",
                "Sterilization validation (ISO 11135 / ISO 11137) — if sterile device",
                "Shelf-life / packaging validation (ASTM F2475 / ISO 11607)",
                "Risk management file (ISO 14971)",
                "Performance testing data summary",
            ],
            [   # Step 5 – Complete Performance Testing
                "Bench testing per applicable FDA-recognized consensus standards",
                "Electrical safety & EMC testing (IEC 60601-1 / IEC 60601-1-2)",
                "Biocompatibility testing per selected ISO 10993 endpoints",
                "Sterilization validation (if sterile) with dose audits",
                "Accelerated aging / real-time aging studies for shelf life",
                "Usability / human factors evaluation (FDA HFE guidance) — if required",
                "Capture test reports with accredited laboratory signatures",
                "Document all testing in Device Master Record (DMR) per 21 CFR Part 820",
            ],
            [   # Step 6 – Submit and Monitor Review
                "Pay 510(k) user fee (Form 3601) before submission",
                "Complete FDA Form 3881 — Indications for Use",
                "Complete 510(k) Summary or Statement (Form 3514)",
                "Prepare eCopy on CD/DVD or USB drive with required folder structure",
                "Submit via FDA CDRH eCopy program or electronic submission gateway",
                "Monitor TPLC for FDA Acceptance Review (target: 15 days)",
                "Track Substantive Review status (FDA goal: 90 days from acceptance)",
                "Respond to Additional Information (AI) requests promptly and completely",
            ],
        ],
        "PMA": [
            [   # Step 1 – Determine Device Classification
                "Confirm device is Class III under 21 CFR Part 860",
                "Verify no reclassification petition or De Novo option is applicable",
                "Identify applicable PMA regulation number (21 CFR Part 814)",
                "Consider Breakthrough Device Designation if device addresses unmet need",
                "Document classification rationale",
            ],
            [   # Step 2 – Request Q-Sub Meeting
                "Identify key open questions for FDA (clinical design, endpoints, acceptability)",
                "Prepare Q-Submission request letter with specific questions",
                "Submit Q-Sub package to CDRH (allow ≥90 days for meeting)",
                "Attend Pre-Submission (B-Meeting) with FDA",
                "Receive and review FDA written response / meeting minutes",
                "Update development plan based on FDA feedback",
            ],
            [   # Step 3 – Design and Conduct Clinical Trials
                "Draft clinical study protocol with FDA-aligned endpoints",
                "Develop statistical analysis plan (SAP) with adequate power",
                "Submit IDE application (Form 3674) for significant-risk studies",
                "Obtain IDE approval from FDA and IRB approval from all sites",
                "Establish informed consent process and data monitoring committee",
                "Conduct clinical trial per approved protocol and GCP (21 CFR Part 812)",
                "Collect and lock data; conduct statistical analysis per SAP",
                "Prepare Clinical Study Report (CSR)",
            ],
            [   # Step 4 – Compile PMA Application
                "Prepare Table of Contents and cover letter",
                "Write Summary of Safety and Effectiveness Data (SSED)",
                "Include all non-clinical (bench / animal) study reports",
                "Include full Clinical Study Report and individual patient data",
                "Compile manufacturing information per 21 CFR 814.20(b)(4)",
                "Prepare draft labeling (IFU, device label, package insert)",
                "Complete financial disclosure statements (Forms 3454 / 3455)",
                "Prepare proposed post-approval study protocol (if conditioning anticipated)",
            ],
            [   # Step 5 – Submit PMA and Respond
                "Pay PMA user fee (Form 3602A) prior to submission",
                "Submit full PMA via FDA eSubmitter or eCopy",
                "Respond to FDA Filing Review questions within 45-day window",
                "Prepare presentation materials if advisory panel is scheduled",
                "Respond to Major Deficiency Letter (MDL) within 180 days",
                "Respond to Not Approvable (NA) or Approvable (A) letters with required changes",
                "Track 180-day PMA review clock and panel dates",
            ],
            [   # Step 6 – Post-Approval Studies and Reporting
                "Implement Medical Device Reporting (MDR) system (21 CFR Part 803)",
                "File PMA Annual Report each year from approval date",
                "Conduct post-approval studies as required by approval order conditions",
                "Submit PMA Supplements for device / labeling / manufacturing changes",
                "Maintain Establishment Registration and Device Listing (annually)",
                "Implement Quality System Regulation (21 CFR Part 820 / ISO 13485)",
            ],
        ],
        "De_Novo": [
            [   # Step 1 – Determine Device Classification
                "Confirm device is novel — not substantially equivalent to any predicate",
                "Verify automatic Class III designation applies (21 CFR 513(f)(1))",
                "Confirm De Novo is preferred route vs. full PMA",
                "Identify whether device is Class I or Class II candidate after De Novo",
                "Document novelty and classification analysis",
            ],
            [   # Step 2 – Confirm No Predicate
                "Thoroughly search FDA 510(k) database for similar intended use devices",
                "Search FDA De Novo database for similar novel devices",
                "Search FDA PMA database for related Class III devices",
                "Review international regulatory databases (CE, TGA, Health Canada)",
                "Review scientific literature for similar devices",
                "Document all search results and no-predicate conclusion",
            ],
            [   # Step 3 – Develop Proposed Classification & Special Controls
                "Define proposed device type name (for the new classification regulation)",
                "Propose Class I or Class II classification with rationale",
                "Identify all risks of the device using risk management (ISO 14971)",
                "Develop draft special controls that mitigate each identified risk",
                "Conduct performance testing to support proposed special controls",
                "Draft performance criteria and testing guidance for future 510(k) submitters",
            ],
            [   # Step 4 – Prepare De Novo Request Package
                "Prepare De Novo request cover sheet (CDRH template)",
                "Complete FDA Form 3881 — Indications for Use",
                "Write device description with diagrams and specifications",
                "Include all performance test reports",
                "Include risk/benefit analysis and special controls rationale",
                "Prepare draft device labeling (IFU, labels)",
                "Compile full request package per FDA De Novo guidance checklist",
            ],
            [   # Step 5 – Submit and Respond
                "Pay De Novo user fee before submission",
                "Submit via FDA eSubmitter or eCopy program",
                "Monitor FDA Acceptance Review (15-day goal)",
                "Respond to FDA Interactive Review questions promptly",
                "Prepare for advisory panel meeting if FDA convenes one",
                "Track 150-day FDA review goal from date of filing",
            ],
            [   # Step 6 – Leverage Grant Order
                "Receive De Novo Grant Order from FDA",
                "Register new device type / product code in FDA CDRH database",
                "Ensure device labeling aligns with grant order indications",
                "Implement any post-market surveillance required by grant order",
                "Monitor future 510(k) submissions citing your device as predicate",
                "File annual reports if required by grant order conditions",
            ],
        ],
    }

    def _render_stepper_expanders(pw_key, color):
        # Inject arrow CSS inline (most reliable injection point)
        st.html("""<style>
[data-testid="stExpander"] details > summary svg,
[data-testid="stExpander"] details > summary > svg,
[data-testid="stExpander"] details > summary span svg { display:none !important; }
[data-testid="stExpander"] details > summary::-webkit-details-marker { display:none; }
[data-testid="stExpander"] details > summary { list-style:none; position:relative; padding-right:28px !important; }
[data-testid="stExpander"] details > summary::after {
    content:''; position:absolute; right:10px; top:50%;
    width:7px; height:7px;
    border-right:1.5px solid #94a3b8; border-bottom:1.5px solid #94a3b8;
    transform:translateY(-70%) rotate(-45deg); transition:transform .18s ease;
    pointer-events:none;
}
[data-testid="stExpander"] details[open] > summary::after {
    transform:translateY(-30%) rotate(45deg);
}
</style>""")
        steps = CHECKLISTS[pw_key]
        items_list = STEP_ITEMS.get(pw_key, [[] for _ in steps])
        for i, step in enumerate(steps):
            is_done = step["done"]
            badge = "Completed" if is_done else f"Step {i + 1} of {len(steps)}"
            badge_color = "#065f46" if is_done else "#002046"
            badge_bg = "#d1fae5" if is_done else "#dbeafe"
            label = f"{'[done]  ' if is_done else f'{i+1}.  '}{step['title']}"
            with st.expander(label, expanded=(i == 0)):
                st.markdown(
                    f'<span style="background:{badge_bg};color:{badge_color};font-size:11px;'
                    f'font-weight:700;padding:2px 10px;border-radius:999px;">{badge}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="color:#374151;font-size:14px;margin:10px 0 6px;">{step["content"]}</p>',
                    unsafe_allow_html=True,
                )
                if step.get("tip"):
                    st.markdown(
                        f'<div style="background:#eff6ff;border-left:3px solid {color};'
                        f'padding:8px 12px;border-radius:0 6px 6px 0;'
                        f'font-size:12px;color:#334155;margin-bottom:10px;">'
                        f'{step["tip"]}</div>',
                        unsafe_allow_html=True,
                    )
                # Sub-checklist items
                sub_items = items_list[i] if i < len(items_list) else []
                if sub_items:
                    st.markdown(
                        '<div style="font-size:11px;font-weight:700;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">'
                        'Checklist</div>',
                        unsafe_allow_html=True,
                    )
                    for j, item in enumerate(sub_items):
                        ck_key = f"ck_{pw_key}_{i}_{j}"
                        if ck_key not in st.session_state:
                            st.session_state[ck_key] = is_done
                        st.checkbox(item, key=ck_key)

                # Progress indicator
                if sub_items:
                    done_count = sum(
                        st.session_state.get(f"ck_{pw_key}_{i}_{j}", False)
                        for j in range(len(sub_items))
                    )
                    pct = int(done_count / len(sub_items) * 100)
                    st.markdown(
                        f'<div style="margin-top:8px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:11px;color:#64748b;margin-bottom:4px;">'
                        f'<span>Progress</span><span>{done_count}/{len(sub_items)} items</span></div>'
                        f'<div style="background:#e2e8f0;border-radius:999px;height:6px;">'
                        f'<div style="width:{pct}%;background:{color};height:6px;'
                        f'border-radius:999px;transition:width .3s;"></div></div></div>',
                        unsafe_allow_html=True,
                    )

    def _render_pathway_tab(pw_key, pw_label, color):
        cl_left, cl_right = st.columns([2, 1], gap="large")

        with cl_left:
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;'
                f'letter-spacing:.06em;margin-bottom:12px;">Submission Steps</div>',
                unsafe_allow_html=True,
            )
            _render_stepper_expanders(pw_key, color)

            # Required Forms & Documents
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;'
                f'letter-spacing:.06em;margin:20px 0 12px;">Required Forms & Documents</div>',
                unsafe_allow_html=True,
            )
            for form_name, form_url, form_desc in FORMS[pw_key]:
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;'
                    f'margin-bottom:8px;background:#fff;">'
                    f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">'
                    f'<div style="flex:1;">'
                    f'<div style="font-size:13px;font-weight:600;color:#0b1c30;margin-bottom:3px;">{form_name}</div>'
                    f'<div style="font-size:12px;color:#64748b;">{form_desc}</div>'
                    f'</div>'
                    f'<a href="{form_url}" target="_blank" style="flex-shrink:0;font-size:11px;font-weight:700;'
                    f'color:{color};text-decoration:none;border:1px solid {color};padding:4px 10px;'
                    f'border-radius:4px;white-space:nowrap;">Open ↗</a>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        with cl_right:
            # Key Resources
            res_links = "".join(
                f'<a href="{url}" target="_blank" style="display:flex;align-items:center;gap:6px;'
                f'font-size:13px;color:{color};text-decoration:none;margin-bottom:8px;padding:6px 8px;'
                f'border-radius:6px;background:#f8fafc;border:1px solid #e2e8f0;">'
                f'{label} &rarr;</a>'
                for label, url in RESOURCES[pw_key]
            )
            st.markdown(
                f'<div class="fda-card"><div class="ic-title">Key Resources</div>{res_links}</div>',
                unsafe_allow_html=True,
            )

            # Typical Timeline
            tl_rows = "".join(
                f'<div style="display:flex;gap:10px;margin-bottom:10px;align-items:flex-start;">'
                f'<span style="font-size:11px;font-weight:700;color:{color};white-space:nowrap;'
                f'min-width:90px;padding-top:2px;">{period}</span>'
                f'<span style="font-size:13px;color:#374151;">{desc}</span></div>'
                for period, desc in TIMELINES[pw_key]
            )
            st.markdown(
                f'<div class="fda-card" style="margin-top:0;">'
                f'<div class="ic-title">Typical Timeline</div>{tl_rows}</div>',
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
                    <div class="ntc-txt">For complex submissions, consider requesting a Q-Sub meeting with FDA before filing.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Three tabs always visible ──────────────────────────────────────────────
    tab_510k, tab_pma, tab_denovo = st.tabs(["510(k)", "PMA", "De Novo"])

    with tab_510k:
        _render_pathway_tab("510k", "510(k)", "#378ADD")

    with tab_pma:
        _render_pathway_tab("PMA", "PMA", "#D85A30")

    with tab_denovo:
        _render_pathway_tab("De_Novo", "De Novo", "#1D9E75")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Pathway Predictor", key="back_to_pred"):
        st.session_state["_nav_target"] = "Pathway Predictor"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    import plotly.express as px
    import plotly.graph_objects as go

    PW_COLORS = {"510k": "#002046", "PMA": "#D85A30", "De_Novo": "#1D9E75"}
    PW_LABELS = {"510k": "510(k)", "PMA": "PMA", "De_Novo": "De Novo"}

    st.markdown(
        '<h1 style="font-size:26px;font-weight:700;color:#0f172a;">EDA Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;margin-bottom:24px;font-size:15px;">'
        'Distribution and correlation of all 18 record features with regulatory pathway.</p>',
        unsafe_allow_html=True,
    )

    fdf = load_features()
    if fdf is None:
        st.error("Feature data not found. Run the pipeline first.")
        st.stop()

    # ── Year range filter ───────────────────────────────────────────────────────
    import datetime as _dt
    _cur_year = _dt.date.today().year
    _min_year = int(fdf["decision_year"].min()) if "decision_year" in fdf.columns else 1990
    _max_year = int(fdf["decision_year"].max()) if "decision_year" in fdf.columns else _cur_year

    st.markdown(
        '<div style="font-size:14px;font-weight:600;color:#0b1c30;margin-bottom:6px;">'
        'Filter by Year</div>',
        unsafe_allow_html=True,
    )
    _yf_col1, _yf_col2, _yf_col3 = st.columns([2, 2, 6])
    with _yf_col1:
        _yr_from = st.number_input(
            "From", min_value=_min_year, max_value=_max_year,
            value=max(2000, _min_year), step=1, key="eda_yr_from",
        )
    with _yf_col2:
        _yr_to = st.number_input(
            "To", min_value=_min_year, max_value=_max_year,
            value=_max_year, step=1, key="eda_yr_to",
        )
    if _yr_from > _yr_to:
        st.warning("'From' year must be ≤ 'To' year.")
        _yr_from, _yr_to = _yr_to, _yr_from

    # Apply filter to dataframe
    if "decision_year" in fdf.columns:
        fdf = fdf[(fdf["decision_year"] >= _yr_from) & (fdf["decision_year"] <= _yr_to)]

    if len(fdf) == 0:
        st.warning("No records in the selected year range.")
        st.stop()

    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin-bottom:16px;">'
        f'Showing <b>{len(fdf):,}</b> submissions from <b>{_yr_from}</b> to <b>{_yr_to}</b></div>',
        unsafe_allow_html=True,
    )

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(fdf):,}")
    m2.metric("510(k)", f"{(fdf['pathway'] == '510k').sum():,}")
    m3.metric("PMA", f"{(fdf['pathway'] == 'PMA').sum():,}")
    m4.metric("De Novo", f"{(fdf['pathway'] == 'De_Novo').sum():,}")

    # ── Feature list (18 ML features, skip id/pathway/pathway_encoded) ─────────
    SKIP = {"submission_id", "pathway", "pathway_encoded"}
    feat_cols = [c for c in fdf.columns if c not in SKIP]

    # Correlation of each feature with pathway_encoded (numeric proxy)
    corr_series = (
        fdf[feat_cols + ["pathway_encoded"]]
        .corr()["pathway_encoded"]
        .drop("pathway_encoded")
        .dropna()
        .sort_values(key=abs, ascending=False)
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1a. Pathway Breakdown by Device Class ─────────────────────────────────
    st.markdown(
        '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:4px;">'
        'Which Pathway Does Each Device Class Take?</div>'
        '<p style="font-size:13px;color:#64748b;margin-bottom:16px;">'
        'Share of submissions reaching each regulatory pathway, grouped by FDA device class.</p>',
        unsafe_allow_html=True,
    )
    if "device_class" in fdf.columns:
        _dc = fdf[fdf["device_class"].isin([1, 2, 3])].copy()
        _dc["Device Class"] = _dc["device_class"].map({1: "Class I — Low Risk", 2: "Class II — Moderate Risk", 3: "Class III — High Risk"})
        _ct = _dc.groupby(["Device Class", "pathway"]).size().unstack(fill_value=0)
        _ct_pct = _ct.div(_ct.sum(axis=1), axis=0) * 100
        _pw_colors = {"510k": "#378ADD", "PMA": "#D85A30", "De_Novo": "#1D9E75"}
        _order = ["Class I — Low Risk", "Class II — Moderate Risk", "Class III — High Risk"]
        _ct_pct = _ct_pct.reindex([r for r in _order if r in _ct_pct.index])
        fig_dc = go.Figure()
        for pw in ["510k", "De_Novo", "PMA"]:
            if pw in _ct_pct.columns:
                fig_dc.add_trace(go.Bar(
                    name={"510k": "510(k)", "De_Novo": "De Novo", "PMA": "PMA"}[pw],
                    x=_ct_pct.index,
                    y=_ct_pct[pw],
                    marker_color=_pw_colors[pw],
                    text=[f"{v:.0f}%" if v >= 5 else "" for v in _ct_pct[pw]],
                    textposition="inside",
                    insidetextanchor="middle",
                    insidetextfont=dict(size=15, color="white", family="Inter"),
                    textfont=dict(size=15, family="Inter"),
                ))
        fig_dc.update_layout(
            barmode="stack",
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="% of submissions", ticksuffix="%", range=[0, 100], tickfont=dict(size=13)),
            xaxis=dict(tickfont=dict(size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=13)),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=14),
            uniformtext=dict(mode="hide", minsize=11),
        )
        st.plotly_chart(fig_dc, width='stretch')

    st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:24px 0;'>",
                unsafe_allow_html=True)

    # ── 1b. Predictive Strength of Each Feature ────────────────────────────────
    st.markdown(
        '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:4px;">'
        'Predictive Strength of Each Feature</div>'
        '<p style="font-size:13px;color:#64748b;margin-bottom:16px;">'
        'How useful each feature is for predicting the pathway — '
        'longer bar means stronger signal. Features with no meaningful signal are shown in gray.</p>',
        unsafe_allow_html=True,
    )
    # Merge sin/cos seasonal features and deduplicate same-label features (keep strongest)
    _MERGE = {
        "month_sin": "Seasonal Pattern", "month_cos": "Seasonal Pattern",
        "advisory_committee_encoded": "Medical Specialty",
        "medical_specialty_encoded": "Medical Specialty",
    }
    _raw = corr_series.abs()
    _seen, _dedup_idx, _dedup_vals, _dedup_lbls = {}, [], [], []
    for k in _raw.sort_values(ascending=False).index:
        lbl = _MERGE.get(k, FEAT_LABELS.get(k, k.replace("_", " ").title()))
        if lbl in _seen:
            continue  # keep only the strongest for duplicate labels
        _seen[lbl] = True
        _dedup_idx.append(k); _dedup_vals.append(_raw[k]); _dedup_lbls.append(lbl)
    # Re-sort ascending for horizontal bar display
    _sorted = sorted(zip(_dedup_vals, _dedup_lbls), key=lambda x: x[0])
    _abs_corr_vals = [v for v, _ in _sorted]
    _abs_labels = [l for _, l in _sorted]
    import pandas as _pd2
    _abs_corr = _pd2.Series(_abs_corr_vals, index=_abs_labels)
    def _strength_color(v):
        if v >= 0.3: return "#002046"
        if v >= 0.15: return "#378ADD"
        if v >= 0.05: return "#93C5FD"
        return "#CBD5E1"
    _max_val = _abs_corr.max() if len(_abs_corr) > 0 and _abs_corr.max() > 0 else 1.0
    _x_range = max(_max_val * 1.55, 0.5)  # always enough room for "Negligible" label
    fig_strength = go.Figure(go.Bar(
        x=_abs_corr.values,
        y=_abs_labels,
        orientation="h",
        marker_color=[_strength_color(v) for v in _abs_corr.values],
        text=[
            "Strong" if v >= 0.3 else "Moderate" if v >= 0.15 else "Weak" if v >= 0.05 else "Negligible"
            for v in _abs_corr.values
        ],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=13, family="Inter"),
    ))
    fig_strength.update_layout(
        height=max(360, len(_abs_corr) * 34),
        margin=dict(l=10, r=110, t=10, b=30),
        xaxis=dict(
            title="Predictive strength (0 = none, 1 = perfect)",
            range=[0, _x_range],
            tickfont=dict(size=13),
        ),
        yaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=13),
    )
    st.plotly_chart(fig_strength, width='stretch')

    st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:24px 0;'>",
                unsafe_allow_html=True)

    # ── 2. Per-Feature Distribution by Pathway ────────────────────────────────
    st.markdown(
        '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:16px;">'
        'Feature Distribution by Pathway</div>',
        unsafe_allow_html=True,
    )

    # Feature selector
    feat_display = {FEAT_LABELS.get(c, c.replace("_", " ").title()): c for c in feat_cols}
    selected_label = st.selectbox(
        "Select feature",
        options=list(feat_display.keys()),
        index=0,
        key="eda_feat_sel",
    )
    selected_feat = feat_display[selected_label]

    plot_df = fdf[[selected_feat, "pathway"]].copy()
    plot_df["Pathway"] = plot_df["pathway"].map(PW_LABELS).fillna(plot_df["pathway"])
    n_unique = plot_df[selected_feat].nunique()

    left_c, right_c = st.columns([3, 1], gap="large")

    with left_c:
        if n_unique <= 10:
            # Grouped bar chart for low-cardinality / binary features
            counts = (
                plot_df.groupby([selected_feat, "Pathway"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                counts,
                x=selected_feat,
                y="count",
                color="Pathway",
                barmode="group",
                color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                labels={selected_feat: selected_label, "count": "Submissions"},
                title=f"{selected_label} — Count by Pathway",
            )
        else:
            # Box plot for continuous features
            fig = px.box(
                plot_df,
                x="Pathway",
                y=selected_feat,
                color="Pathway",
                color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                labels={selected_feat: selected_label},
                title=f"{selected_label} — Distribution by Pathway",
                points="outliers",
            )
        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=12),
            margin=dict(l=0, r=0, t=36, b=0),
            legend_title_text="Pathway",
            showlegend=True,
        )
        st.plotly_chart(fig, width='stretch')

    with right_c:
        # Stats per pathway for this feature
        st.markdown(
            '<div style="font-size:12px;font-weight:700;color:#002046;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:12px;">Stats by Pathway</div>',
            unsafe_allow_html=True,
        )
        for pw_key, pw_label in PW_LABELS.items():
            vals = plot_df.loc[plot_df["pathway"] == pw_key, selected_feat].dropna()
            if vals.empty:
                continue
            if n_unique <= 10:
                mode_val = vals.mode().iloc[0] if not vals.mode().empty else "—"
                stat_html = (
                    f'<div style="margin-bottom:14px;">'
                    f'<div style="font-size:13px;font-weight:700;color:{PW_COLORS[pw_key]};">{pw_label}</div>'
                    f'<div style="font-size:12px;color:#374151;">n = {len(vals):,}</div>'
                    f'<div style="font-size:12px;color:#374151;">mode = {mode_val}</div>'
                    f'</div>'
                )
            else:
                stat_html = (
                    f'<div style="margin-bottom:14px;">'
                    f'<div style="font-size:13px;font-weight:700;color:{PW_COLORS[pw_key]};">{pw_label}</div>'
                    f'<div style="font-size:12px;color:#374151;">median = {vals.median():.1f}</div>'
                    f'<div style="font-size:12px;color:#374151;">mean = {vals.mean():.1f}</div>'
                    f'<div style="font-size:12px;color:#374151;">std = {vals.std():.1f}</div>'
                    f'</div>'
                )
            st.markdown(stat_html, unsafe_allow_html=True)

        r_val = corr_series.get(selected_feat, float("nan"))
        st.markdown(
            f'<div style="margin-top:8px;padding:12px;background:#f8fafc;border-radius:8px;">'
            f'<div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;">Correlation r</div>'
            f'<div style="font-size:22px;font-weight:700;color:{"#002046" if r_val >= 0 else "#D85A30"};">'
            f'{r_val:+.3f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3. All-Features Grid (mini charts) ────────────────────────────────────
    with st.expander("All features — pathway breakdown grid"):
        grid_cols = st.columns(3)
        for i, feat in enumerate(feat_cols):
            n_u = fdf[feat].nunique()
            label = FEAT_LABELS.get(feat, feat.replace("_", " ").title())
            gdf = fdf[[feat, "pathway"]].copy()
            gdf["Pathway"] = gdf["pathway"].map(PW_LABELS).fillna(gdf["pathway"])
            with grid_cols[i % 3]:
                if n_u <= 10:
                    gcounts = (
                        gdf.groupby([feat, "Pathway"]).size()
                        .reset_index(name="n")
                    )
                    gfig = px.bar(
                        gcounts, x=feat, y="n", color="Pathway", barmode="group",
                        color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                        title=label, labels={"n": ""},
                    )
                else:
                    gfig = px.box(
                        gdf, x="Pathway", y=feat, color="Pathway",
                        color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                        title=label, points=False,
                    )
                gfig.update_layout(
                    height=220, showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(family="Inter", size=10),
                )
                st.plotly_chart(gfig, width='stretch')

    st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:24px 0;'>",
                unsafe_allow_html=True)

    # ── 4. Device Specialty vs Pathway ────────────────────────────────────────
    st.markdown(
        '<div style="font-size:18px;font-weight:600;color:#002046;margin-bottom:4px;">'
        'Device Specialty vs Pathway</div>'
        '<p style="font-size:13px;color:#64748b;margin-bottom:16px;">'
        'Keyword-based classification of device names into high-level specialty categories '
        '(SaMD, IVD, Implantable, POC/Analyzer) and their regulatory pathway distribution.</p>',
        unsafe_allow_html=True,
    )

    cdf = load_clean()
    if cdf is not None:
        _SAMD_KW = [
            "software", "app", "algorithm", "ai ", "artificial intelligence",
            "machine learning", "clinical decision", "decision support",
            "image analysis", "cad ", "computer-aided", "neural network",
            "deep learning", "digital health", "mobile health", "mhealth",
            "samd", "remote monitoring", "telemedicine",
        ]
        _IVD_KW = [
            "ivd", "in vitro", "test strip", "glucometer",
            "reagent", "assay", "immunoassay", "lateral flow", "pcr",
            "elisa", "immunodiagnostic", "diagnostic kit",
            "blood glucose", "hba1c", "cholesterol test", "pregnancy test",
            "urinalysis", "urine test", "rapid test", "culture media",
        ]
        _POC_KW = [
            "point of care", "poc ", "analyzer", "analyser",
            "hematology analyzer", "coagulation analyzer",
        ]
        _IMPLANT_KW = [
            "implant", "implantable", "implanted",
            "pacemaker", "stent", "defibrillator", "icd ",
            "hip replacement", "knee replacement", "joint replacement",
            "intraocular lens", "iol ", "cochlear implant",
            "spinal cord stimulator", "deep brain stimulator",
            "breast implant", "vascular graft", "cardiac implant",
            "orthopedic implant", "dental implant", "bone screw",
            "neurostimulator", "sacral neuromodulation",
        ]

        _AC_LABELS = {
            "CV": "CV — Cardiovascular", "OR": "OR — Orthopedic",
            "SU": "SU — Surgery", "RA": "RA — Radiology",
            "CH": "CH — Chemistry", "GU": "GU — Gastro/Urology",
            "NE": "NE — Neurology", "OP": "OP — Ophthalmic",
            "AN": "AN — Anesthesiology", "DE": "DE — Dental",
            "EN": "EN — ENT", "HO": "HO — Hospital Devices",
            "PA": "PA — Pathology", "PM": "PM — Physical Medicine",
            "OB": "OB — OB/GYN", "IM": "IM — Immunology",
            "MI": "MI — Microbiology", "TX": "TX — Toxicology",
            "SAMD": "SaMD", "IVD": "IVD",
            "IMPLANT": "Implantable", "ANALYZER_POC": "POC / Analyzer",
            "UNKNOWN": "Unknown",
        }

        name_lower = cdf["device_name"].fillna("").str.lower()

        def _classify(name):
            if any(kw in name for kw in _SAMD_KW):
                return "SaMD"
            if any(kw in name for kw in _IVD_KW):
                return "IVD"
            if any(kw in name for kw in _POC_KW):
                return "POC / Analyzer"
            if any(kw in name for kw in _IMPLANT_KW):
                return "Implantable"
            return None

        cdf = cdf.copy()
        cdf["device_specialty"] = name_lower.apply(_classify)

        # For unclassified records, fall back to FDA advisory_committee label
        unclassified = cdf["device_specialty"].isna()
        cdf.loc[unclassified, "device_specialty"] = (
            cdf.loc[unclassified, "advisory_committee"]
            .map(_AC_LABELS)
            .fillna(cdf.loc[unclassified, "advisory_committee"])
        )

        cdf["Pathway"] = cdf["pathway"].map(PW_LABELS).fillna(cdf["pathway"])

        spec_tabs = st.tabs(["Keyword Specialties", "All Specialties", "Heatmap"])

        with spec_tabs[0]:
            # Only the 4 keyword-derived categories
            kw_spec = ["SaMD", "IVD", "POC / Analyzer", "Implantable"]
            kw_df = cdf[cdf["device_specialty"].isin(kw_spec)].copy()

            if kw_df.empty:
                st.info("No devices matched the keyword classification. Device names in this dataset may not contain the expected keywords.")
            else:
                kw_counts = (
                    kw_df.groupby(["device_specialty", "Pathway"])
                    .size()
                    .reset_index(name="count")
                )
                SPEC_COLORS = {
                    "SaMD": "#7C3AED", "IVD": "#0891B2",
                    "Implantable": "#D85A30", "POC / Analyzer": "#F59E0B",
                }
                fig_kw = px.bar(
                    kw_counts,
                    x="device_specialty",
                    y="count",
                    color="Pathway",
                    barmode="group",
                    color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                    labels={"device_specialty": "Device Specialty", "count": "Submissions"},
                    title="SaMD / IVD / POC / Implantable — Pathway Breakdown",
                )
                fig_kw.update_layout(
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(family="Inter", size=12),
                    margin=dict(l=0, r=0, t=36, b=0),
                )
                st.plotly_chart(fig_kw, width='stretch')

                # Summary table
                pivot = (
                    kw_df.groupby(["device_specialty", "pathway"])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={"510k": "510(k)", "De_Novo": "De Novo"})
                )
                pivot["Total"] = pivot.sum(axis=1)
                for pw in ["510(k)", "PMA", "De Novo"]:
                    if pw in pivot.columns:
                        pivot[f"{pw} %"] = (pivot[pw] / pivot["Total"] * 100).round(1)
                st.dataframe(pivot, width='stretch')

        with spec_tabs[1]:
            # All specialties (top 15 by count)
            top_specs = cdf["device_specialty"].value_counts().head(15).index
            all_df = cdf[cdf["device_specialty"].isin(top_specs)]
            all_counts = (
                all_df.groupby(["device_specialty", "Pathway"])
                .size()
                .reset_index(name="count")
            )
            fig_all = px.bar(
                all_counts,
                x="count",
                y="device_specialty",
                color="Pathway",
                barmode="stack",
                orientation="h",
                color_discrete_map={v: PW_COLORS[k] for k, v in PW_LABELS.items()},
                labels={"device_specialty": "Specialty", "count": "Submissions"},
                title="All Specialties — Stacked Pathway Distribution (Top 15)",
            )
            fig_all.update_layout(
                height=max(340, len(top_specs) * 28),
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(family="Inter", size=12),
                margin=dict(l=0, r=0, t=36, b=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_all, width='stretch')

        with spec_tabs[2]:
            # Heatmap: specialty (rows) × pathway (cols)
            top15 = cdf["device_specialty"].value_counts().head(15).index
            hm_df = cdf[cdf["device_specialty"].isin(top15)]
            hm_pivot = pd.crosstab(hm_df["device_specialty"], hm_df["pathway"])
            # Reorder columns
            col_order = [c for c in ["510k", "PMA", "De_Novo"] if c in hm_pivot.columns]
            hm_pivot = hm_pivot[col_order]
            hm_pivot.columns = [PW_LABELS.get(c, c) for c in hm_pivot.columns]
            hm_pivot = hm_pivot.loc[hm_pivot.sum(axis=1).sort_values(ascending=False).index]

            fig_hm = go.Figure(go.Heatmap(
                z=hm_pivot.values,
                x=list(hm_pivot.columns),
                y=list(hm_pivot.index),
                colorscale=[[0, "#f0f4ff"], [1, "#002046"]],
                text=hm_pivot.values,
                texttemplate="%{text:,}",
                showscale=True,
            ))
            fig_hm.update_layout(
                title="Specialty × Pathway Heatmap (submission counts)",
                height=max(300, len(hm_pivot) * 30),
                font=dict(family="Inter", size=12),
                margin=dict(l=0, r=0, t=36, b=0),
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_hm, width='stretch')

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
                st.image(str(img1), width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
        if img2.exists():
            with ic2:
                st.markdown(
                    '<div class="fda-card" style="padding:16px 20px 20px;">'
                    '<div class="ic-title">Feature Importance</div>',
                    unsafe_allow_html=True,
                )
                st.image(str(img2), width='stretch')
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
                width='stretch',
            )
