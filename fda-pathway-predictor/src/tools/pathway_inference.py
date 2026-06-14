"""Pathway Inference Engine — predict FDA regulatory pathway from intended use and risk profile.

Works even when the exact device name/model has no match in the FDA database.

Steps:
  1. Extract risk factors from device name + description (no-network, always runs)
  2. Score against a curated product-code family table (deterministic)
  3. Query openFDA classification API with extracted keywords (network, graceful)
  4. Optionally enrich with LLM reasoning (Anthropic -> OpenAI, graceful)
  5. Return structured inference dict

Output shape:
  {
    "predicted_pathway":            "510k" | "510k_exempt" | "PMA" | "De_Novo",
    "predicted_class":              1 | 2 | 3,
    "predicted_product_code":       "QMX",
    "predicted_product_code_desc":  "Device Label Or Tag, Non-Sterile",
    "regulation_number":            "21 CFR 878.4800",
    "exemption_status":             "510(k) Exempt",
    "gmp_exempt":                   False,
    "confidence":                   "HIGH" | "MEDIUM" | "LOW",
    "reasoning":                    "...",
    "fda_logic":                    "...",
    "risk_factors":                 {...},
    "candidate_product_codes":      [...],
    "inference_method":             "llm" | "keyword" | "openfda",
  }
"""
import json, logging, os, re, requests
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Risk-factor keyword sets ────────────────────────────────────────────────────
_KW = {
    "patient_contact": {
        "skin", "tissue", "wound", "mucous", "mucosal",
        "applied to patient", "worn by patient", "placed on patient",
        "adheres to skin", "touches patient", "contact with patient",
        "patient contact", "direct contact", "transdermal", "topical",
        "dermal", "finger-stick", "fingerstick", "percutaneous",
        "worn on", "adheres to", "attaches to skin",
    },
    "fluid_contact": {
        "blood", "fluid contact", "contacts fluid", "contacts blood",
        "contacts drug", "contacts medication", "contact with blood",
        "contact with medication", "blood sample", "blood specimen",
        "intravascular", "bloodstream", "lumen", "blood glucose",
        "blood pressure measurement", "urine", "bodily fluid",
    },
    "sterile": {
        "sterile", "sterility", "aseptic", "sterilized", "eo sterilization",
        "gamma sterilization", "sterile field",
    },
    "life_sustaining": {
        "life sustaining", "life-sustaining", "life support", "ventilator",
        "dialysis", "hemodialysis", "cardiac support", "ventricular assist",
        "vad", "ecmo", "infusion pump", "insulin pump", "drug delivery pump",
        "sustains life",
    },
    "implantable": {
        "implant", "implantable", "implanted", "pacemaker", "coronary stent",
        "defibrillator", "icd", "crt", "cochlear implant", "orthopedic implant",
        "hip replacement", "knee replacement", "breast implant",
        "intraocular lens", "iol", "spinal implant", "bone implant",
    },
    "software_based": {
        "samd", "software as a medical device", "mobile app", "clinical algorithm",
        "decision support software", "cloud platform", "digital health platform",
        "health app", "mobile application", "software-based", "software based",
        "runs on", "cloud-based", "ai software", "medical software",
    },
    "ai_enabled": {
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "ai-powered", "ai-enabled", "ml-based", "llm",
        "computer vision", "natural language processing",
    },
    "diagnostic": {
        "diagnose", "diagnoses", "diagnosis", "detect", "detection",
        "monitor", "monitoring", "measure", "measurement", "analyze",
        "screen", "screening", "test", "assay", "imaging",
    },
    "therapeutic": {
        "treat", "treatment", "therapy", "deliver", "delivery", "stimulate",
        "ablate", "ablation", "irradiate", "apply energy", "electrical current",
        "therapeutic", "heat therapy", "cooling therapy",
    },
    "label_or_tag": {
        "label", "tag", "identification", "identify", "color code", "colour code",
        "color-coded", "marking", "marker", "flag", "sticker", "ribbon",
        "line id", "line identification", "visual identification",
    },
    "accessory": {
        "accessory", "clip", "holder", "mount", "bracket", "connector",
        "adapter", "cap", "plug", "strap", "band", "sleeve", "bag", "pouch",
        "does not contact", "does not control", "non-sterile", "non sterile",
        "snap-on", "snap on", "attaches to",
    },
    "iv_related": {
        "iv line", "intravenous line", "infusion line", "iv tubing",
        "iv set", "infusion set", "drip line", "iv identification",
        "infusion identification", "medication line",
    },
}

# ── Curated product-code family table ─────────────────────────────────────────
# Each entry: (score_keywords, product_code, description, class, cfr, is_510k_exempt, gmp_exempt)
_PC_TABLE = [
    # IV line label/tag/clip — no patient contact, no fluid contact
    ({"iv", "line", "identification", "label", "tag", "clip", "identify", "infusion"},
     "QMX", "Device Label or Tag, Non-Sterile", 1, "21 CFR 878.4800", True, False),

    # General disposable label / marking device
    ({"label", "tag", "marking", "identification", "color", "colour", "flag", "sticker"},
     "QMX", "Device Label or Tag, Non-Sterile", 1, "21 CFR 878.4800", True, False),

    # Pulse oximeter
    ({"pulse oximeter", "spo2", "oxygen saturation", "pulse ox"},
     "DQZ", "Pulse Oximeter", 2, "21 CFR 870.2700", False, False),

    # ECG / cardiac rhythm monitor
    ({"ecg", "ekg", "cardiac monitor", "rhythm monitor", "electrocardiograph",
      "heart rate monitor", "cardiac rhythm"},
     "DQK", "Electrocardiograph", 2, "21 CFR 870.2340", False, False),

    # Blood pressure monitor
    ({"blood pressure", "sphygmomanometer", "bp monitor", "bp cuff"},
     "DXN", "Non-Invasive Blood Pressure Monitor", 2, "21 CFR 870.1100", False, False),

    # Blood glucose monitor
    ({"glucose", "blood glucose", "glucometer", "glycemic", "hba1c", "diabetes monitor"},
     "NBW", "Blood Glucose Monitoring System", 2, "21 CFR 862.1345", False, False),

    # Infusion pump
    ({"infusion pump", "iv pump", "syringe pump", "drug delivery pump", "volumetric pump"},
     "FRN", "Infusion Pump", 2, "21 CFR 880.5860", False, False),

    # Insulin pump (external)
    ({"insulin pump", "insulin delivery", "continuous insulin", "subcutaneous insulin delivery"},
     "LZH", "Insulin Infusion System", 2, "21 CFR 880.5860", False, False),

    # Pacemaker
    ({"pacemaker", "cardiac pacemaker", "implantable pacemaker"},
     "LWS", "Pacemaker, Cardiac", 3, "21 CFR 870.3610", False, False),

    # Defibrillator / AED
    ({"defibrillator", "aed", "automated external defibrillator", "shock therapy"},
     "KZH", "Defibrillator", 3, "21 CFR 870.5300", False, False),

    # Coronary stent
    ({"coronary stent", "drug-eluting stent", "bare metal stent"},
     "NIQ", "Stent, Coronary", 3, "21 CFR 870.3545", False, False),

    # Wound dressing
    ({"wound dressing", "dressing", "bandage", "gauze", "wound care"},
     "FRK", "Wound Dressing", 1, "21 CFR 878.4015", True, False),

    # Electronic thermometer
    ({"thermometer", "temperature monitor", "fever", "body temperature"},
     "FLL", "Thermometer, Electronic, Clinical", 1, "21 CFR 880.2920", True, False),

    # Surgical glove
    ({"surgical glove", "examination glove", "latex glove", "nitrile glove"},
     "KZG", "Glove, Patient Examination", 1, "21 CFR 880.6250", True, False),

    # Surgical mask
    ({"surgical mask", "face mask", "medical mask", "procedure mask"},
     "FTL", "Mask, Surgical", 2, "21 CFR 878.4040", False, False),

    # Wheelchair
    ({"wheelchair", "power wheelchair", "motorized wheelchair", "electric wheelchair"},
     "IOR", "Wheelchair, Powered", 1, "21 CFR 890.3850", True, False),

    # Hearing aid
    ({"hearing aid", "hearing device", "hearing amplifier"},
     "IYO", "Hearing Aid", 1, "21 CFR 874.3300", False, False),

    # Ventilator
    ({"ventilator", "mechanical ventilator", "breathing machine", "life support ventilator"},
     "CAW", "Ventilator, Continuous, Home Use", 2, "21 CFR 868.5895", False, False),

    # CPAP
    ({"cpap", "bipap", "sleep apnea", "positive airway pressure", "apnea treatment"},
     "BZD", "Continuous Positive Airway Pressure Device", 2, "21 CFR 868.5130", False, False),

    # MRI
    ({"mri", "magnetic resonance imaging", "magnetic resonance scanner"},
     "ITY", "Magnetic Resonance Imaging System", 2, "21 CFR 892.1000", False, False),

    # Ultrasound
    ({"ultrasound", "sonography", "sonogram", "ultrasonic imaging"},
     "ITZ", "Ultrasonic Pulsed Doppler Imaging System", 2, "21 CFR 892.1560", False, False),

    # X-ray
    ({"x-ray", "xray", "radiograph", "digital radiography", "fluoroscopy"},
     "IZL", "X-Ray System, Diagnostic", 2, "21 CFR 892.1680", False, False),

    # Clinical decision support / SaMD
    ({"clinical decision support", "decision support software", "samd", "software as a medical device"},
     "QMF", "Software, Clinical Decision Support", 2, "21 CFR 880.3740", False, False),

    # AI/ML device
    ({"artificial intelligence", "machine learning", "deep learning", "ai-powered", "ai enabled"},
     "QMF", "AI/ML-Based Software Device", 2, "21 CFR 880.3740", False, False),

    # Cochlear implant
    ({"cochlear implant", "cochlear device", "hearing implant"},
     "MCM", "Cochlear Implant System", 3, "21 CFR 874.3900", False, False),

    # Hip prosthesis
    ({"hip implant", "hip replacement", "hip prosthesis", "total hip"},
     "KWC", "Prosthesis, Hip, Semi-Constrained", 3, "21 CFR 888.3310", False, False),

    # Knee prosthesis
    ({"knee implant", "knee replacement", "knee prosthesis", "total knee"},
     "KWH", "Prosthesis, Knee, Patellofemorotibial", 3, "21 CFR 888.3560", False, False),

    # Breast implant
    ({"breast implant", "mammary implant", "breast prosthesis"},
     "LYZ", "Prosthesis, Breast", 3, "21 CFR 878.3530", False, False),

    # Intraocular lens
    ({"intraocular lens", "iol", "cataract lens"},
     "HQL", "Lens, Intraocular", 3, "21 CFR 886.3600", False, False),

    # Suture
    ({"suture", "absorbable suture", "nonabsorbable suture", "surgical suture"},
     "GAG", "Suture, Nonabsorbable Synthetic Polymer", 2, "21 CFR 878.5000", False, False),

    # Needle / syringe
    ({"hypodermic needle", "syringe", "injection needle", "needle, hypodermic"},
     "KZF", "Needle, Hypodermic", 2, "21 CFR 880.5900", False, False),

    # Hospital bed
    ({"hospital bed", "patient bed", "medical bed", "adjustable bed"},
     "FPA", "Bed, Hospital", 1, "21 CFR 880.5100", True, False),

    # Stretcher
    ({"stretcher", "gurney", "patient stretcher", "transport stretcher"},
     "IYQ", "Stretcher", 1, "21 CFR 880.6700", True, False),

    # Pregnancy test
    ({"pregnancy test", "hcg test", "ovulation test", "fertility test"},
     "MDB", "Pregnancy Test, Over the Counter", 2, "21 CFR 884.1700", False, False),
]


def _flag(text: str, kw_set: set) -> bool:
    t = text.lower()
    return any(kw in t for kw in kw_set)


# Negation phrases that override a positive keyword match
_NEGATIONS: dict[str, list[str]] = {
    "patient_contact": [
        "does not contact the patient", "does not contact patient",
        "no patient contact", "no contact with patient",
        "not in contact with patient", "never contacts patient",
    ],
    "fluid_contact": [
        "does not contact the fluid", "does not contact fluid",
        "does not contact the drug", "does not contact drug",
        "does not contact the medication", "does not contact medication",
        "no fluid contact", "no contact with fluid",
        "not in contact with fluid", "does not contact the blood",
    ],
    "sterile": [
        "non-sterile", "non sterile", "not sterile",
        "does not require sterility", "no sterility",
    ],
    "life_sustaining": [
        "does not sustain life", "not life-sustaining", "not life sustaining",
        "does not control infusion", "does not control flow",
    ],
    "software_based": [
        "no software", "has no software", "without software",
        "does not contain software", "not software-based",
        "no app", "no algorithm",
    ],
    "ai_enabled": [
        "no ai", "no artificial intelligence", "no machine learning",
        "does not use ai", "not ai-powered",
    ],
}


def _extract_risk_factors(device_name: str, device_description: str) -> dict:
    combined = f"{device_name} {device_description}".lower()

    rf: dict = {}
    for key, kw_set in _KW.items():
        positive = _flag(combined, kw_set)
        # Apply negation override
        if positive and key in _NEGATIONS:
            negated = any(neg in combined for neg in _NEGATIONS[key])
            rf[key] = positive and not negated
        else:
            rf[key] = positive

    # Explicit "non-sterile" string always sets sterile = False
    if "non-sterile" in combined or "non sterile" in combined:
        rf["sterile"] = False

    # Derived flags
    rf["no_patient_contact"] = not rf.get("patient_contact") and not rf.get("fluid_contact")
    rf["high_risk"] = rf.get("life_sustaining") or rf.get("implantable")
    return rf


def _score_pc_table(combined: str) -> list[dict]:
    """Score curated product-code table against device text. Returns sorted candidates."""
    results = []
    t = combined.lower()
    for entry in _PC_TABLE:
        kws, pc, desc, cls, cfr, exempt, gmp = entry
        score = sum(1 for kw in kws if kw in t)
        if score > 0:
            results.append({
                "product_code": pc,
                "device_name": desc,
                "device_class": cls,
                "regulation_number": cfr,
                "is_510k_exempt": exempt,
                "gmp_exempt": gmp,
                "match_score": score,
                "source": "curated_table",
            })
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


def _openfda_search(terms: list[str], target_class: int = None) -> list[dict]:
    """Query openFDA /device/classification with extracted key terms. Graceful."""
    candidates = []
    seen = set()
    for term in terms[:4]:          # max 4 queries
        q = f'device_name:"{term}"'
        try:
            r = requests.get(
                "https://api.fda.gov/device/classification.json",
                params={"search": q, "limit": 5},
                timeout=7,
            )
            if r.status_code != 200:
                continue
            for rec in r.json().get("results", []):
                pc = rec.get("product_code", "")
                if pc in seen:
                    continue
                seen.add(pc)
                cls_raw = rec.get("device_class", "")
                try:
                    cls_int = int(cls_raw)
                except (ValueError, TypeError):
                    cls_int = 0
                # Score: +2 if class matches our prediction, +1 per term overlap
                dn = rec.get("device_name", "").lower()
                overlap = sum(1 for t in terms if t.lower() in dn)
                class_bonus = 2 if (target_class and cls_int == target_class) else 0
                candidates.append({
                    "product_code": pc,
                    "device_name": rec.get("device_name", ""),
                    "device_class": cls_int,
                    "regulation_number": rec.get("regulation_number", ""),
                    "submission_type_id": rec.get("submission_type_id", "").strip(),
                    "medical_specialty": rec.get("medical_specialty_description", ""),
                    "match_score": overlap + class_bonus,
                    "source": "openfda_api",
                })
        except Exception as e:
            logger.debug(f"openFDA search '{term}': {e}")
            continue
    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:8]


def _extract_search_terms(device_name: str, device_description: str) -> list[str]:
    """Pull key nouns/phrases from device text for openFDA queries."""
    STOP = {
        "a", "an", "the", "and", "or", "of", "to", "in", "is", "it",
        "for", "on", "with", "that", "this", "as", "by", "be", "are",
        "not", "does", "do", "no", "so", "its", "which", "has", "have",
        "from", "into", "used", "use", "can", "also", "each", "such",
        "will", "at", "but", "was", "were", "been", "any", "all",
    }
    combined = f"{device_name} {device_description}"
    # Extract multi-word phrases first (2-gram bigrams of interest)
    bigrams = []
    words = re.findall(r"[a-zA-Z]+", combined.lower())
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if words[i] not in STOP and words[i+1] not in STOP:
            bigrams.append(bg)
    # Single meaningful words
    singles = [w for w in words if len(w) > 3 and w not in STOP]
    # Deduplicate, keep most specific first (device name terms first)
    terms = []
    seen = set()
    name_words = re.findall(r"[a-zA-Z]+", device_name.lower())
    for w in name_words:
        if len(w) > 3 and w not in STOP and w not in seen:
            terms.append(w)
            seen.add(w)
    for bg in bigrams:
        if bg not in seen:
            terms.append(bg)
            seen.add(bg)
    for w in singles:
        if w not in seen:
            terms.append(w)
            seen.add(w)
    return terms[:6]


def _deterministic_predict(rf: dict, pc_candidates: list) -> dict:
    """Apply FDA-logic decision tree from risk factors. Returns pathway fields."""
    best_pc = pc_candidates[0] if pc_candidates else {}

    # Class III: life-sustaining or implantable → PMA (unless strong predicate path)
    if rf["life_sustaining"] or rf["implantable"]:
        pc = best_pc.get("product_code", "")
        cls = best_pc.get("device_class", 3)
        cfr = best_pc.get("regulation_number", "")
        if cls == 3 or rf["life_sustaining"]:
            return {
                "predicted_pathway": "PMA",
                "predicted_class": 3,
                "predicted_product_code": pc,
                "predicted_product_code_desc": best_pc.get("device_name", ""),
                "regulation_number": cfr,
                "exemption_status": "Requires PMA",
                "gmp_exempt": False,
                "fda_logic": (
                    "Life-sustaining or implantable device with significant risk — "
                    "Class III classification applies. Premarket Approval (PMA) required "
                    "unless a valid predicate exists for 510(k) submission (Class II upgrade)."
                ),
            }
        # Class II implantable with predicate
        return {
            "predicted_pathway": "510k",
            "predicted_class": 2,
            "predicted_product_code": pc,
            "predicted_product_code_desc": best_pc.get("device_name", ""),
            "regulation_number": cfr,
            "exemption_status": "Requires 510(k)",
            "gmp_exempt": False,
            "fda_logic": (
                "Implantable device — Class II. "
                "A valid predicate device may support 510(k) Premarket Notification."
            ),
        }

    # Accessory / label / tag with NO patient contact and NO fluid contact
    if (rf["label_or_tag"] or rf["accessory"]) and rf["no_patient_contact"]:
        pc = best_pc.get("product_code", "QMX")
        desc = best_pc.get("device_name", "Device Label or Tag, Non-Sterile")
        cfr = best_pc.get("regulation_number", "21 CFR 878.4800")
        gmp = best_pc.get("gmp_exempt", False)
        return {
            "predicted_pathway": "510k_exempt",
            "predicted_class": 1,
            "predicted_product_code": pc,
            "predicted_product_code_desc": desc,
            "regulation_number": cfr,
            "exemption_status": "510(k) Exempt",
            "gmp_exempt": gmp,
            "fda_logic": (
                "Non-contact accessory/identification device — no patient contact, no fluid "
                "contact, no active components. Falls into Class I under 21 CFR Part 807.65 "
                "510(k) exemption provisions. Quality System Regulation (21 CFR Part 820 / "
                "QMSR) still applies unless the device is also listed as GMP-exempt."
            ),
        }

    # Software / AI — Class II
    if rf["software_based"] or rf["ai_enabled"]:
        pc = best_pc.get("product_code", "QMF")
        cfr = best_pc.get("regulation_number", "21 CFR 880.3740")
        return {
            "predicted_pathway": "510k",
            "predicted_class": 2,
            "predicted_product_code": pc,
            "predicted_product_code_desc": best_pc.get("device_name", "Software, Clinical Decision Support"),
            "regulation_number": cfr,
            "exemption_status": "Requires 510(k) or De Novo",
            "gmp_exempt": False,
            "fda_logic": (
                "Software as a Medical Device (SaMD) or AI/ML-based device — "
                "generally Class II requiring 510(k). Novel AI/ML functions without "
                "a predicate may require De Novo classification."
            ),
        }

    # Patient contact but not life-sustaining / implantable → Class II → 510(k)
    if rf["patient_contact"] or rf["fluid_contact"]:
        pc = best_pc.get("product_code", "")
        cfr = best_pc.get("regulation_number", "")
        return {
            "predicted_pathway": "510k",
            "predicted_class": 2,
            "predicted_product_code": pc,
            "predicted_product_code_desc": best_pc.get("device_name", ""),
            "regulation_number": cfr,
            "exemption_status": "Requires 510(k)",
            "gmp_exempt": False,
            "fda_logic": (
                "Device has direct or indirect patient/fluid contact — "
                "Class II with moderate risk. 510(k) Premarket Notification required "
                "demonstrating substantial equivalence to a cleared predicate."
            ),
        }

    # Low-risk, no contact → Class I, likely exempt
    pc = best_pc.get("product_code", "")
    cfr = best_pc.get("regulation_number", "")
    gmp = best_pc.get("gmp_exempt", False)
    return {
        "predicted_pathway": "510k_exempt",
        "predicted_class": 1,
        "predicted_product_code": pc,
        "predicted_product_code_desc": best_pc.get("device_name", ""),
        "regulation_number": cfr,
        "exemption_status": "510(k) Exempt (Class I)",
        "gmp_exempt": gmp,
        "fda_logic": (
            "Low-risk device with no patient contact, no active components, and no life-sustaining "
            "function — Class I classification. Likely exempt from 510(k) premarket notification "
            "under 21 CFR Part 807.65. Quality System Regulation (QMSR) may still apply."
        ),
    }


def _llm_infer(
    device_name: str,
    device_description: str,
    risk_factors: dict,
    pc_candidates: list,
) -> dict:
    """LLM-enriched inference. Tries Anthropic → OpenAI. Returns {} on failure."""
    top_pc = pc_candidates[:3] if pc_candidates else []
    top_pc_str = json.dumps(top_pc, indent=2) if top_pc else "none found"

    rf_readable = {
        "Patient contact": risk_factors.get("patient_contact"),
        "Fluid contact": risk_factors.get("fluid_contact"),
        "Sterile": risk_factors.get("sterile"),
        "Life-sustaining": risk_factors.get("life_sustaining"),
        "Implantable": risk_factors.get("implantable"),
        "Software-based": risk_factors.get("software_based"),
        "AI-enabled": risk_factors.get("ai_enabled"),
        "Diagnostic function": risk_factors.get("diagnostic"),
        "Therapeutic function": risk_factors.get("therapeutic"),
        "Label / tag / accessory": risk_factors.get("label_or_tag") or risk_factors.get("accessory"),
        "IV-related": risk_factors.get("iv_related"),
    }

    prompt = f"""You are a senior FDA regulatory affairs specialist. A user has described a medical device and you must predict its FDA regulatory pathway using intended use and risk profile — even if the exact device model is not in the FDA database.

Device Name: {device_name}
Device Description: {device_description}

Extracted Risk Factors:
{json.dumps(rf_readable, indent=2)}

Top product code candidates found so far:
{top_pc_str}

Apply FDA's risk-based classification logic:
1. Assess patient contact, fluid contact, sterility, life-sustaining function, implantability
2. Consider whether this is software, diagnostic, therapeutic, or an accessory/label
3. Identify the most likely FDA product code family and 21 CFR regulation
4. Determine whether 510(k) premarket notification is required or if the device is exempt
5. Check whether GMP/Quality System (21 CFR Part 820) applies

Return ONLY a valid JSON object with these exact keys:
{{
  "predicted_pathway": "510k_exempt" | "510k" | "PMA" | "De_Novo",
  "predicted_class": 1 | 2 | 3,
  "predicted_product_code": "e.g. QMX",
  "predicted_product_code_desc": "e.g. Device Label or Tag, Non-Sterile",
  "regulation_number": "e.g. 21 CFR 878.4800",
  "exemption_status": "e.g. 510(k) Exempt",
  "gmp_exempt": true | false,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "2-3 sentence explanation of your decision",
  "fda_logic": "Step-by-step FDA reasoning: patient contact assessment -> risk class -> pathway",
  "key_risk_questions": ["Does it contact the patient?", "..."],
  "alternative_pathway": "alternative if primary is uncertain, or null"
}}

No markdown, no explanation outside the JSON."""

    for provider in ("anthropic", "openai"):
        key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            continue
        try:
            if provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text.strip()
            else:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()

            # Strip any markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            result["inference_method"] = f"llm_{provider}"
            return result
        except Exception as e:
            logger.warning(f"LLM inference ({provider}) failed: {e}")

    return {}


def _confidence_from_sources(
    llm_result: dict,
    pc_candidates: list,
    openfda_candidates: list,
) -> str:
    if llm_result:
        return llm_result.get("confidence", "MEDIUM")
    if pc_candidates and pc_candidates[0]["match_score"] >= 3:
        return "HIGH"
    if pc_candidates and openfda_candidates:
        return "MEDIUM"
    if pc_candidates or openfda_candidates:
        return "LOW"
    return "LOW"


def infer_regulatory_pathway(
    device_name: str = "",
    device_description: str = "",
    device_class: int = None,
    advisory_committee: str = None,
) -> dict:
    """
    Main entry point. Returns a full pathway inference dict.
    Never raises — returns a minimal dict on total failure.

    Parameters
    ----------
    device_name         : free-text device name (can be brand/model/generic)
    device_description  : intended use, mechanism, patient contact details
    device_class        : hint from ML model (1/2/3); used to bias openFDA scoring
    advisory_committee  : hint from ML model (e.g. "CV"); informational only
    """
    fallback = {
        "predicted_pathway": "510k",
        "predicted_class": device_class or 2,
        "predicted_product_code": "",
        "predicted_product_code_desc": "",
        "regulation_number": "",
        "exemption_status": "Requires 510(k)",
        "gmp_exempt": False,
        "confidence": "LOW",
        "reasoning": "Insufficient information to make a confident prediction.",
        "fda_logic": "",
        "risk_factors": {},
        "candidate_product_codes": [],
        "inference_method": "fallback",
        "key_risk_questions": [],
        "alternative_pathway": None,
    }

    if not device_name and not device_description:
        return fallback

    try:
        combined = f"{device_name} {device_description}"

        # Step 1 — risk factor extraction
        risk_factors = _extract_risk_factors(device_name, device_description)

        # Step 2 — curated product-code scoring
        pc_candidates = _score_pc_table(combined)

        # Step 3 — openFDA semantic search
        search_terms = _extract_search_terms(device_name, device_description)
        openfda_candidates = _openfda_search(search_terms, target_class=device_class)

        # Merge candidates (curated first, then openFDA)
        all_candidates = pc_candidates + [
            c for c in openfda_candidates
            if c["product_code"] not in {x["product_code"] for x in pc_candidates}
        ]

        # Step 4 — deterministic prediction (baseline)
        prediction = _deterministic_predict(risk_factors, pc_candidates or openfda_candidates)

        # Step 5 — LLM enrichment (replaces/overrides deterministic if available)
        llm_result = _llm_infer(device_name, device_description, risk_factors, all_candidates[:3])

        # Merge: LLM fields take priority, fall back to deterministic
        if llm_result:
            for field in (
                "predicted_pathway", "predicted_class", "predicted_product_code",
                "predicted_product_code_desc", "regulation_number",
                "exemption_status", "gmp_exempt", "reasoning", "fda_logic",
                "key_risk_questions", "alternative_pathway",
            ):
                if field in llm_result:
                    prediction[field] = llm_result[field]
            prediction["inference_method"] = llm_result.get("inference_method", "llm")
        else:
            # Deterministic — build readable reasoning from risk factors
            rf_flags = [k for k, v in risk_factors.items() if v and k not in ("no_patient_contact", "high_risk")]
            prediction.setdefault("reasoning", (
                f"Based on risk profile analysis: "
                f"{'no patient or fluid contact detected' if risk_factors.get('no_patient_contact') else 'patient or fluid contact detected'}. "
                f"Active risk flags: {', '.join(rf_flags) if rf_flags else 'none'}."
            ))
            prediction.setdefault("key_risk_questions", [
                "Does the device contact the patient (skin/tissue/blood)?",
                "Does the device contact the medication or fluid being infused?",
                "Is the device sterile or does it require sterility?",
                "Is the device life-sustaining or implantable?",
                "Does the device contain software or AI/ML components?",
            ])
            prediction.setdefault("alternative_pathway", None)
            prediction["inference_method"] = "keyword_deterministic"

        # Attach metadata
        prediction["risk_factors"] = risk_factors
        prediction["candidate_product_codes"] = all_candidates[:6]
        prediction["confidence"] = _confidence_from_sources(llm_result, pc_candidates, openfda_candidates)

        return prediction

    except Exception as e:
        logger.warning(f"Pathway inference failed: {e}")
        return fallback
