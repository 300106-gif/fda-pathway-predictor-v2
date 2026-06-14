"""Device Understanding — convert free-text device info into structured regulatory attributes.

LLM extraction when API key is present; deterministic keyword fallback otherwise.
Never breaks existing app flow — all errors return a safe default dict.
"""
import json, logging, os, re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Keyword rule sets (deterministic fallback) ─────────────────────────────────
_SOFTWARE_KEYWORDS = {
    "software", "app", "application", "algorithm", "digital", "platform",
    "system", "samd", "mobile", "cloud", "data analytics", "analytics",
    "decision support", "clinical decision", "monitoring system",
}
_AI_KEYWORDS = {
    "artificial intelligence", "machine learning", "deep learning", "neural",
    "ai-powered", "ai enabled", "ai-enabled", "ml-based", "ml based",
    "computer vision", "natural language", "llm", "generative",
}
_IMPLANT_KEYWORDS = {
    "implant", "implantable", "implanted", "pacemaker", "stent", "defibrillator",
    "icd", "crt", "cochlear", "orthopedic implant", "spinal", "hip replacement",
    "knee replacement", "breast implant", "intraocular", "iol",
}
_LIFE_SUSTAIN_KEYWORDS = {
    "life sustaining", "life-sustaining", "life support", "ventilator",
    "dialysis", "hemodialysis", "cardiac support", "ventricular assist",
    "vad", "ecmo", "infusion pump", "drug delivery", "insulin pump",
}
_SPECIALTY_KEYWORDS = {
    "CV": {"cardiac", "cardiovascular", "heart", "artery", "vascular", "ecg", "ekg",
           "rhythm", "pacemaker", "defibrillator", "stent", "catheter"},
    "NE": {"neuro", "brain", "spinal cord", "epilepsy", "deep brain", "nerve",
           "neurology", "neurological", "cranial"},
    "OR": {"orthopedic", "bone", "joint", "hip", "knee", "spine", "fracture",
           "implant", "prosthetic", "arthroplasty"},
    "OP": {"ophthalmic", "eye", "ocular", "retina", "cornea", "intraocular",
           "lens", "glaucoma", "vision"},
    "IM": {"immunology", "microbiology", "infectious", "pathogen", "antibody",
           "antigen", "diagnostics", "assay", "test kit"},
    "MI": {"ivd", "in vitro", "clinical chemistry", "laboratory", "glucose",
           "blood glucose", "hba1c", "cholesterol", "chemistry analyzer"},
    "RA": {"radiology", "imaging", "mri", "ct scan", "x-ray", "ultrasound",
           "nuclear", "pet scan", "mammography", "fluoroscopy"},
    "SU": {"surgical", "surgery", "laparoscopic", "endoscopic", "robotic surgery",
           "wound", "suture", "staple", "ablation"},
    "DE": {"dental", "oral", "tooth", "teeth", "gum", "orthodontic", "periodontal"},
    "HO": {"hospital", "monitoring", "patient monitor", "general", "bedside",
           "wearable", "pulse oximeter", "thermometer", "blood pressure"},
    "AN": {"anesthesia", "anesthetic", "ventilator", "breathing", "airway",
           "intubation", "respiratory"},
}


def _text_contains(text: str, keywords: set) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _infer_specialty(text: str, advisory_committee: str = None) -> str:
    if advisory_committee and advisory_committee not in ("UNKNOWN", ""):
        return advisory_committee
    t = text.lower()
    best_ac, best_count = "HO", 0
    for ac, kws in _SPECIALTY_KEYWORDS.items():
        count = sum(1 for kw in kws if kw in t)
        if count > best_count:
            best_ac, best_count = ac, count
    return best_ac


def _deterministic_extract(
    device_name: str,
    device_description: str,
    device_class: int,
    has_predicate: bool,
    advisory_committee: str = None,
) -> dict:
    """Rule-based extraction — always available, no network required."""
    combined = f"{device_name} {device_description}".strip()
    return {
        "device_type": device_name.strip() or "Unknown",
        "specialty": _infer_specialty(combined, advisory_committee),
        "software_based": _text_contains(combined, _SOFTWARE_KEYWORDS),
        "ai_enabled": _text_contains(combined, _AI_KEYWORDS),
        "implantable": _text_contains(combined, _IMPLANT_KEYWORDS),
        "life_sustaining": _text_contains(combined, _LIFE_SUSTAIN_KEYWORDS),
        "predicate_claimed": has_predicate,
        "device_class": device_class,
        "extraction_method": "deterministic",
    }


def _llm_extract(
    device_name: str,
    device_description: str,
    device_class: int,
    has_predicate: bool,
) -> dict:
    """LLM-based extraction. Tries Anthropic then OpenAI."""
    prompt = f"""You are an FDA regulatory expert. Analyze this medical device and return a JSON object with the following fields:
- device_type: short device category name (e.g. "cardiac monitor", "orthopedic implant")
- specialty: FDA advisory committee code (AN/CV/CH/DE/EN/GU/HO/IM/MI/NE/OB/OP/OR/PA/PM/RA/SU/TX)
- software_based: true if the device is primarily software (SaMD, app, algorithm)
- ai_enabled: true if uses AI/ML
- implantable: true if the device is implanted in the body
- life_sustaining: true if device sustains life
- predicate_claimed: {has_predicate}
- device_class: {device_class}

Device Name: {device_name}
Device Description: {device_description}

Respond with ONLY valid JSON, no markdown."""

    # Try Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            result = json.loads(raw)
            result["extraction_method"] = "anthropic"
            return result
        except Exception as e:
            logger.warning(f"Anthropic extraction failed: {e}")

    # Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)
            result["extraction_method"] = "openai"
            return result
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}")

    return {}  # Caller falls back to deterministic


def extract_device_attributes(
    device_name: str = "",
    device_description: str = "",
    device_class: int = 2,
    has_predicate: bool = False,
    advisory_committee: str = None,
    output_path: str = "artifacts/structured_device_profile.json",
) -> dict:
    """
    Main entry point. Returns structured device profile dict.
    Never raises — falls back to deterministic extraction on any error.

    Returns:
        {
            "device_type": "cardiac rhythm monitor",
            "specialty": "CV",
            "software_based": True,
            "ai_enabled": False,
            "implantable": False,
            "life_sustaining": False,
            "predicate_claimed": True,
            "device_class": 2,
            "extraction_method": "deterministic" | "anthropic" | "openai",
        }
    """
    has_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    result = {}
    if has_llm and (device_name or device_description):
        result = _llm_extract(device_name, device_description, device_class, has_predicate)

    if not result:
        result = _deterministic_extract(
            device_name, device_description, device_class, has_predicate, advisory_committee
        )

    try:
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save device profile: {e}")

    return result


if __name__ == "__main__":
    import pprint
    r = extract_device_attributes(
        device_name="Wireless cardiac rhythm monitor",
        device_description="AI-powered wearable ECG monitor using deep learning algorithms",
        device_class=2,
        has_predicate=True,
        advisory_committee="CV",
    )
    pprint.pprint(r)
