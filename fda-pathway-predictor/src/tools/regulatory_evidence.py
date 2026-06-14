"""Regulatory Evidence Engine — aggregate intelligence and score evidence strength.

Orchestrates:
  device_understanding -> structured device profile
  semantic_similarity  -> ranked similar devices
  predicate_lookup     -> scored predicate candidates
  FDA classification API (via similar_devices exemption check)

Scores evidence as HIGH / MEDIUM / LOW (informational only).
Never changes model predictions. Never raises — returns None on total failure.

Output: artifacts/regulatory_evidence_report.json
"""
import json, logging, requests
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Evidence scoring rules ─────────────────────────────────────────────────────
def _score_evidence(
    total_similar: int,
    predicate_count: int,
    has_classification: bool,
    has_product_code: bool,
) -> tuple[str, list[str]]:
    """
    Returns (strength, supporting_facts).

    HIGH:   product code + classification + >=3 predicates + >100 similar devices
    MEDIUM: classification + some similar devices
    LOW:    otherwise
    """
    facts = []

    if has_product_code:
        facts.append("Product code identified in FDA classification database")
    if has_classification:
        facts.append("Device classification confirmed via FDA API")
    if total_similar > 0:
        facts.append(f"{total_similar:,} similar devices found in historical FDA data")
    if predicate_count > 0:
        facts.append(f"{predicate_count} predicate candidate(s) located")
    if predicate_count == 0:
        facts.append("No predicate candidates found — device may be novel (consider De Novo)")

    if has_product_code and has_classification and predicate_count >= 3 and total_similar > 100:
        strength = "HIGH"
    elif has_classification and total_similar > 0:
        strength = "MEDIUM"
    else:
        strength = "LOW"
        if not has_classification:
            facts.append("No FDA classification record found for this product code / device name")
        if total_similar == 0:
            facts.append("No similar devices in historical data — evidence is limited")

    return strength, facts


# ── Classification lookup (lightweight, no regulatory_intel_tool dependency) ───
def _lookup_classification(product_code: str = None, device_name: str = None) -> dict:
    if not product_code and not device_name:
        return {}
    q = f"product_code:{product_code}" if product_code else f"device_name:{device_name}"
    try:
        r = requests.get(
            f"https://api.fda.gov/device/classification.json?search={q}&limit=1",
            timeout=8,
        )
        if r.status_code == 200:
            res = r.json().get("results", [])
            if res:
                rec = res[0]
                return {
                    "product_code": rec.get("product_code", product_code or ""),
                    "device_class": rec.get("device_class", ""),
                    "regulation_number": rec.get("regulation_number", ""),
                    "device_name": rec.get("device_name", ""),
                    "submission_type_id": rec.get("submission_type_id", "").strip(),
                    "advisory_committee": rec.get("advisory_committee", ""),
                    "medical_specialty": rec.get("medical_specialty_description", ""),
                }
    except Exception as e:
        logger.warning(f"Classification lookup: {e}")
    return {}


# ── Regulatory references (static, no scraping) ───────────────────────────────
_PATHWAY_REFS = {
    "510k": [
        {"label": "510(k) Submission Guidance",
         "url": "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k"},
        {"label": "Search 510(k) Database",
         "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"},
        {"label": "21 CFR Part 807 Subpart E — Premarket Notification",
         "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-807/subpart-E"},
    ],
    "PMA": [
        {"label": "PMA Application Guidance",
         "url": "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-approval-pma"},
        {"label": "Search PMA Database",
         "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm"},
        {"label": "21 CFR Part 814 — Premarket Approval",
         "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-814"},
    ],
    "De_Novo": [
        {"label": "De Novo Classification Guidance",
         "url": "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request"},
        {"label": "Search De Novo Database",
         "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm"},
        {"label": "21 CFR 860.260 — De Novo Request",
         "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-860/section-860.260"},
    ],
    "Exempt": [
        {"label": "Class I/II Exemptions Overview",
         "url": "https://www.fda.gov/medical-devices/classify-your-medical-device/class-i-and-class-ii-device-exemptions"},
        {"label": "21 CFR Part 807 Subpart B — Exemptions",
         "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-807/subpart-B"},
    ],
}
_GENERAL_REFS = [
    {"label": "FDA Device Classification Database",
     "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm"},
    {"label": "21 CFR Part 820 — Quality System Regulation",
     "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820"},
    {"label": "Biocompatibility — ISO 10993-1 Guidance",
     "url": "https://www.fda.gov/medical-devices/science-and-research-medical-devices/use-international-standard-iso-10993-1-biological-evaluation-medical-devices-part-1-evaluation-and"},
]


def _build_regulatory_refs(pathway: str, regulation_number: str = "") -> list:
    refs = list(_PATHWAY_REFS.get(pathway, _PATHWAY_REFS["510k"]))
    if regulation_number:
        prefix = regulation_number.replace(".", "")[:3]
        refs.insert(0, {
            "label": f"21 CFR Part {prefix} — Device-Specific Regulation",
            "url": f"https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-{prefix}",
        })
    refs.extend(_GENERAL_REFS)
    return refs


# ── Main entry point ───────────────────────────────────────────────────────────
def build_regulatory_evidence(
    device_name: str = "",
    device_description: str = "",
    device_class: int = None,
    advisory_committee: str = None,
    product_code: str = None,
    pathway: str = "510k",
    has_predicate: bool = False,
    data_path: str = "artifacts/clean_data.csv",
    output_path: str = "artifacts/regulatory_evidence_report.json",
) -> dict | None:
    """
    Build full regulatory evidence report. Returns None on total failure.

    Returns:
        {
            "evidence_strength": "HIGH",
            "supporting_facts": ["..."],
            "device_profile": {...},
            "classification": {...},
            "similar_devices": {...},
            "predicate_candidates": [...],
            "regulatory_references": [...],
        }
    """
    result: dict = {
        "evidence_strength": "LOW",
        "supporting_facts": [],
        "device_profile": {},
        "classification": {},
        "similar_devices": {},
        "predicate_candidates": [],
        "regulatory_references": [],
    }

    # 1. Device understanding
    try:
        from src.tools.device_understanding import extract_device_attributes
        profile = extract_device_attributes(
            device_name=device_name,
            device_description=device_description,
            device_class=device_class or 2,
            has_predicate=has_predicate,
            advisory_committee=advisory_committee,
        )
        result["device_profile"] = profile
        logger.info(f"Device profile: {profile.get('device_type')} method={profile.get('extraction_method')}")
    except Exception as e:
        logger.warning(f"Device understanding failed: {e}")

    # 2. FDA classification lookup
    classification = {}
    try:
        classification = _lookup_classification(
            product_code=product_code if product_code and product_code != "UNK" else None,
            device_name=device_name or None,
        )
        result["classification"] = classification
    except Exception as e:
        logger.warning(f"Classification lookup failed: {e}")

    has_classification = bool(classification)
    has_product_code = bool(
        classification.get("product_code")
        or (product_code and product_code not in ("UNK", ""))
    )
    resolved_product_code = classification.get("product_code") or product_code or ""
    regulation_number = classification.get("regulation_number", "")

    # 3. Semantic similarity
    semantic_result = None
    try:
        from src.tools.semantic_similarity import find_semantic_similar
        semantic_result = find_semantic_similar(
            device_name=device_name,
            device_description=device_description,
            device_class=device_class,
            advisory_committee=advisory_committee,
            top_n=10,
            data_path=data_path,
        )
        if semantic_result:
            result["similar_devices"] = semantic_result
            logger.info(
                f"Semantic similar: {semantic_result['total_matches']} total, "
                f"method={semantic_result.get('similarity_method')}"
            )
    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")

    # Fall back to structural similar devices if semantic failed
    total_similar = 0
    if semantic_result:
        total_similar = semantic_result.get("total_matches", 0)
    else:
        try:
            from src.tools.similar_devices import find_similar_devices
            sim = find_similar_devices(
                device_class=device_class,
                advisory_committee=advisory_committee,
                product_code=resolved_product_code,
                data_path=data_path,
            )
            if sim:
                result["similar_devices"] = sim
                total_similar = sim.get("total_similar", 0)
        except Exception as e:
            logger.warning(f"Structural similar devices failed: {e}")

    # 4. Predicate lookup
    predicates = []
    try:
        from src.tools.predicate_lookup import find_predicates
        predicates = find_predicates(
            device_name=device_name,
            device_description=device_description,
            device_class=device_class,
            advisory_committee=advisory_committee,
            product_code=resolved_product_code,
            pathway=pathway,
            top_n=5,
            data_path=data_path,
            semantic_result=semantic_result,
        )
        result["predicate_candidates"] = predicates
        logger.info(f"Predicate candidates: {len(predicates)}")
    except Exception as e:
        logger.warning(f"Predicate lookup failed: {e}")

    # 5. Evidence scoring
    strength, facts = _score_evidence(
        total_similar=total_similar,
        predicate_count=len(predicates),
        has_classification=has_classification,
        has_product_code=has_product_code,
    )
    result["evidence_strength"] = strength
    result["supporting_facts"] = facts

    # Add device profile facts
    profile = result.get("device_profile", {})
    if profile.get("software_based"):
        facts.append("Device identified as Software as a Medical Device (SaMD)")
    if profile.get("ai_enabled"):
        facts.append("AI/ML component detected — additional FDA guidance applies")
    if profile.get("implantable"):
        facts.append("Implantable device — enhanced biocompatibility testing required")
    if profile.get("life_sustaining"):
        facts.append("Life-sustaining device — heightened regulatory scrutiny expected")

    # 6. Regulatory references
    result["regulatory_references"] = _build_regulatory_refs(pathway, regulation_number)

    # Save
    try:
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Evidence report saved: strength={strength}, facts={len(facts)}")
    except Exception as e:
        logger.warning(f"Could not save evidence report: {e}")

    return result


if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    r = build_regulatory_evidence(
        device_name="Wireless cardiac rhythm monitor",
        device_description="ECG monitoring with AI-powered arrhythmia detection",
        device_class=2,
        advisory_committee="CV",
        product_code="DQK",
        pathway="510k",
    )
    if r:
        print(f"\nEvidence Strength: {r['evidence_strength']}")
        for fact in r["supporting_facts"]:
            print(f"  - {fact}")
