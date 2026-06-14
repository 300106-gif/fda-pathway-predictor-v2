"""Crew 3 — Regulatory Intelligence Crew.

Single agent: Regulatory Research Agent
Wraps regulatory_evidence.py which internally calls:
  device_understanding, semantic_similarity, predicate_lookup, FDA API
"""
import logging
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class RegulatoryEvidenceTool(BaseTool):
    """Aggregates regulatory evidence for a device submission."""
    name: str = "regulatory_evidence_engine"
    description: str = (
        "Builds a complete regulatory evidence report for a medical device by retrieving "
        "FDA classifications, identifying similar historical devices, finding predicate "
        "candidates from 510(k) records, and scoring overall evidence strength (HIGH/MEDIUM/LOW). "
        "Input: device_name, device_description, device_class, advisory_committee, "
        "product_code, pathway. "
        "Output: evidence strength, supporting facts, predicate candidates, regulatory references."
    )

    device_name: str = ""
    device_description: str = ""
    device_class: int = 2
    advisory_committee: str = ""
    product_code: str = ""
    pathway: str = "510k"

    def _run(self, **kwargs) -> str:
        from src.tools.regulatory_evidence import build_regulatory_evidence

        name = kwargs.get("device_name", self.device_name)
        desc = kwargs.get("device_description", self.device_description)
        cls  = int(kwargs.get("device_class", self.device_class))
        ac   = kwargs.get("advisory_committee", self.advisory_committee)
        pc   = kwargs.get("product_code", self.product_code)
        pw   = kwargs.get("pathway", self.pathway)

        result = build_regulatory_evidence(
            device_name=name,
            device_description=desc,
            device_class=cls,
            advisory_committee=ac,
            product_code=pc,
            pathway=pw,
        )

        if not result:
            return "Regulatory evidence collection failed — all sources unavailable."

        strength    = result.get("evidence_strength", "LOW")
        facts       = result.get("supporting_facts", [])
        predicates  = result.get("predicate_candidates", [])
        similar     = result.get("similar_devices", {})
        profile     = result.get("device_profile", {})
        refs        = result.get("regulatory_references", [])

        total_similar = (
            similar.get("total_matches") or similar.get("total_similar") or 0
        )

        lines = [
            f"Evidence Strength: {strength}",
            f"Device Type: {profile.get('device_type', 'Unknown')}",
            f"Software-based: {profile.get('software_based', False)}",
            f"AI-enabled: {profile.get('ai_enabled', False)}",
            f"Implantable: {profile.get('implantable', False)}",
            "",
            f"Supporting Evidence ({len(facts)} facts):",
        ]
        for fact in facts:
            lines.append(f"  - {fact}")

        lines += [
            "",
            f"Similar Devices: {total_similar:,}",
            f"Predicate Candidates: {len(predicates)}",
        ]
        for p in predicates[:3]:
            lines.append(
                f"  [{p.get('submission_id', 'N/A')}] {p.get('device_name', '')} "
                f"(score={p.get('similarity_score', 0):.2f})"
            )

        lines += [
            "",
            f"Regulatory References: {len(refs)} links available",
            "Full report saved to artifacts/regulatory_evidence_report.json",
        ]

        return "\n".join(lines)


def create_regulatory_crew(
    device_name: str = "",
    device_description: str = "",
    device_class: int = 2,
    advisory_committee: str = "",
    product_code: str = "",
    pathway: str = "510k",
) -> Crew:
    """Create Crew 3: Regulatory Intelligence Crew."""
    tool = RegulatoryEvidenceTool(
        device_name=device_name,
        device_description=device_description,
        device_class=device_class,
        advisory_committee=advisory_committee,
        product_code=product_code,
        pathway=pathway,
    )

    researcher = Agent(
        role="Regulatory Research Agent",
        goal=(
            "Retrieve comprehensive FDA regulatory intelligence for a medical device: "
            "confirm classification, gather similar historical devices, identify predicate "
            "candidates, and score overall evidence strength."
        ),
        backstory=(
            "You are a senior FDA regulatory affairs specialist with 20 years of experience. "
            "You navigate FDA databases, eCFR, and CDRH resources to build complete "
            "regulatory profiles that support premarket submission decisions."
        ),
        tools=[tool],
        verbose=True,
        allow_delegation=False,
    )

    inputs_desc = []
    if device_name:        inputs_desc.append(f"device '{device_name}'")
    if device_class:       inputs_desc.append(f"Class {device_class}")
    if advisory_committee: inputs_desc.append(f"advisory committee {advisory_committee}")
    if pathway:            inputs_desc.append(f"predicted pathway {pathway}")
    inputs_str = ", ".join(inputs_desc) if inputs_desc else "the device under analysis"

    task = Task(
        description=(
            f"Build a complete regulatory evidence report for {inputs_str}. "
            "Use the regulatory_evidence_engine tool to: "
            "(1) extract structured device attributes, "
            "(2) retrieve FDA classification and regulation number, "
            "(3) identify semantically similar historical FDA devices, "
            "(4) find top predicate candidates from 510(k) records, "
            "(5) score evidence strength as HIGH/MEDIUM/LOW, "
            "(6) compile relevant regulatory references. "
            "Save all results to artifacts/regulatory_evidence_report.json."
        ),
        expected_output=(
            "Evidence strength score (HIGH/MEDIUM/LOW), list of supporting facts, "
            "count of similar devices, count of predicate candidates with top examples, "
            "and confirmation that regulatory_evidence_report.json was saved."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    crew = create_regulatory_crew(
        device_name="Wireless cardiac rhythm monitor",
        device_class=2,
        advisory_committee="CV",
        pathway="510k",
    )
    print(f"Regulatory Intelligence Crew — agents: {[a.role for a in crew.agents]}")
