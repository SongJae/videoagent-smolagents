"""
Final Response Skill Executor

This module defines the execution logic for the final_response skill.
It generates structured battlefield analysis reports.
"""

from typing import Dict, Any, List

# Define available actions for this skill
ACTIONS = {
    "generate_report": {
        "description": "Generate a structured battlefield analysis report",
        "params": ["title", "summary", "findings", "video_evidence", "pdf_references",
                   "tactical_assessment", "recommendations", "methodology"]
    }
}


def execute(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a final_response action.

    Args:
        action: The action to perform
        params: Dictionary of parameters for the action

    Returns:
        Dictionary with status, action, result, and message
    """
    try:
        if action == "generate_report":
            return _generate_report(params)
        else:
            available_actions = ", ".join(ACTIONS.keys())
            return {
                "status": "error",
                "action": action,
                "result": None,
                "message": f"Unknown action: {action}. Available actions: {available_actions}"
            }

    except Exception as e:
        return {
            "status": "error",
            "action": action,
            "result": None,
            "message": f"Error executing final_response/{action}: {str(e)}"
        }


def _generate_report(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured report from the provided parameters."""

    # Extract parameters with defaults
    title = params.get("title", "Battlefield Analysis Report")
    summary = params.get("summary", "")
    findings = params.get("findings", [])
    video_evidence = params.get("video_evidence", [])
    pdf_references = params.get("pdf_references", [])
    tactical_assessment = params.get("tactical_assessment", "")
    recommendations = params.get("recommendations", [])
    methodology = params.get("methodology", "")

    # Generate structured report
    report_sections = []

    # Title and Summary
    report_sections.append(f"# {title}")
    if summary:
        report_sections.append(f"\n## Executive Summary\n{summary}")

    # Methodology (if provided)
    if methodology:
        report_sections.append(f"\n## Methodology\n{methodology}")

    # Key Findings
    if findings:
        report_sections.append("\n## Key Findings")
        for i, finding in enumerate(findings, 1):
            if isinstance(finding, dict):
                finding_text = finding.get("text", finding.get("description", str(finding)))
                confidence = finding.get("confidence", "")
                source = finding.get("source", "")
                entry = f"{i}. {finding_text}"
                if confidence:
                    entry += f" (Confidence: {confidence})"
                if source:
                    entry += f" [Source: {source}]"
                report_sections.append(entry)
            else:
                report_sections.append(f"{i}. {finding}")

    # Video Evidence
    if video_evidence:
        report_sections.append("\n## Video Evidence")
        for evidence in video_evidence:
            if isinstance(evidence, dict):
                video_id = evidence.get("video_id", "Unknown")
                segment = evidence.get("segment_id", "")
                time_range = evidence.get("time_range", "")
                description = evidence.get("description", "")
                objects = evidence.get("objects", [])

                entry = f"- **Video {video_id}**"
                if segment:
                    entry += f", Segment {segment}"
                if time_range:
                    entry += f" ({time_range})"
                if description:
                    entry += f": {description}"
                if objects:
                    entry += f" | Objects: {', '.join(str(o) for o in objects)}"
                report_sections.append(entry)
            else:
                report_sections.append(f"- {evidence}")

    # PDF References
    if pdf_references:
        report_sections.append("\n## Reference Documentation")
        for ref in pdf_references:
            if isinstance(ref, dict):
                source = ref.get("source", "Unknown")
                text = ref.get("text", "")
                relevance = ref.get("relevance_score", "")
                entry = f"- **{source}**"
                if relevance:
                    if isinstance(relevance, float):
                        entry += f" (Relevance: {relevance:.2f})"
                    else:
                        entry += f" (Relevance: {relevance})"
                if text:
                    # Truncate long text
                    text_preview = text[:200] + "..." if len(text) > 200 else text
                    entry += f"\n  > {text_preview}"
                report_sections.append(entry)
            else:
                report_sections.append(f"- {ref}")

    # Tactical Assessment
    if tactical_assessment:
        report_sections.append(f"\n## Tactical Assessment\n{tactical_assessment}")

    # Recommendations
    if recommendations:
        report_sections.append("\n## Recommendations")
        for i, rec in enumerate(recommendations, 1):
            report_sections.append(f"{i}. {rec}")

    # Assemble final report
    final_report = "\n".join(report_sections)

    return {
        "status": "success",
        "action": "generate_report",
        "result": {
            "report": final_report,
            "sections": {
                "title": title,
                "has_summary": bool(summary),
                "num_findings": len(findings),
                "num_video_evidence": len(video_evidence),
                "num_pdf_references": len(pdf_references),
                "has_tactical_assessment": bool(tactical_assessment),
                "num_recommendations": len(recommendations)
            }
        },
        "message": f"Generated report with {len(findings)} findings, {len(video_evidence)} video evidence items, and {len(recommendations)} recommendations"
    }
