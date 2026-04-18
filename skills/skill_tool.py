"""
Skill Tools: Smolagents tool wrappers for the Agent Skills pattern.

This module provides smolagents-compatible tools that invoke skills dynamically.
No hardcoding is required when new skills are added - SkillManager auto-discovers
skills from their executor.py files.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import smolagents
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

from smolagents import tool

# Import skill manager
from .skill_manager import get_skill_manager


# =============================================================================
# SKILL-BASED TOOLS
# These tools wrap skills for use by the smolagents CodeAgent
# =============================================================================

@tool
def get_available_skills(include_actions: bool = True) -> dict:
    """
    Get a list of all available skills that can be invoked.
    Call this to discover what skills are available and their actions.

    Args:
        include_actions: Whether to include available actions for each skill

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - skills: List of available skills with name, description, and actions
        - message: Summary of available skills

    Example:
        skills = get_available_skills()
        for s in skills["skills"]:
            print(f"{s['name']}: {s['description']}")
            if 'actions' in s:
                print(f"  Actions: {s['actions']}")
    """
    try:
        manager = get_skill_manager()
        skills_list = manager.list_skills(include_actions=include_actions)

        return {
            "status": "success",
            "skills": skills_list,
            "message": f"Found {len(skills_list)} available skill(s): {', '.join(s['name'] for s in skills_list)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "skills": [],
            "message": f"Error getting available skills: {str(e)}"
        }


@tool
def get_skill_actions(skill_name: str) -> dict:
    """
    Get available actions for a specific skill.

    Args:
        skill_name: Name of the skill (e.g., "videodb_query", "pdf_rag")

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - skill_name: The skill name
        - actions: List of available actions with their parameters
        - message: Summary

    Example:
        actions = get_skill_actions("videodb_query")
        for a in actions["actions"]:
            print(f"{a['name']}: {a['description']}")
    """
    try:
        manager = get_skill_manager()
        actions = manager.get_skill_actions(skill_name)

        if not actions:
            return {
                "status": "error",
                "skill_name": skill_name,
                "actions": [],
                "message": f"Skill '{skill_name}' not found or has no actions"
            }

        return {
            "status": "success",
            "skill_name": skill_name,
            "actions": actions,
            "message": f"Found {len(actions)} action(s) for skill '{skill_name}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "skill_name": skill_name,
            "actions": [],
            "message": f"Error getting skill actions: {str(e)}"
        }


@tool
def invoke_skill(skill_name: str, action: str,
                 # Generic parameters that work across skills
                 query: Optional[str] = None,
                 object_type: Optional[str] = None,
                 video_ids: Optional[list] = None,
                 segment_id: Optional[int] = None,
                 video_id: Optional[str] = None,
                 pdf_sources: Optional[list] = None,
                 unit_name: Optional[str] = None,
                 unit_type: Optional[str] = None,
                 top_k: int = 5,
                 min_count: int = 1,
                 # Report generation parameters
                 title: Optional[str] = None,
                 summary: Optional[str] = None,
                 findings: Optional[list] = None,
                 video_evidence: Optional[list] = None,
                 pdf_references: Optional[list] = None,
                 tactical_assessment: Optional[str] = None,
                 recommendations: Optional[list] = None,
                 methodology: Optional[str] = None) -> dict:
    """
    Invoke a skill action dynamically. This is the main entry point for using skills.
    New skills are automatically available without code changes.

    Use get_available_skills() to see all available skills and their actions.

    Args:
        skill_name: Name of the skill to invoke (e.g., "videodb_query", "pdf_rag", "wargame_query", "final_response")
        action: Specific action within the skill (e.g., "get_contexts", "search", "generate_report")
        query: Search query for semantic/event search or PDF search
        object_type: Object type for object_search (tank, truck, soldier)
        video_ids: Optional list of video IDs to search
        segment_id: Segment ID for segment_details
        video_id: Single video ID (for segment_details)
        pdf_sources: Optional list of PDF filenames for pdf_rag search
        unit_name: Unit name for wargame unit_details/unit_waypoints
        unit_type: Unit type for wargame units_by_type (INFANTRY, ARMOR, etc.)
        top_k: Number of results for search actions (default: 5)
        min_count: Minimum object count for object_search (default: 1)
        title: Report title for final_response
        summary: Executive summary for final_response
        findings: List of findings for final_response
        video_evidence: Video evidence list for final_response
        pdf_references: PDF references list for final_response
        tactical_assessment: Tactical assessment for final_response
        recommendations: Recommendations list for final_response
        methodology: Methodology description for final_response

    Returns:
        Dictionary with:
        - status: "success", "no_results", or "error"
        - action: The action that was performed
        - result: The result data from the skill
        - message: Human-readable description of the result

    Example:
        # Get selected contexts first
        contexts = invoke_skill("videodb_query", "get_contexts")

        # Search for tanks
        result = invoke_skill("videodb_query", "object_search", object_type="tank")

        # Semantic search
        result = invoke_skill("videodb_query", "semantic_search", query="convoy movement")

        # Search PDFs
        result = invoke_skill("pdf_rag", "search", query="tank specifications")

        # Get tactical situation
        result = invoke_skill("wargame_query", "tactical_situation")

        # Generate final report
        report = invoke_skill("final_response", "generate_report",
            title="Analysis Report",
            summary="Brief summary...",
            findings=["Finding 1", "Finding 2"]
        )
    """
    try:
        manager = get_skill_manager()

        # Build params dictionary from all provided arguments
        params = {}

        # Query-related params
        if query is not None:
            params["query"] = query
            params["event_query"] = query  # For event_search compatibility
        if object_type is not None:
            params["object_type"] = object_type
        if video_ids is not None:
            params["video_ids"] = video_ids
        if segment_id is not None:
            params["segment_id"] = segment_id
        if video_id is not None:
            params["video_id"] = video_id
        if pdf_sources is not None:
            params["pdf_sources"] = pdf_sources
        if unit_name is not None:
            params["unit_name"] = unit_name
        if unit_type is not None:
            params["unit_type"] = unit_type
        params["top_k"] = top_k
        if min_count != 1:
            params["min_count"] = min_count

        # Report generation params
        if title is not None:
            params["title"] = title
        if summary is not None:
            params["summary"] = summary
        if findings is not None:
            params["findings"] = findings
        if video_evidence is not None:
            params["video_evidence"] = video_evidence
        if pdf_references is not None:
            params["pdf_references"] = pdf_references
        if tactical_assessment is not None:
            params["tactical_assessment"] = tactical_assessment
        if recommendations is not None:
            params["recommendations"] = recommendations
        if methodology is not None:
            params["methodology"] = methodology

        # Invoke through SkillManager (dynamic routing)
        return manager.invoke_skill(skill_name, action, params)

    except Exception as e:
        return {
            "status": "error",
            "action": action,
            "result": None,
            "message": f"Error invoking skill {skill_name}/{action}: {str(e)}"
        }


# =============================================================================
# CONVENIENCE WRAPPER TOOLS
# These are optional shortcuts for common skill operations.
# The agent can always use invoke_skill() directly instead.
# =============================================================================

@tool
def videodb_query_skill(action: str, query: Optional[str] = None, object_type: Optional[str] = None,
                        video_ids: Optional[list] = None, segment_id: Optional[int] = None,
                        video_id: Optional[str] = None, top_k: int = 5, min_count: int = 1) -> dict:
    """
    Video database query skill for analyzing reconnaissance footage.
    Use this to search videos, find objects, and get segment details.

    Actions:
    - get_contexts: Get selected video/PDF contexts (CALL THIS FIRST!)
    - get_summary: Get overview of selected videos
    - semantic_search: Search by description (requires query)
    - object_search: Find objects by type (requires object_type: tank/truck/soldier)
    - event_search: Search by event description (requires query)
    - segment_details: Get details for a specific segment (requires segment_id)

    Args:
        action: The action to perform (see above)
        query: Search query for semantic_search or event_search
        object_type: Object type for object_search (tank, truck, soldier)
        video_ids: Optional list of video IDs to search
        segment_id: Segment ID for segment_details
        video_id: Single video ID for segment_details
        top_k: Number of results for search actions
        min_count: Minimum object count for object_search

    Returns:
        Dictionary with search results and status

    Example:
        # First get contexts
        contexts = videodb_query_skill("get_contexts")

        # Then search for tanks
        tanks = videodb_query_skill("object_search", object_type="tank")
    """
    return invoke_skill(
        skill_name="videodb_query",
        action=action,
        query=query,
        object_type=object_type,
        video_ids=video_ids,
        segment_id=segment_id,
        video_id=video_id,
        top_k=top_k,
        min_count=min_count
    )


@tool
def pdf_rag_skill(action: str, query: Optional[str] = None, pdf_sources: Optional[list] = None, top_k: int = 5) -> dict:
    """
    PDF RAG (Retrieval-Augmented Generation) skill for searching reference documents.
    Use this to find information in uploaded PDF manuals and guides.

    Actions:
    - search: Search PDF documents for relevant passages (requires query)

    Args:
        action: The action to perform (currently: "search")
        query: Search query describing what information you need
        pdf_sources: Optional list of PDF filenames to search (uses selected PDFs if not provided)
        top_k: Number of results to return

    Returns:
        Dictionary with search results including:
        - status: success/no_results/error
        - result: Contains matched text passages with relevance scores
        - message: Summary of results

    Example:
        # Search for tank specifications
        result = pdf_rag_skill("search", query="tank operational range capabilities")
    """
    return invoke_skill(
        skill_name="pdf_rag",
        action=action,
        query=query,
        pdf_sources=pdf_sources,
        top_k=top_k
    )


@tool
def wargame_query_skill(action: str, unit_name: Optional[str] = None,
                        unit_type: Optional[str] = None) -> dict:
    """
    Tactical map query skill for accessing war game state information.
    Use this to get information about friendly and hostile unit positions.

    Actions:
    - tactical_situation: Get overall tactical overview
    - friendly_units: List all friendly (blue) forces
    - hostile_units: List all hostile (red) forces
    - unit_details: Get details for a specific unit (requires unit_name)
    - unit_waypoints: Get movement waypoints for a unit (requires unit_name)
    - units_by_type: Find units by type (requires unit_type)

    Args:
        action: The action to perform (see above)
        unit_name: Name of specific unit for unit_details/unit_waypoints
        unit_type: Type of unit for units_by_type (INFANTRY, ARMOR, ARTILLERY, etc.)

    Returns:
        Dictionary with tactical information and status

    Example:
        # Get overall situation
        situation = wargame_query_skill("tactical_situation")

        # Get all armor units
        armor = wargame_query_skill("units_by_type", unit_type="ARMOR")
    """
    return invoke_skill(
        skill_name="wargame_query",
        action=action,
        unit_name=unit_name,
        unit_type=unit_type
    )


@tool
def final_response_skill(action: str, title: Optional[str] = None, summary: Optional[str] = None,
                        findings: Optional[list] = None,
                        video_evidence: Optional[list] = None,
                        pdf_references: Optional[list] = None,
                        tactical_assessment: Optional[str] = None,
                        recommendations: Optional[list] = None,
                        methodology: Optional[str] = None) -> dict:
    """
    Final response skill for generating structured battlefield analysis reports.
    Use this to format your final answer in a consistent, professional report format.

    Actions:
    - generate_report: Generate a structured analysis report (requires title, summary, findings)

    Args:
        action: The action to perform (currently: "generate_report")
        title: Report title (e.g., "Tank Activity Analysis Report")
        summary: Executive summary of the analysis (2-3 sentences)
        findings: List of key findings. Each can be a string or dict with:
            - text/description: Finding text
            - confidence: Confidence level
            - source: Source of finding
        video_evidence: List of video evidence items. Each can be a dict with:
            - video_id: Video identifier
            - segment_id: Segment number
            - time_range: Time range string
            - description: What was observed
            - objects: List of detected objects
        pdf_references: List of PDF reference items. Each can be a dict with:
            - source: PDF filename
            - text: Relevant text excerpt
            - relevance_score: Relevance score
        tactical_assessment: Overall tactical assessment paragraph
        recommendations: List of recommended actions
        methodology: Description of analysis methodology used

    Returns:
        Dictionary with:
        - status: success/error
        - result: Contains 'report' (formatted markdown) and 'sections' metadata
        - message: Summary of report contents

    Example:
        report = final_response_skill(
            "generate_report",
            title="Reconnaissance Analysis Report",
            summary="Analysis of drone footage reveals significant enemy activity...",
            findings=["3 tanks detected moving eastward", "Convoy formation suggests defensive posture"],
            video_evidence=[{"video_id": "m-abc123", "segment_id": 2, "description": "Tank formation"}],
            recommendations=["Monitor sector Alpha", "Prepare defensive positions"]
        )
    """
    return invoke_skill(
        skill_name="final_response",
        action=action,
        title=title,
        summary=summary,
        findings=findings,
        video_evidence=video_evidence,
        pdf_references=pdf_references,
        tactical_assessment=tactical_assessment,
        recommendations=recommendations,
        methodology=methodology
    )


