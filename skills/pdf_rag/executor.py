"""
PDF RAG Skill Executor

This module defines the execution logic for the pdf_rag skill.
It is auto-loaded by SkillManager and provides dynamic action routing.
"""

import json
from typing import Dict, Any

# Define available actions for this skill
ACTIONS = {
    "search": {
        "description": "Search PDF documents for relevant passages",
        "params": ["query", "pdf_sources", "top_k"]
    }
}

# Lazy import to avoid circular dependencies
_tools_loaded = False
_pdf_rag_search = None


def _load_tools():
    """Lazy load underlying tools."""
    global _tools_loaded, _pdf_rag_search

    if _tools_loaded:
        return

    from tools.pdf_rag_tool import pdf_rag_search
    _pdf_rag_search = pdf_rag_search
    _tools_loaded = True


def execute(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a pdf_rag action.

    Args:
        action: The action to perform
        params: Dictionary of parameters for the action

    Returns:
        Dictionary with status, action, result, and message
    """
    _load_tools()

    try:
        if action == "search":
            query = params.get("query", "")
            pdf_sources = params.get("pdf_sources")
            top_k = params.get("top_k", 5)

            # pdf_rag_search returns JSON string, parse it
            result_str = _pdf_rag_search(query=query, pdf_sources=pdf_sources, top_k=top_k)
            result = json.loads(result_str)

            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": f"Found {result.get('num_results', 0)} relevant passage(s)" if result.get("status") == "success" else result.get("message", "")
            }

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
            "message": f"Error executing pdf_rag/{action}: {str(e)}"
        }
