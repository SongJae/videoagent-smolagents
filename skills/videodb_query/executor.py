"""
VideoDB Query Skill Executor

This module defines the execution logic for the videodb_query skill.
It is auto-loaded by SkillManager and provides dynamic action routing.
"""

from typing import Dict, Any, Optional, List

# Define available actions for this skill
ACTIONS = {
    "get_contexts": {
        "description": "Get selected video/PDF contexts (CALL THIS FIRST!)",
        "params": []
    },
    "get_summary": {
        "description": "Get overview of selected videos",
        "params": ["video_ids"]
    },
    "semantic_search": {
        "description": "Search by description",
        "params": ["query", "video_ids", "top_k"]
    },
    "object_search": {
        "description": "Find objects by type (tank/truck/soldier)",
        "params": ["object_type", "video_ids", "min_count"]
    },
    "event_search": {
        "description": "Search by event description",
        "params": ["query", "video_ids", "top_k"]
    },
    "segment_details": {
        "description": "Get details for a specific segment",
        "params": ["segment_id", "video_id"]
    }
}

# Lazy import to avoid circular dependencies
_tools_loaded = False
_get_selected_contexts = None
_get_video_summary = None
_query_video_semantic = None
_query_video_by_object = None
_query_video_by_event = None
_get_segment_details = None


def _load_tools():
    """Lazy load underlying tools."""
    global _tools_loaded
    global _get_selected_contexts, _get_video_summary, _query_video_semantic
    global _query_video_by_object, _query_video_by_event, _get_segment_details

    if _tools_loaded:
        return

    from tools.videodb_query_tool import (
        get_selected_contexts,
        get_video_summary,
        query_video_semantic,
        query_video_by_object,
        query_video_by_event,
        get_segment_details,
    )

    _get_selected_contexts = get_selected_contexts
    _get_video_summary = get_video_summary
    _query_video_semantic = query_video_semantic
    _query_video_by_object = query_video_by_object
    _query_video_by_event = query_video_by_event
    _get_segment_details = get_segment_details
    _tools_loaded = True


def execute(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a videodb_query action.

    Args:
        action: The action to perform
        params: Dictionary of parameters for the action

    Returns:
        Dictionary with status, action, result, and message
    """
    _load_tools()

    try:
        if action == "get_contexts":
            result = _get_selected_contexts()
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
            }

        elif action == "get_summary":
            video_ids = params.get("video_ids")
            result = _get_video_summary(video_ids=video_ids)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": f"Retrieved summary for {result.get('num_videos', 0)} video(s)" if result.get("status") == "success" else result.get("message", "")
            }

        elif action == "semantic_search":
            query = params.get("query", "")
            video_ids = params.get("video_ids")
            top_k = params.get("top_k", 5)
            result = _query_video_semantic(query=query, video_ids=video_ids, top_k=top_k)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": f"Found {result.get('num_results', 0)} matching segment(s)" if result.get("status") == "success" else result.get("message", "")
            }

        elif action == "object_search":
            object_type = params.get("object_type", "")
            video_ids = params.get("video_ids")
            min_count = params.get("min_count", 1)
            result = _query_video_by_object(object_type=object_type, video_ids=video_ids, min_count=min_count)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("summary_text", "") if result.get("status") == "success" else result.get("message", "")
            }

        elif action == "event_search":
            event_query = params.get("query", params.get("event_query", ""))
            video_ids = params.get("video_ids")
            top_k = params.get("top_k", 5)
            result = _query_video_by_event(event_query=event_query, video_ids=video_ids, top_k=top_k)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": f"Found {result.get('num_results', 0)} matching segment(s)" if result.get("status") == "success" else result.get("message", "")
            }

        elif action == "segment_details":
            segment_id = params.get("segment_id")
            video_id = params.get("video_id")
            if segment_id is None:
                return {
                    "status": "error",
                    "action": action,
                    "result": None,
                    "message": "segment_id is required for segment_details action"
                }
            result = _get_segment_details(segment_id=segment_id, video_id=video_id)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": f"Retrieved details for segment {segment_id}" if result.get("status") == "success" else result.get("message", "")
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
            "message": f"Error executing videodb_query/{action}: {str(e)}"
        }
