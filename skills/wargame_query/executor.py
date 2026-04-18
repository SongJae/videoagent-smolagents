"""
Wargame Query Skill Executor

This module defines the execution logic for the wargame_query skill.
It is auto-loaded by SkillManager and provides dynamic action routing.
"""

from typing import Dict, Any

# Define available actions for this skill
ACTIONS = {
    "tactical_situation": {
        "description": "Get overall tactical overview",
        "params": []
    },
    "friendly_units": {
        "description": "List all friendly (blue) forces",
        "params": []
    },
    "hostile_units": {
        "description": "List all hostile (red) forces",
        "params": []
    },
    "unit_details": {
        "description": "Get details for a specific unit",
        "params": ["unit_name"]
    },
    "unit_waypoints": {
        "description": "Get movement waypoints for a unit",
        "params": ["unit_name"]
    },
    "units_by_type": {
        "description": "Find units by type",
        "params": ["unit_type"]
    }
}

# Lazy import to avoid circular dependencies
_tools_loaded = False
_get_tactical_situation = None
_get_friendly_units = None
_get_hostile_units = None
_get_unit_details = None
_get_unit_waypoints = None
_get_units_by_type = None


def _load_tools():
    """Lazy load underlying tools."""
    global _tools_loaded
    global _get_tactical_situation, _get_friendly_units, _get_hostile_units
    global _get_unit_details, _get_unit_waypoints, _get_units_by_type

    if _tools_loaded:
        return

    from tools.wargame_query_tool import (
        get_tactical_situation,
        get_friendly_units,
        get_hostile_units,
        get_unit_details,
        get_unit_waypoints,
        get_units_by_type,
    )

    _get_tactical_situation = get_tactical_situation
    _get_friendly_units = get_friendly_units
    _get_hostile_units = get_hostile_units
    _get_unit_details = get_unit_details
    _get_unit_waypoints = get_unit_waypoints
    _get_units_by_type = get_units_by_type
    _tools_loaded = True


def execute(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a wargame_query action.

    Args:
        action: The action to perform
        params: Dictionary of parameters for the action

    Returns:
        Dictionary with status, action, result, and message
    """
    _load_tools()

    try:
        if action == "tactical_situation":
            result = _get_tactical_situation()
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("summary", "") if result.get("status") == "success" else result.get("message", "")
            }

        elif action == "friendly_units":
            result = _get_friendly_units()
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
            }

        elif action == "hostile_units":
            result = _get_hostile_units()
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
            }

        elif action == "unit_details":
            unit_name = params.get("unit_name", "")
            result = _get_unit_details(unit_name=unit_name)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
            }

        elif action == "unit_waypoints":
            unit_name = params.get("unit_name", "")
            result = _get_unit_waypoints(unit_name=unit_name)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
            }

        elif action == "units_by_type":
            unit_type = params.get("unit_type", "")
            result = _get_units_by_type(unit_type=unit_type)
            return {
                "status": result.get("status", "error"),
                "action": action,
                "result": result,
                "message": result.get("message", "")
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
            "message": f"Error executing wargame_query/{action}: {str(e)}"
        }
