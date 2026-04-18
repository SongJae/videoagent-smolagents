"""
WargameQueryTool: Smolagents tools for querying tactical map state
Provides tools to access wargame_state.json for battlefield situational awareness
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Add smolagents to path
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

# Add map_mcp to path
map_mcp_path = parent_dir / "map_mcp" / "src"
sys.path.insert(0, str(map_mcp_path))

from smolagents import tool
from map_mcp.wargame_ui import _get_temp_dir


def _get_wargame_state_path() -> Path:
    """
    Get the path to wargame_state.json dynamically.
    Uses the same temp directory as the Gradio UI.

    Returns:
        Path to wargame_state.json
    """
    return _get_temp_dir() / "wargame_state.json"


def _load_wargame_state() -> Dict[str, Any]:
    """
    Load the current wargame state from JSON file.

    Returns:
        Dictionary containing the wargame state, or error dict if not found
    """
    try:
        wargame_state_path = _get_wargame_state_path()
        if not wargame_state_path.exists():
            return {
                "error": True,
                "message": f"No tactical map state found at {wargame_state_path}. The tactical map may not have been initialized yet."
            }

        with open(wargame_state_path, 'r') as f:
            state = json.load(f)

        # Validate that state is a dictionary
        if not isinstance(state, dict):
            return {
                "error": True,
                "message": f"Invalid tactical map state format: expected dictionary, got {type(state).__name__}. The tactical map may need to be reinitialized."
            }

        return state

    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Error parsing tactical map state: {str(e)}"
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Error loading tactical map state: {str(e)}"
        }


def _get_all_units(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all units from state (both protected and map-clicked).

    Args:
        state: Wargame state dictionary

    Returns:
        List of all military units
    """
    units = []

    # Get protected units (from Python state)
    if "military_units" in state:
        units.extend(state["military_units"])

    # Get map-clicked units (from JavaScript state)
    if "map_clicked_units" in state:
        units.extend(state["map_clicked_units"])

    return units


def _filter_units_by_affiliation(units: List[Dict[str, Any]], affiliation: str) -> List[Dict[str, Any]]:
    """
    Filter units by affiliation.

    Args:
        units: List of unit dictionaries
        affiliation: Affiliation to filter by (FRIEND, HOSTILE, NEUTRAL, UNKNOWN)

    Returns:
        Filtered list of units
    """
    return [u for u in units if u.get("affiliation", "").upper() == affiliation.upper()]


def _parse_location(location) -> Dict[str, float]:
    """
    Parse location from either list [lat, lon] or dict {"lat": ..., "lon": ...} format.

    Args:
        location: Location in either format

    Returns:
        Dictionary with lat and lon keys
    """
    if isinstance(location, list) and len(location) >= 2:
        return {"lat": location[0], "lon": location[1]}
    elif isinstance(location, dict):
        return {"lat": location.get("lat", 0), "lon": location.get("lon", 0)}
    else:
        return {"lat": 0, "lon": 0}


def _parse_waypoints(waypoints) -> List[Dict[str, float]]:
    """
    Parse waypoints from either list of lists [[lat, lon], ...] or list of dicts format.

    Args:
        waypoints: Waypoints in either format

    Returns:
        List of dictionaries with lat and lon keys
    """
    if not waypoints:
        return []

    result = []
    for wp in waypoints:
        if isinstance(wp, list) and len(wp) >= 2:
            result.append({"lat": wp[0], "lon": wp[1]})
        elif isinstance(wp, dict):
            result.append({"lat": wp.get("lat", 0), "lon": wp.get("lon", 0)})
    return result


def _format_unit_summary(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a unit dictionary for clean output.

    Args:
        unit: Raw unit dictionary

    Returns:
        Formatted unit summary
    """
    location = _parse_location(unit.get("location", {}))
    waypoints = unit.get("waypoints", [])

    return {
        "id": unit.get("id", "unknown"),
        "name": unit.get("name", "Unknown Unit"),
        "affiliation": unit.get("affiliation", "UNKNOWN"),
        "unit_type": unit.get("unit_type", "UNKNOWN"),
        "echelon": unit.get("echelon", "UNKNOWN"),
        "location": location,
        "has_waypoints": len(waypoints) > 0,
        "num_waypoints": len(waypoints),
        "protected": unit.get("protected", False)
    }


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool
def get_tactical_situation() -> dict:
    """
    Get the overall tactical situation from the war game map.
    Provides a summary of all friendly and hostile units on the tactical map.

    Returns:
        Dictionary with tactical situation overview:
        - status: "success" or "error"
        - map_center: Center coordinates of the map (lat, lon)
        - zoom_level: Current map zoom level
        - total_units: Total number of units on map
        - friendly_count: Number of friendly units
        - hostile_count: Number of hostile units
        - neutral_count: Number of neutral units
        - unknown_count: Number of unknown affiliation units
        - summary: Text summary of the tactical situation
        - message: Error message if applicable

    Example:
        situation = get_tactical_situation()
        print(f"Friendly: {situation['friendly_count']}, Hostile: {situation['hostile_count']}")
    """
    state = _load_wargame_state()

    if state.get("error"):
        return {
            "status": "error",
            "message": state.get("message", "Unknown error loading tactical map state")
        }

    # Get all units
    all_units = _get_all_units(state)

    # Count by affiliation
    friendly_units = _filter_units_by_affiliation(all_units, "FRIEND")
    hostile_units = _filter_units_by_affiliation(all_units, "HOSTILE")
    neutral_units = _filter_units_by_affiliation(all_units, "NEUTRAL")
    unknown_units = _filter_units_by_affiliation(all_units, "UNKNOWN")

    # Build summary text
    summary_parts = []
    if friendly_units:
        friendly_types = {}
        for u in friendly_units:
            unit_type = u.get("unit_type", "UNKNOWN")
            friendly_types[unit_type] = friendly_types.get(unit_type, 0) + 1
        types_str = ", ".join([f"{count} {utype}" for utype, count in friendly_types.items()])
        summary_parts.append(f"Friendly forces: {types_str}")

    if hostile_units:
        hostile_types = {}
        for u in hostile_units:
            unit_type = u.get("unit_type", "UNKNOWN")
            hostile_types[unit_type] = hostile_types.get(unit_type, 0) + 1
        types_str = ", ".join([f"{count} {utype}" for utype, count in hostile_types.items()])
        summary_parts.append(f"Hostile forces: {types_str}")

    if not summary_parts:
        summary_parts.append("No military units currently on the tactical map")

    # Get map center (handle both list [lat, lon] and dict {"lat": ..., "lon": ...} formats)
    center = _parse_location(state.get("center", {}))

    return {
        "status": "success",
        "map_center": center,
        "zoom_level": state.get("zoom", 10),
        "total_units": len(all_units),
        "friendly_count": len(friendly_units),
        "hostile_count": len(hostile_units),
        "neutral_count": len(neutral_units),
        "unknown_count": len(unknown_units),
        "summary": ". ".join(summary_parts)
    }


@tool
def get_friendly_units() -> dict:
    """
    Get all friendly (blue force) military units from the tactical map.
    This tool returns ONLY units with affiliation='FRIEND' (our forces).
    For enemy units, use get_hostile_units() instead.

    Returns:
        Dictionary with friendly unit information:
        - status: "success", "no_units", or "error"
        - result_type: "FRIENDLY_UNITS" - indicates these are all friendly forces
        - count: Number of friendly units
        - units: List of friendly units, each containing:
            - id: Unit identifier
            - name: Unit name/designation (friendly units typically named F-1, F-2, etc.)
            - unit_type: Type of unit (INFANTRY, ARMOR, ARTILLERY, etc.)
            - echelon: Unit size (TEAM, SQUAD, PLATOON, COMPANY, etc.)
            - location: {lat, lon} coordinates
            - has_waypoints: Whether unit has movement waypoints
            - num_waypoints: Number of waypoints
        - message: Status or error message

    Example:
        result = get_friendly_units()
        for unit in result["units"]:
            print(f"FRIENDLY: {unit['name']} ({unit['unit_type']}) at {unit['location']}")
    """
    state = _load_wargame_state()

    if state.get("error"):
        return {
            "status": "error",
            "message": state.get("message", "Unknown error loading tactical map state")
        }

    # Get all units and filter by affiliation
    all_units = _get_all_units(state)
    friendly_units = _filter_units_by_affiliation(all_units, "FRIEND")

    if not friendly_units:
        return {
            "status": "no_units",
            "result_type": "FRIENDLY_UNITS",
            "count": 0,
            "units": [],
            "message": "No friendly units (blue force) on the tactical map"
        }

    # Format unit summaries
    formatted_units = [_format_unit_summary(u) for u in friendly_units]

    return {
        "status": "success",
        "result_type": "FRIENDLY_UNITS",
        "count": len(formatted_units),
        "units": formatted_units,
        "message": f"FRIENDLY FORCES: Found {len(formatted_units)} friendly unit(s) (blue force) on the tactical map"
    }


@tool
def get_hostile_units() -> dict:
    """
    Get all hostile/enemy (red force) military units from the tactical map.
    This tool returns ONLY units with affiliation='HOSTILE' (enemy forces).
    For our units, use get_friendly_units() instead.

    Returns:
        Dictionary with hostile unit information:
        - status: "success", "no_units", or "error"
        - result_type: "HOSTILE_UNITS" - indicates these are all enemy forces
        - count: Number of hostile units
        - units: List of hostile units, each containing:
            - id: Unit identifier
            - name: Unit name/designation (hostile units typically named H-1, H-2, etc.)
            - unit_type: Type of unit (INFANTRY, ARMOR, ARTILLERY, etc.)
            - echelon: Unit size (TEAM, SQUAD, PLATOON, COMPANY, etc.)
            - location: {lat, lon} coordinates
            - has_waypoints: Whether unit has movement waypoints
            - num_waypoints: Number of waypoints
        - message: Status or error message

    Example:
        result = get_hostile_units()
        for unit in result["units"]:
            print(f"HOSTILE: Enemy {unit['unit_type']} at lat:{unit['location']['lat']}, lon:{unit['location']['lon']}")
    """
    state = _load_wargame_state()

    if state.get("error"):
        return {
            "status": "error",
            "message": state.get("message", "Unknown error loading tactical map state")
        }

    # Get all units and filter by affiliation
    all_units = _get_all_units(state)
    hostile_units = _filter_units_by_affiliation(all_units, "HOSTILE")

    if not hostile_units:
        return {
            "status": "no_units",
            "result_type": "HOSTILE_UNITS",
            "count": 0,
            "units": [],
            "message": "No hostile units (red/enemy force) on the tactical map"
        }

    # Format unit summaries
    formatted_units = [_format_unit_summary(u) for u in hostile_units]

    return {
        "status": "success",
        "result_type": "HOSTILE_UNITS",
        "count": len(formatted_units),
        "units": formatted_units,
        "message": f"HOSTILE FORCES: Found {len(formatted_units)} hostile unit(s) (red/enemy force) on the tactical map"
    }


@tool
def get_unit_details(unit_name: str) -> dict:
    """
    Get detailed information about a specific military unit by name.

    Args:
        unit_name: Name or partial name of the unit to search for (case-insensitive)

    Returns:
        Dictionary with unit details:
        - status: "success", "not_found", or "error"
        - unit: Full unit information if found:
            - id: Unit identifier
            - name: Unit name/designation
            - affiliation: FRIEND, HOSTILE, NEUTRAL, or UNKNOWN
            - unit_type: Type of unit (INFANTRY, ARMOR, ARTILLERY, etc.)
            - echelon: Unit size
            - sidc: NATO Symbol Identification Code
            - location: {lat, lon} coordinates
            - waypoints: List of movement waypoints with {lat, lon} coordinates
            - protected: Whether unit was created via settings (vs map click)
        - message: Status or error message

    Example:
        result = get_unit_details("Alpha Company")
        if result["status"] == "success":
            unit = result["unit"]
            print(f"Unit {unit['name']} is at {unit['location']}")
            print(f"Movement plan: {len(unit['waypoints'])} waypoints")
    """
    state = _load_wargame_state()

    if state.get("error"):
        return {
            "status": "error",
            "message": state.get("message", "Unknown error loading tactical map state")
        }

    # Get all units
    all_units = _get_all_units(state)

    if not all_units:
        return {
            "status": "not_found",
            "message": "No units on the tactical map"
        }

    # Search for unit by name (case-insensitive, partial match)
    search_name = unit_name.lower()
    matching_units = [u for u in all_units if search_name in u.get("name", "").lower()]

    if not matching_units:
        # List available unit names for user
        available_names = [u.get("name", "Unknown") for u in all_units]
        return {
            "status": "not_found",
            "message": f"Unit '{unit_name}' not found. Available units: {available_names}"
        }

    # Return first match (or exact match if available)
    unit = matching_units[0]
    for u in matching_units:
        if u.get("name", "").lower() == search_name:
            unit = u
            break

    # Format waypoints (handle both list of lists and list of dicts formats)
    waypoints = _parse_waypoints(unit.get("waypoints", []))

    # Format location (handle both list [lat, lon] and dict formats)
    location = _parse_location(unit.get("location", {}))

    return {
        "status": "success",
        "unit": {
            "id": unit.get("id", "unknown"),
            "name": unit.get("name", "Unknown Unit"),
            "affiliation": unit.get("affiliation", "UNKNOWN"),
            "unit_type": unit.get("unit_type", "UNKNOWN"),
            "echelon": unit.get("echelon", "UNKNOWN"),
            "sidc": unit.get("sidc", ""),
            "location": location,
            "waypoints": waypoints,
            "protected": unit.get("protected", False)
        },
        "message": f"Found unit: {unit.get('name', 'Unknown')}"
    }


@tool
def get_unit_waypoints(unit_name: str) -> dict:
    """
    Get the movement waypoints (planned route) for a specific military unit.

    Args:
        unit_name: Name or partial name of the unit (case-insensitive)

    Returns:
        Dictionary with waypoint information:
        - status: "success", "no_waypoints", "not_found", or "error"
        - unit_name: Name of the unit
        - unit_location: Current location {lat, lon}
        - waypoints: List of waypoint coordinates in order [{lat, lon}, ...]
        - num_waypoints: Number of waypoints
        - message: Status or error message

    Example:
        result = get_unit_waypoints("Bravo Platoon")
        if result["status"] == "success":
            print(f"Unit starts at {result['unit_location']}")
            for i, wp in enumerate(result["waypoints"]):
                print(f"Waypoint {i+1}: lat={wp['lat']}, lon={wp['lon']}")
    """
    # First get unit details
    unit_result = get_unit_details(unit_name)

    if unit_result["status"] == "error":
        return unit_result

    if unit_result["status"] == "not_found":
        return {
            "status": "not_found",
            "message": unit_result["message"]
        }

    unit = unit_result["unit"]
    waypoints = unit.get("waypoints", [])

    if not waypoints:
        return {
            "status": "no_waypoints",
            "unit_name": unit["name"],
            "unit_location": unit["location"],
            "waypoints": [],
            "num_waypoints": 0,
            "message": f"Unit '{unit['name']}' has no movement waypoints defined"
        }

    return {
        "status": "success",
        "unit_name": unit["name"],
        "unit_location": unit["location"],
        "waypoints": waypoints,
        "num_waypoints": len(waypoints),
        "message": f"Unit '{unit['name']}' has {len(waypoints)} waypoint(s) in its movement plan"
    }


@tool
def get_units_by_type(unit_type: str) -> dict:
    """
    Get all military units of a specific type from the tactical map.

    Args:
        unit_type: Type of unit to search for. Valid types include:
            INFANTRY, ARMOR, ARTILLERY, AIR_DEFENSE, AVIATION, ENGINEER,
            RECONNAISSANCE, SIGNAL, MEDICAL, SUPPLY, HEADQUARTERS, MECHANIZED,
            MOTORIZED, AIRBORNE, SPECIAL_FORCES, NAVY, CYBER, ELECTRONIC_WARFARE, UAV

    Returns:
        Dictionary with units of the specified type:
        - status: "success", "no_units", or "error"
        - unit_type: The type searched for
        - count: Number of units found
        - friendly_count: Number of friendly units of this type
        - hostile_count: Number of hostile units of this type
        - units: List of matching units with:
            - id, name, affiliation, unit_type, echelon, location, has_waypoints
        - message: Status or error message

    Example:
        result = get_units_by_type("ARMOR")
        print(f"Found {result['friendly_count']} friendly and {result['hostile_count']} hostile armor units")
    """
    state = _load_wargame_state()

    if state.get("error"):
        return {
            "status": "error",
            "message": state.get("message", "Unknown error loading tactical map state")
        }

    # Get all units
    all_units = _get_all_units(state)

    # Filter by unit type (case-insensitive)
    search_type = unit_type.upper()
    matching_units = [u for u in all_units if u.get("unit_type", "").upper() == search_type]

    if not matching_units:
        # List available unit types for user
        available_types = list(set(u.get("unit_type", "UNKNOWN") for u in all_units))
        return {
            "status": "no_units",
            "unit_type": search_type,
            "count": 0,
            "units": [],
            "message": f"No units of type '{search_type}' found. Available types on map: {available_types}"
        }

    # Count by affiliation
    friendly_units = [u for u in matching_units if u.get("affiliation", "").upper() == "FRIEND"]
    hostile_units = [u for u in matching_units if u.get("affiliation", "").upper() == "HOSTILE"]

    # Format unit summaries
    formatted_units = [_format_unit_summary(u) for u in matching_units]

    return {
        "status": "success",
        "unit_type": search_type,
        "count": len(formatted_units),
        "friendly_count": len(friendly_units),
        "hostile_count": len(hostile_units),
        "units": formatted_units,
        "message": f"Found {len(formatted_units)} {search_type} unit(s): {len(friendly_units)} friendly, {len(hostile_units)} hostile"
    }
