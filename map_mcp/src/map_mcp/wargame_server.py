"""MCP Server for war game operations.

This module implements an MCP server that provides tools for
military operations planning and war gaming on offline maps.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
)

from .military_symbols import Affiliation, Echelon, UnitType
from .wargame_map import WarGameMap


# Global war game map instance
_current_map: Optional[WarGameMap] = None
_tiles_dir: Optional[Path] = None


def get_current_map() -> WarGameMap:
    """Get or create the current war game map instance."""
    global _current_map
    if _current_map is None:
        _current_map = WarGameMap(tiles_dir=_tiles_dir)
    return _current_map


def set_tiles_directory(path: str | Path) -> None:
    """Set the tiles directory for offline maps."""
    global _tiles_dir
    _tiles_dir = Path(path)


# Create the MCP server
server = Server("wargame-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available war game tools."""
    # Get available unit types and echelons for schema
    unit_types = [e.name for e in UnitType]
    echelons = [e.name for e in Echelon]
    affiliations = ["FRIEND", "HOSTILE", "NEUTRAL", "UNKNOWN"]

    return [
        Tool(
            name="create_wargame_map",
            description="Create a new war game map centered at the specified location",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Center latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Center longitude",
                    },
                    "zoom": {
                        "type": "integer",
                        "description": "Initial zoom level (1-18)",
                        "default": 10,
                    },
                },
                "required": ["latitude", "longitude"],
            },
        ),
        Tool(
            name="add_military_unit",
            description="Add a military unit to the war game map with NATO APP-6/MIL-STD-2525 symbology",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unit name (e.g., '1st Platoon, Alpha Company')",
                    },
                    "affiliation": {
                        "type": "string",
                        "description": "Unit affiliation",
                        "enum": affiliations,
                    },
                    "unit_type": {
                        "type": "string",
                        "description": "Type of military unit",
                        "enum": unit_types,
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Unit location latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Unit location longitude",
                    },
                    "echelon": {
                        "type": "string",
                        "description": "Unit size/echelon",
                        "enum": echelons,
                        "default": "PLATOON",
                    },
                    "designation": {
                        "type": "string",
                        "description": "Unit designation code (e.g., '1-1-A')",
                    },
                    "strength": {
                        "type": "integer",
                        "description": "Personnel strength",
                    },
                    "direction": {
                        "type": "number",
                        "description": "Unit heading in degrees (0-360)",
                    },
                    "is_headquarters": {
                        "type": "boolean",
                        "description": "Whether this is a headquarters unit",
                        "default": False,
                    },
                },
                "required": ["name", "affiliation", "unit_type", "latitude", "longitude"],
            },
        ),
        Tool(
            name="add_friendly_unit",
            description="Add a friendly (blue) military unit to the map",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unit name",
                    },
                    "unit_type": {
                        "type": "string",
                        "description": "Type of military unit",
                        "enum": unit_types,
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Unit location latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Unit location longitude",
                    },
                    "echelon": {
                        "type": "string",
                        "description": "Unit size/echelon",
                        "enum": echelons,
                        "default": "PLATOON",
                    },
                    "designation": {
                        "type": "string",
                        "description": "Unit designation code",
                    },
                    "strength": {
                        "type": "integer",
                        "description": "Personnel strength",
                    },
                },
                "required": ["name", "unit_type", "latitude", "longitude"],
            },
        ),
        Tool(
            name="add_hostile_unit",
            description="Add a hostile (red) military unit to the map",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unit name",
                    },
                    "unit_type": {
                        "type": "string",
                        "description": "Type of military unit",
                        "enum": unit_types,
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Unit location latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Unit location longitude",
                    },
                    "echelon": {
                        "type": "string",
                        "description": "Unit size/echelon",
                        "enum": echelons,
                        "default": "PLATOON",
                    },
                    "designation": {
                        "type": "string",
                        "description": "Unit designation code",
                    },
                    "strength": {
                        "type": "integer",
                        "description": "Personnel strength",
                    },
                },
                "required": ["name", "unit_type", "latitude", "longitude"],
            },
        ),
        Tool(
            name="add_waypoint",
            description="Add a waypoint to a unit's movement plan",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID to add waypoint to",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Waypoint latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Waypoint longitude",
                    },
                },
                "required": ["unit_id", "latitude", "longitude"],
            },
        ),
        Tool(
            name="set_unit_waypoints",
            description="Set all waypoints for a unit's movement plan",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID",
                    },
                    "waypoints": {
                        "type": "array",
                        "description": "Array of [latitude, longitude] waypoint coordinates",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "required": ["unit_id", "waypoints"],
            },
        ),
        Tool(
            name="clear_unit_waypoints",
            description="Clear all waypoints for a unit",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="update_unit_location",
            description="Update a unit's current location",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "New latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "New longitude",
                    },
                },
                "required": ["unit_id", "latitude", "longitude"],
            },
        ),
        Tool(
            name="remove_unit",
            description="Remove a military unit from the map",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID to remove",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="get_unit_info",
            description="Get information about a specific unit",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "The unit ID",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="list_friendly_units",
            description="List all friendly units on the map",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_hostile_units",
            description="List all hostile units on the map",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_all_units",
            description="List all military units on the map",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="clear_all_units",
            description="Remove all military units from the map",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_wargame_state",
            description="Get the complete war game state including all units and their waypoints",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="render_wargame_map",
            description="Render the war game map to HTML",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save the HTML file",
                    },
                },
            },
        ),
        Tool(
            name="save_scenario",
            description="Save the current war game scenario to a JSON file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to save the scenario JSON file",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="load_scenario",
            description="Load a war game scenario from a JSON file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the scenario JSON file",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    try:
        if name == "create_wargame_map":
            global _current_map
            _current_map = WarGameMap(
                center=(arguments["latitude"], arguments["longitude"]),
                zoom=arguments.get("zoom", 10),
                tiles_dir=_tiles_dir,
            )
            return [
                TextContent(
                    type="text",
                    text=f"Created war game map centered at ({arguments['latitude']}, {arguments['longitude']}) with zoom {arguments.get('zoom', 10)}",
                )
            ]

        elif name == "add_military_unit":
            m = get_current_map()
            unit = m.add_military_unit(
                name=arguments["name"],
                affiliation=arguments["affiliation"],
                unit_type=arguments["unit_type"],
                location=(arguments["latitude"], arguments["longitude"]),
                echelon=arguments.get("echelon", "PLATOON"),
                designation=arguments.get("designation"),
                strength=arguments.get("strength"),
                direction=arguments.get("direction"),
                is_headquarters=arguments.get("is_headquarters", False),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added {unit.affiliation.name_display} unit '{unit.name}' (ID: {unit.id}) at ({arguments['latitude']}, {arguments['longitude']})\nSIDC: {unit.generate_sidc()}",
                )
            ]

        elif name == "add_friendly_unit":
            m = get_current_map()
            unit = m.add_friendly_unit(
                name=arguments["name"],
                unit_type=arguments["unit_type"],
                location=(arguments["latitude"], arguments["longitude"]),
                echelon=arguments.get("echelon", "PLATOON"),
                designation=arguments.get("designation"),
                strength=arguments.get("strength"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added friendly unit '{unit.name}' (ID: {unit.id}) at ({arguments['latitude']}, {arguments['longitude']})",
                )
            ]

        elif name == "add_hostile_unit":
            m = get_current_map()
            unit = m.add_hostile_unit(
                name=arguments["name"],
                unit_type=arguments["unit_type"],
                location=(arguments["latitude"], arguments["longitude"]),
                echelon=arguments.get("echelon", "PLATOON"),
                designation=arguments.get("designation"),
                strength=arguments.get("strength"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added hostile unit '{unit.name}' (ID: {unit.id}) at ({arguments['latitude']}, {arguments['longitude']})",
                )
            ]

        elif name == "add_waypoint":
            m = get_current_map()
            success = m.add_waypoint(
                arguments["unit_id"],
                (arguments["latitude"], arguments["longitude"]),
            )
            if success:
                unit = m.get_unit(arguments["unit_id"])
                return [
                    TextContent(
                        type="text",
                        text=f"Added waypoint to unit {arguments['unit_id']}. Total waypoints: {len(unit.waypoints)}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "set_unit_waypoints":
            m = get_current_map()
            waypoints = [tuple(wp) for wp in arguments["waypoints"]]
            success = m.set_waypoints(arguments["unit_id"], waypoints)
            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"Set {len(waypoints)} waypoints for unit {arguments['unit_id']}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "clear_unit_waypoints":
            m = get_current_map()
            success = m.clear_waypoints(arguments["unit_id"])
            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"Cleared waypoints for unit {arguments['unit_id']}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "update_unit_location":
            m = get_current_map()
            success = m.update_unit_location(
                arguments["unit_id"],
                (arguments["latitude"], arguments["longitude"]),
            )
            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"Updated unit {arguments['unit_id']} location to ({arguments['latitude']}, {arguments['longitude']})",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "remove_unit":
            m = get_current_map()
            success = m.remove_unit(arguments["unit_id"])
            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"Removed unit {arguments['unit_id']}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "get_unit_info":
            m = get_current_map()
            unit = m.get_unit(arguments["unit_id"])
            if unit:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(unit.to_dict(), indent=2),
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=f"Unit {arguments['unit_id']} not found",
                )
            ]

        elif name == "list_friendly_units":
            m = get_current_map()
            units = m.get_friendly_units()
            result = [u.to_dict() for u in units]
            return [
                TextContent(
                    type="text",
                    text=f"Friendly units ({len(units)}):\n{json.dumps(result, indent=2)}",
                )
            ]

        elif name == "list_hostile_units":
            m = get_current_map()
            units = m.get_hostile_units()
            result = [u.to_dict() for u in units]
            return [
                TextContent(
                    type="text",
                    text=f"Hostile units ({len(units)}):\n{json.dumps(result, indent=2)}",
                )
            ]

        elif name == "list_all_units":
            m = get_current_map()
            units = m.military_units
            result = [u.to_dict() for u in units]
            return [
                TextContent(
                    type="text",
                    text=f"All units ({len(units)}):\n{json.dumps(result, indent=2)}",
                )
            ]

        elif name == "clear_all_units":
            m = get_current_map()
            count = len(m.military_units)
            m.clear_all_units()
            return [
                TextContent(
                    type="text",
                    text=f"Cleared {count} military units from the map",
                )
            ]

        elif name == "get_wargame_state":
            m = get_current_map()
            state = m.get_state()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(state, indent=2),
                )
            ]

        elif name == "render_wargame_map":
            m = get_current_map()
            html = m.to_html()
            if arguments.get("save_path"):
                m.save(arguments["save_path"])
                return [
                    TextContent(
                        type="text",
                        text=f"War game map saved to {arguments['save_path']}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=html,
                )
            ]

        elif name == "save_scenario":
            m = get_current_map()
            state = m.get_state()
            file_path = Path(arguments["file_path"])
            if not file_path.suffix:
                file_path = file_path.with_suffix(".json")
            with open(file_path, "w") as f:
                json.dump(state, f, indent=2)
            return [
                TextContent(
                    type="text",
                    text=f"Scenario saved to {file_path} with {len(m.military_units)} units",
                )
            ]

        elif name == "load_scenario":
            m = get_current_map()
            file_path = Path(arguments["file_path"])
            with open(file_path, "r") as f:
                state = json.load(f)
            m.load_state(state)
            return [
                TextContent(
                    type="text",
                    text=f"Loaded scenario from {file_path} with {len(m.military_units)} units",
                )
            ]

        else:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def run_server(tiles_dir: Optional[str] = None) -> None:
    """Run the MCP server.

    Args:
        tiles_dir: Optional path to the tiles directory
    """
    if tiles_dir:
        set_tiles_directory(tiles_dir)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point for the War Game MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="War Game MCP Server")
    parser.add_argument(
        "--tiles-dir",
        type=str,
        help="Directory containing offline map tiles",
    )
    args = parser.parse_args()

    asyncio.run(run_server(args.tiles_dir))


if __name__ == "__main__":
    main()
