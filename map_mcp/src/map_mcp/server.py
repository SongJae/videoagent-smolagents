"""MCP Server for offline map operations.

This module implements an MCP server that provides tools for
creating and manipulating maps in an offline environment.
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

from .map_utils import OfflineMap


# Global map instance
_current_map: Optional[OfflineMap] = None
_tiles_dir: Optional[Path] = None


def get_current_map() -> OfflineMap:
    """Get or create the current map instance."""
    global _current_map
    if _current_map is None:
        _current_map = OfflineMap(tiles_dir=_tiles_dir)
    return _current_map


def set_tiles_directory(path: str | Path) -> None:
    """Set the tiles directory for offline maps."""
    global _tiles_dir
    _tiles_dir = Path(path)


# Create the MCP server
server = Server("map-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available map tools."""
    return [
        Tool(
            name="create_map",
            description="Create a new map centered at the specified location",
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
            name="add_marker",
            description="Add a marker to the map at the specified location",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Marker latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Marker longitude",
                    },
                    "popup": {
                        "type": "string",
                        "description": "Popup text/HTML content",
                    },
                    "tooltip": {
                        "type": "string",
                        "description": "Tooltip text shown on hover",
                    },
                    "color": {
                        "type": "string",
                        "description": "Marker color",
                        "default": "blue",
                    },
                    "icon": {
                        "type": "string",
                        "description": "Icon name (e.g., 'info-sign', 'star', 'home')",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        ),
        Tool(
            name="add_polyline",
            description="Draw a polyline (path) on the map connecting multiple points",
            inputSchema={
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "Array of [latitude, longitude] coordinate pairs",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "color": {
                        "type": "string",
                        "description": "Line color",
                        "default": "blue",
                    },
                    "weight": {
                        "type": "integer",
                        "description": "Line width in pixels",
                        "default": 3,
                    },
                    "opacity": {
                        "type": "number",
                        "description": "Line opacity (0-1)",
                        "default": 1.0,
                    },
                    "popup": {
                        "type": "string",
                        "description": "Popup text/HTML content",
                    },
                },
                "required": ["points"],
            },
        ),
        Tool(
            name="add_polygon",
            description="Draw a filled polygon on the map",
            inputSchema={
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "Array of [latitude, longitude] vertex coordinates",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "color": {
                        "type": "string",
                        "description": "Border color",
                        "default": "blue",
                    },
                    "fill_color": {
                        "type": "string",
                        "description": "Fill color",
                    },
                    "fill_opacity": {
                        "type": "number",
                        "description": "Fill opacity (0-1)",
                        "default": 0.3,
                    },
                    "popup": {
                        "type": "string",
                        "description": "Popup text/HTML content",
                    },
                },
                "required": ["points"],
            },
        ),
        Tool(
            name="add_circle",
            description="Draw a circle on the map centered at a point with a given radius",
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
                    "radius": {
                        "type": "number",
                        "description": "Radius in meters",
                    },
                    "color": {
                        "type": "string",
                        "description": "Circle color",
                        "default": "blue",
                    },
                    "fill_opacity": {
                        "type": "number",
                        "description": "Fill opacity (0-1)",
                        "default": 0.3,
                    },
                    "popup": {
                        "type": "string",
                        "description": "Popup text/HTML content",
                    },
                },
                "required": ["latitude", "longitude", "radius"],
            },
        ),
        Tool(
            name="add_geojson",
            description="Add a GeoJSON layer to the map from a file or data",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a GeoJSON file",
                    },
                    "data": {
                        "type": "object",
                        "description": "GeoJSON data object",
                    },
                    "style": {
                        "type": "object",
                        "description": "Style properties for the features",
                        "properties": {
                            "color": {"type": "string"},
                            "weight": {"type": "number"},
                            "fillColor": {"type": "string"},
                            "fillOpacity": {"type": "number"},
                        },
                    },
                    "popup_property": {
                        "type": "string",
                        "description": "Property name to use for popups",
                    },
                },
            },
        ),
        Tool(
            name="set_view",
            description="Set the map center and zoom level",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "New center latitude",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "New center longitude",
                    },
                    "zoom": {
                        "type": "integer",
                        "description": "New zoom level",
                    },
                },
            },
        ),
        Tool(
            name="clear_map",
            description="Clear all elements from the map",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_map_state",
            description="Get the current state of the map including all elements",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="render_map",
            description="Render the current map to HTML",
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    try:
        if name == "create_map":
            global _current_map
            _current_map = OfflineMap(
                center=(arguments["latitude"], arguments["longitude"]),
                zoom=arguments.get("zoom", 10),
                tiles_dir=_tiles_dir,
            )
            return [
                TextContent(
                    type="text",
                    text=f"Created map centered at ({arguments['latitude']}, {arguments['longitude']}) with zoom {arguments.get('zoom', 10)}",
                )
            ]

        elif name == "add_marker":
            m = get_current_map()
            m.add_marker(
                location=(arguments["latitude"], arguments["longitude"]),
                popup=arguments.get("popup"),
                tooltip=arguments.get("tooltip"),
                icon=arguments.get("icon"),
                color=arguments.get("color", "blue"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added marker at ({arguments['latitude']}, {arguments['longitude']})",
                )
            ]

        elif name == "add_polyline":
            m = get_current_map()
            points = [tuple(p) for p in arguments["points"]]
            m.add_polyline(
                locations=points,
                color=arguments.get("color", "blue"),
                weight=arguments.get("weight", 3),
                opacity=arguments.get("opacity", 1.0),
                popup=arguments.get("popup"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added polyline with {len(points)} points",
                )
            ]

        elif name == "add_polygon":
            m = get_current_map()
            points = [tuple(p) for p in arguments["points"]]
            m.add_polygon(
                locations=points,
                color=arguments.get("color", "blue"),
                fill_color=arguments.get("fill_color"),
                fill_opacity=arguments.get("fill_opacity", 0.3),
                popup=arguments.get("popup"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added polygon with {len(points)} vertices",
                )
            ]

        elif name == "add_circle":
            m = get_current_map()
            m.add_circle(
                location=(arguments["latitude"], arguments["longitude"]),
                radius=arguments["radius"],
                color=arguments.get("color", "blue"),
                fill_opacity=arguments.get("fill_opacity", 0.3),
                popup=arguments.get("popup"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Added circle at ({arguments['latitude']}, {arguments['longitude']}) with radius {arguments['radius']}m",
                )
            ]

        elif name == "add_geojson":
            m = get_current_map()
            data = arguments.get("data")
            if not data and arguments.get("file_path"):
                data = arguments["file_path"]
            m.add_geojson(
                data=data,
                style=arguments.get("style"),
                popup_property=arguments.get("popup_property"),
            )
            return [
                TextContent(
                    type="text",
                    text="Added GeoJSON layer to the map",
                )
            ]

        elif name == "set_view":
            m = get_current_map()
            if "latitude" in arguments and "longitude" in arguments:
                m.set_center(arguments["latitude"], arguments["longitude"])
            if "zoom" in arguments:
                m.set_zoom(arguments["zoom"])
            return [
                TextContent(
                    type="text",
                    text=f"Updated map view to center ({m.center[0]}, {m.center[1]}) zoom {m.zoom}",
                )
            ]

        elif name == "clear_map":
            m = get_current_map()
            m.clear()
            return [
                TextContent(
                    type="text",
                    text="Cleared all map elements",
                )
            ]

        elif name == "get_map_state":
            m = get_current_map()
            state = m.get_state()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(state, indent=2),
                )
            ]

        elif name == "render_map":
            m = get_current_map()
            html = m.to_html()
            if arguments.get("save_path"):
                m.save(arguments["save_path"])
                return [
                    TextContent(
                        type="text",
                        text=f"Map saved to {arguments['save_path']}",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=html,
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
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Map MCP Server")
    parser.add_argument(
        "--tiles-dir",
        type=str,
        help="Directory containing offline map tiles",
    )
    args = parser.parse_args()

    asyncio.run(run_server(args.tiles_dir))


if __name__ == "__main__":
    main()
