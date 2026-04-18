# Offline Map MCP Server

A Gradio web UI for rendering and interacting with maps in an offline environment using MCP (Model Context Protocol).

## Features

- **Fully Offline Operation**: Works without internet connectivity using pre-downloaded map tiles
- **Interactive Map UI**: Gradio-based web interface for map manipulation
- **MCP Integration**: Provides map tools via MCP for AI assistant integration
- **Multiple Map Elements**: Support for markers, polylines, polygons, circles, and GeoJSON layers
- **Tile Pre-download**: Utility to download map tiles while online for offline use

## Installation

```bash
# Clone or navigate to the project directory
cd map_mcp

# Install the package
pip install -e .

# For tile downloading capabilities (requires internet)
pip install -e ".[download]"
```

## Preparing for Offline Use

Before going offline, download map tiles for your area of interest:

```bash
# Download tiles around Seoul, Korea (10km radius, zoom levels 1-15)
map-mcp-download --lat 37.5665 --lon 126.9780 --radius 10 --output ./tiles

# Download tiles for a specific bounding box
map-mcp-download --bbox 37.4,37.7,126.8,127.2 --zoom-min 10 --zoom-max 16 --output ./tiles

# Use a different tile server (options: osm, carto-light, carto-dark, stamen-terrain, stamen-toner)
map-mcp-download --lat 37.5665 --lon 126.9780 --server carto-light --output ./tiles
```

## Usage

### Gradio Web UI

Launch the interactive map viewer:

```bash
# Basic usage (no tiles - map will show overlay elements only)
map-mcp-ui

# With offline tiles
map-mcp-ui --tiles-dir ./tiles

# Custom host and port
map-mcp-ui --tiles-dir ./tiles --host 0.0.0.0 --port 8080
```

### MCP Server

Run the MCP server for AI assistant integration:

```bash
# Start the MCP server
map-mcp-server --tiles-dir ./tiles
```

### MCP Server Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "map": {
      "command": "map-mcp-server",
      "args": ["--tiles-dir", "/path/to/tiles"]
    }
  }
}
```

## MCP Tools

The MCP server provides the following tools:

| Tool | Description |
|------|-------------|
| `create_map` | Create a new map centered at a location |
| `add_marker` | Add a marker with popup and tooltip |
| `add_polyline` | Draw a line connecting multiple points |
| `add_polygon` | Draw a filled polygon |
| `add_circle` | Draw a circle with specified radius |
| `add_geojson` | Add a GeoJSON layer |
| `set_view` | Change map center and zoom |
| `clear_map` | Remove all map elements |
| `get_map_state` | Get current map state as JSON |
| `render_map` | Render map to HTML |

## Project Structure

```
map_mcp/
├── pyproject.toml           # Project configuration
├── README.md                # This file
├── tiles/                   # Downloaded map tiles (git-ignored)
├── data/                    # Sample data files
│   └── sample.geojson       # Example GeoJSON file
└── src/map_mcp/
    ├── __init__.py          # Package init
    ├── server.py            # MCP server implementation
    ├── gradio_ui.py         # Gradio web interface
    ├── map_utils.py         # Map rendering (PIL-based, fully offline)
    └── tile_downloader.py   # Tile download utility
```

## Offline Tile Storage

Tiles are stored in a standard slippy map format:

```
tiles/
├── 10/
│   ├── 873/
│   │   ├── 395.png
│   │   └── 396.png
│   └── 874/
│       └── 395.png
├── 11/
...
```

## Tips for Offline Use

1. **Estimate tile count**: Higher zoom levels require exponentially more tiles. Zoom 15 for a 10km radius is typically sufficient for city-level detail.

2. **Tile server selection**: OpenStreetMap (`osm`) is the default. Consider `carto-light` for a cleaner look.

3. **Storage requirements**: Approximately 10-50KB per tile. A 10km radius at zoom 1-15 might require 50-100MB.

4. **GeoJSON overlays**: Even without base map tiles, you can display GeoJSON data, markers, and shapes.

## License

MIT License
