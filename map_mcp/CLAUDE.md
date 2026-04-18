# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package (editable mode)
pip install -e .

# Install with tile download support
pip install -e ".[download]"

# Install with dev dependencies
pip install -e ".[dev]"

# Run Gradio UI
map-mcp-ui --tiles-dir ./tiles

# Run MCP server
map-mcp-server --tiles-dir ./tiles

# Run War Game UI
wargame-ui --tiles-dir ./tiles

# Run War Game MCP server
wargame-mcp-server --tiles-dir ./tiles

# Download tiles for offline use (requires internet)
# Standard (slow, 1 req/sec - respects rate limits)
map-mcp-download --lat 37.5665 --lon 126.9780 --radius 10 --output ./tiles

# Fast mode (recommended) - uses async with 50 concurrent connections
map-mcp-download --lat 37.5665 --lon 126.9780 --radius 100 --output ./tiles --fast

# Fast mode with custom settings
map-mcp-download --lat 35.0 --lon 125.0 --radius 1500 --output ./tiles --fast --concurrent 50 --rate 30

# Run tests
pytest

# Run a single test file
pytest tests/test_wargame.py -v

# Run a specific test
pytest tests/test_wargame.py::TestWarGameMap::test_add_friendly_unit -v
```

## Architecture

This project provides offline map rendering via MCP with a Gradio web UI. The key constraint is **no internet connectivity** for map tile fetching at runtime.

### Core Components

1. **map_utils.py** - `OfflineMap` class generates interactive HTML using embedded Leaflet.js:
   - Loads pre-downloaded tiles from disk and embeds them as base64 in the HTML
   - Creates a custom `OfflineTileLayer` that serves embedded tiles or generates placeholder tiles via canvas
   - Supports interactive zoom, pan, click-to-add markers, and waypoint creation
   - The `to_html()` method returns a full HTML document with Leaflet.js

2. **gradio_ui.py** - `MapUI` class wraps `OfflineMap` and provides Gradio event handlers:
   - Uses iframe with data URI to embed the interactive map HTML
   - Provides form-based controls for adding markers, polylines, polygons, circles, and GeoJSON
   - The iframe approach ensures proper isolation and full Leaflet functionality

3. **server.py** - MCP server using `mcp.server.Server`:
   - Maintains a global `_current_map` instance
   - Tool handlers modify this shared map state
   - Uses stdio transport for MCP communication

4. **tile_downloader.py** - Tile downloading utilities:
   - `TileDownloader` class: Sync downloader with rate limiting (1 req/sec default)
   - `AsyncTileDownloader` class: Fast async downloader using aiohttp (~50x faster)
     - Uses multiple subdomains (a/b/c.tile.openstreetmap.org) for parallel downloads
     - Connection pooling with `TCPConnector` for TCP reuse
     - Semaphore-based concurrency control (default: 50 concurrent)
   - Both include retry logic with exponential backoff for 429 errors

### War Game Components

5. **military_symbols.py** - Military symbol utilities:
   - `Affiliation` enum (FRIEND, HOSTILE, NEUTRAL, UNKNOWN) with colors and display names
   - `UnitType` enum (INFANTRY, ARMOR, ARTILLERY, etc. - 25+ types)
   - `Echelon` enum (TEAM through ARMY) with MIL-STD-2525 codes
   - `MilitaryUnit` dataclass with location, waypoints, and SIDC generation
   - `generate_sidc()` / `parse_sidc()` functions for NATO symbol codes
   - JavaScript integration code for milsymbol.js + Leaflet

6. **wargame_map.py** - `WarGameMap` class extends `OfflineMap`:
   - `add_friendly_unit()` / `add_hostile_unit()` convenience methods
   - Waypoint management: `add_waypoint()`, `set_waypoints()`, `clear_waypoints()`
   - Unit tracking by affiliation with `get_friendly_units()` / `get_hostile_units()`
   - Embeds bundled milsymbol.min.js for offline NATO symbol rendering

7. **wargame_ui.py** - `WarGameUI` class for Gradio war game interface:
   - Unit placement with type/echelon selection
   - Click-to-add units and waypoints on map
   - Scenario save/load (JSON format)
   - **Uses file serving instead of data URI** for iframe (see below)

8. **wargame_server.py** - MCP server with 18 war game tools:
   - Unit CRUD: `add_military_unit`, `add_friendly_unit`, `add_hostile_unit`, `remove_unit`
   - Waypoints: `add_waypoint`, `set_unit_waypoints`, `clear_unit_waypoints`
   - Queries: `list_friendly_units`, `list_hostile_units`, `get_unit_info`
   - Scenario: `save_scenario`, `load_scenario`

### Data Flow

**Standard Map UI (gradio_ui.py):**
```
User clicks on Gradio UI
        ↓
gradio_ui.py creates/updates OfflineMap
        ↓
map_utils.py generates HTML with:
  - Leaflet.js/CSS embedded inline (from bundled assets/)
  - Pre-loaded tiles as base64 JSON object
  - Custom OfflineTileLayer that reads from embedded data
  - Interactive JavaScript for markers/waypoints
        ↓
HTML embedded in iframe (data URI) displayed in Gradio
```

**War Game UI (wargame_ui.py):**
```
User clicks on War Game UI
        ↓
wargame_ui.py creates/updates WarGameMap
        ↓
wargame_map.py generates HTML with:
  - Leaflet.js/CSS embedded inline
  - milsymbol.js loaded via external URL (not embedded)
  - Pre-loaded tiles as base64 JSON object
  - Military unit rendering via milsymbol.js
        ↓
HTML saved to temp file in /tmp/map_mcp/
        ↓
iframe loads HTML via Gradio file serving (/gradio_api/file=...)
```

### Why File Serving for War Game UI

The war game UI uses Gradio's static file serving instead of data URI because:
1. **Size**: milsymbol.js is ~850KB; embedding it makes HTML ~1MB, which as base64 becomes 1.33MB
2. **Browser limits**: Data URIs with large content cause browser rendering issues
3. **CORS**: Data URIs have `null` origin, blocking external script loads

Solution: Save HTML to temp file, serve via `gr.set_static_paths()`, load milsymbol.js as external script from `/gradio_api/file=.../milsymbol.min.js`

### Offline Tile Strategy

1. **Pre-download phase** (requires internet): Use `tile_downloader.py` to download tiles
2. **Runtime** (fully offline):
   - `OfflineMap._generate_visible_tiles_json()` reads tiles from disk
   - Tiles are embedded as base64 in a JavaScript object
   - Custom `OfflineTileLayer` serves embedded tiles
   - Missing tiles show placeholder with coordinate grid (generated via canvas)

### Military Symbol (SIDC) Format

SIDC codes are 20-character identifiers for NATO military symbols (MIL-STD-2525D/E):
- Positions 0-1: Version ("10" for 2525D)
- Position 3: Affiliation (3=Friend, 6=Hostile, 4=Neutral, 0=Unknown)
- Positions 4-5: Symbol set ("10" for land unit)
- Positions 8-9: Echelon code
- Positions 10-15: Function ID (unit type)

### Bundled Assets (src/map_mcp/assets/)

- `leaflet.min.js` / `leaflet.min.css` - Leaflet mapping library
- `milsymbol.min.js` - NATO military symbol library (built from milsymbol/ submodule)

### Interactive Features

**Standard Map:**
- Mouse scroll: Zoom in/out
- Click & drag: Pan the map
- Add Marker mode: Click on map to place red markers
- Add Waypoint mode: Click to create connected green waypoints with polyline
- Clear All: Remove all user-added markers and waypoints

**War Game Map:**
- All standard map features plus:
- Collapsible control panel (click header to toggle)
- Collapsible Settings Panel in Gradio (groups all left-side controls)
- Add Military Unit mode: Click to place unit at location
- Unit type/echelon selection via dropdown
- Affiliation toggle (Friendly/Hostile)
- Click unit icon to delete (map-created units only)
- Waypoint management:
  - Click map in waypoint mode to add waypoints to selected unit
  - Drag waypoint markers to move them
  - Right-click waypoint to delete it
  - Waypoint paths shown as solid lines with arrow markers indicating direction
- Scenario save/load to JSON files

### Unit Protection and State Sync

The war game tracks two types of units:

1. **Protected units** (`protected: true`): Created via Gradio Settings Panel
   - Cannot be deleted by clicking on the map
   - Managed via "Clear Units (Settings)" button
   - Always included in scenario saves

2. **Map-created units** (`protected: false`): Created by clicking on the map
   - Can be deleted by clicking on the unit icon
   - Click "Sync from Map" to permanently save to Python state
   - Auto-displayed in status panels via Timer polling

**Real-time State Sync (Hidden Textbox Bridge Pattern):**

Due to iframe cross-origin restrictions, the solution uses postMessage + hidden textbox bridge:

```
Map iframe                          Parent Window                    Gradio
─────────────                       ─────────────                    ──────
notifyStateChange()
setInterval(500ms)
        │
        └── postMessage ──────────> message listener (state_sync_js)
                                          │
                                    querySelector('#map_state_holder textarea')
                                    textbox.value = stateJson
                                    dispatchEvent(new Event('input'))
                                          │
                                          └──────────────────────────> map_state_holder.change()
                                                                            │
                                                                      update_all_status()
                                                                            │
                                                                      - Units Summary (text)
                                                                      - Status (brief)
                                                                      - JSON display + auto-save
```

Key implementation:
- **wargame_map.py**: `notifyStateChange()` sends postMessage to parent with `stateJson`
- **wargame_map.py**: `setInterval()` sends periodic updates every 500ms
- **wargame_ui.py**: `state_sync_js` listens for postMessage and updates hidden textbox
- **wargame_ui.py**: `map_state_holder.change()` triggers `update_all_status()` Python callback
- **wargame_ui.py**: `auto_save_state()` saves combined state to JSON file

### Gradio Compatibility Notes

**Hidden Components (Gradio 5.x):**
- Use `visible="hidden"` instead of `visible=False` for components that need JavaScript access
- `visible=False` doesn't render in DOM (performance optimization in Gradio 5.x)
- `visible="hidden"` keeps element in DOM but hides it visually
- Reference: https://github.com/gradio-app/gradio/issues/11974

**JavaScript Injection:**
- Use `demo.load(fn=None, js=...)` for page-load scripts (works across Gradio versions)
- `gr.Blocks(js=...)` and `gr.Blocks(head=...)` may not work in older versions
- Event handlers support `js` parameter for client-side preprocessing

**Event Triggering from JavaScript:**
- Dispatch both `input` and `change` events for reliability
- Use `{ bubbles: true }` in Event constructor
