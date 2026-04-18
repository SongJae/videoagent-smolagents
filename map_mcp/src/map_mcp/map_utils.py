"""Map utilities for offline map rendering.

This module provides functions for creating and manipulating maps
with interactive Leaflet-based rendering for offline operation.
"""

import base64
import io
import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

# Cache for Leaflet assets (loaded once)
_LEAFLET_JS_CACHE: Optional[str] = None
_LEAFLET_CSS_CACHE: Optional[str] = None


def _get_leaflet_assets() -> tuple[str, str]:
    """Load Leaflet JS and CSS from bundled assets.

    Returns:
        Tuple of (js_content, css_content)
    """
    global _LEAFLET_JS_CACHE, _LEAFLET_CSS_CACHE

    if _LEAFLET_JS_CACHE is None or _LEAFLET_CSS_CACHE is None:
        assets_dir = Path(__file__).parent / "assets"

        js_path = assets_dir / "leaflet.min.js"
        css_path = assets_dir / "leaflet.min.css"

        if js_path.exists():
            _LEAFLET_JS_CACHE = js_path.read_text(encoding="utf-8")
        else:
            _LEAFLET_JS_CACHE = ""

        if css_path.exists():
            _LEAFLET_CSS_CACHE = css_path.read_text(encoding="utf-8")
        else:
            _LEAFLET_CSS_CACHE = ""

    return _LEAFLET_JS_CACHE, _LEAFLET_CSS_CACHE


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Convert latitude/longitude to tile numbers (with fractional part)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)


def num2deg(xtile: float, ytile: float, zoom: int) -> tuple[float, float]:
    """Convert tile numbers to latitude/longitude."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


class OfflineMap:
    """A map instance that works completely offline with interactive Leaflet."""

    TILE_SIZE = 256

    def __init__(
        self,
        center: tuple[float, float] = (37.5665, 126.9780),  # Seoul, Korea
        zoom: int = 10,
        tiles_dir: Optional[str | Path] = None,
        width: int = 800,
        height: int = 600,
    ):
        """Initialize an offline map.

        Args:
            center: Initial map center as (latitude, longitude)
            zoom: Initial zoom level
            tiles_dir: Directory containing offline tiles
            width: Map width in pixels
            height: Map height in pixels
        """
        self.center = center
        self.zoom = zoom
        self.tiles_dir = Path(tiles_dir) if tiles_dir else None
        self.width = width
        self.height = height
        self.markers: list[dict[str, Any]] = []
        self.polylines: list[dict[str, Any]] = []
        self.polygons: list[dict[str, Any]] = []
        self.circles: list[dict[str, Any]] = []
        self.geojson_layers: list[dict[str, Any]] = []
        self._map_id = f"map_{uuid.uuid4().hex[:8]}"
        self._tile_cache: dict[str, str] = {}  # Cache for base64 tiles

    def add_marker(
        self,
        location: tuple[float, float],
        popup: Optional[str] = None,
        tooltip: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "blue",
    ) -> dict[str, Any]:
        """Add a marker to the map."""
        marker = {
            "location": location,
            "popup": popup,
            "tooltip": tooltip,
            "icon": icon,
            "color": color,
        }
        self.markers.append(marker)
        return marker

    def add_polyline(
        self,
        locations: list[tuple[float, float]],
        color: str = "blue",
        weight: int = 3,
        opacity: float = 1.0,
        popup: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a polyline to the map."""
        polyline = {
            "locations": locations,
            "color": color,
            "weight": weight,
            "opacity": opacity,
            "popup": popup,
        }
        self.polylines.append(polyline)
        return polyline

    def add_polygon(
        self,
        locations: list[tuple[float, float]],
        color: str = "blue",
        fill: bool = True,
        fill_color: Optional[str] = None,
        fill_opacity: float = 0.3,
        weight: int = 2,
        popup: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a polygon to the map."""
        polygon = {
            "locations": locations,
            "color": color,
            "fill": fill,
            "fill_color": fill_color or color,
            "fill_opacity": fill_opacity,
            "weight": weight,
            "popup": popup,
        }
        self.polygons.append(polygon)
        return polygon

    def add_circle(
        self,
        location: tuple[float, float],
        radius: float,
        color: str = "blue",
        fill: bool = True,
        fill_color: Optional[str] = None,
        fill_opacity: float = 0.3,
        popup: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a circle to the map."""
        circle = {
            "location": location,
            "radius": radius,
            "color": color,
            "fill": fill,
            "fill_color": fill_color or color,
            "fill_opacity": fill_opacity,
            "popup": popup,
        }
        self.circles.append(circle)
        return circle

    def add_geojson(
        self,
        data: dict | str | Path,
        style: Optional[dict[str, Any]] = None,
        popup_property: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a GeoJSON layer to the map."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
            else:
                data = json.loads(str(data))

        layer = {
            "data": data,
            "style": style or {},
            "popup_property": popup_property,
        }
        self.geojson_layers.append(layer)
        return layer

    def set_center(self, lat: float, lon: float) -> None:
        """Set the map center."""
        self.center = (lat, lon)

    def set_zoom(self, zoom: int) -> None:
        """Set the zoom level."""
        self.zoom = zoom

    def clear(self) -> None:
        """Clear all map elements."""
        self.markers.clear()
        self.polylines.clear()
        self.polygons.clear()
        self.circles.clear()
        self.geojson_layers.clear()

    def _get_tile_base64(self, z: int, x: int, y: int) -> Optional[str]:
        """Get a tile as base64 string from disk."""
        cache_key = f"{z}/{x}/{y}"
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        if not self.tiles_dir:
            return None

        tile_path = self.tiles_dir / str(z) / str(x) / f"{y}.png"
        if tile_path.exists():
            with open(tile_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                self._tile_cache[cache_key] = b64
                return b64
        return None

    def _generate_visible_tiles_json(self) -> str:
        """Generate JSON object containing visible tiles as base64."""
        if not self.tiles_dir:
            return "{}"

        tiles = {}
        center_x, center_y = deg2num(self.center[0], self.center[1], self.zoom)

        # Calculate visible tile range (with buffer)
        tiles_x = math.ceil(self.width / self.TILE_SIZE) + 2
        tiles_y = math.ceil(self.height / self.TILE_SIZE) + 2

        # Load tiles for current zoom and adjacent zoom levels
        for z in range(max(1, self.zoom - 1), min(18, self.zoom + 2)):
            scale = 2 ** (z - self.zoom)
            cx, cy = deg2num(self.center[0], self.center[1], z)

            range_x = int(tiles_x * scale) + 2
            range_y = int(tiles_y * scale) + 2

            for dx in range(-range_x, range_x + 1):
                for dy in range(-range_y, range_y + 1):
                    tx = int(cx) + dx
                    ty = int(cy) + dy

                    if tx < 0 or ty < 0 or tx >= 2**z or ty >= 2**z:
                        continue

                    b64 = self._get_tile_base64(z, tx, ty)
                    if b64:
                        key = f"{z}/{tx}/{ty}"
                        tiles[key] = b64

        return json.dumps(tiles)

    def _get_color_value(self, color: str) -> str:
        """Convert color name to hex value."""
        color_map = {
            "blue": "#0d6efd",
            "red": "#dc3545",
            "green": "#198754",
            "orange": "#fd7e14",
            "purple": "#6f42c1",
            "darkblue": "#00008b",
            "darkgreen": "#006400",
            "cadetblue": "#5f9ea0",
            "darkred": "#8b0000",
            "lightred": "#ff8080",
            "beige": "#f5f5dc",
            "darkpurple": "#301934",
            "pink": "#ffc0cb",
            "lightblue": "#add8e6",
            "lightgreen": "#90ee90",
            "gray": "#808080",
            "black": "#000000",
            "lightgray": "#d3d3d3",
            "white": "#ffffff",
        }
        if color.lower() in color_map:
            return color_map[color.lower()]
        if color.startswith("#"):
            return color
        return "#0d6efd"

    def to_html(self) -> str:
        """Render the map as interactive HTML with Leaflet.

        Returns:
            HTML string with embedded interactive map
        """
        map_id = self._map_id
        tiles_json = self._generate_visible_tiles_json()

        # Load embedded Leaflet assets
        leaflet_js, leaflet_css = _get_leaflet_assets()

        # Generate markers JavaScript
        markers_js = ""
        for marker in self.markers:
            lat, lon = marker["location"]
            color = self._get_color_value(marker["color"])
            popup_js = f'.bindPopup("{marker["popup"]}")' if marker.get("popup") else ""
            tooltip_js = f'.bindTooltip("{marker["tooltip"]}")' if marker.get("tooltip") else ""
            markers_js += f"""
            L.circleMarker([{lat}, {lon}], {{
                radius: 8,
                fillColor: '{color}',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}){popup_js}{tooltip_js}.addTo(map);
            """

        # Generate polylines JavaScript
        polylines_js = ""
        for polyline in self.polylines:
            coords = [[lat, lon] for lat, lon in polyline["locations"]]
            color = self._get_color_value(polyline["color"])
            popup_js = f'.bindPopup("{polyline["popup"]}")' if polyline.get("popup") else ""
            polylines_js += f"""
            L.polyline({json.dumps(coords)}, {{
                color: '{color}',
                weight: {polyline["weight"]},
                opacity: {polyline["opacity"]}
            }}){popup_js}.addTo(map);
            """

        # Generate polygons JavaScript
        polygons_js = ""
        for polygon in self.polygons:
            coords = [[lat, lon] for lat, lon in polygon["locations"]]
            color = self._get_color_value(polygon["color"])
            fill_color = self._get_color_value(polygon["fill_color"])
            popup_js = f'.bindPopup("{polygon["popup"]}")' if polygon.get("popup") else ""
            polygons_js += f"""
            L.polygon({json.dumps(coords)}, {{
                color: '{color}',
                fillColor: '{fill_color}',
                fillOpacity: {polygon["fill_opacity"]},
                weight: {polygon["weight"]}
            }}){popup_js}.addTo(map);
            """

        # Generate circles JavaScript
        circles_js = ""
        for circle in self.circles:
            lat, lon = circle["location"]
            color = self._get_color_value(circle["color"])
            fill_color = self._get_color_value(circle["fill_color"])
            popup_js = f'.bindPopup("{circle["popup"]}")' if circle.get("popup") else ""
            circles_js += f"""
            L.circle([{lat}, {lon}], {{
                radius: {circle["radius"]},
                color: '{color}',
                fillColor: '{fill_color}',
                fillOpacity: {circle["fill_opacity"]}
            }}){popup_js}.addTo(map);
            """

        # Generate GeoJSON JavaScript
        geojson_js = ""
        for layer in self.geojson_layers:
            style = layer["style"]
            style_js = json.dumps(style) if style else "{}"
            geojson_js += f"""
            L.geoJSON({json.dumps(layer["data"])}, {{
                style: function(feature) {{ return {style_js}; }}
            }}).addTo(map);
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
{leaflet_css}
    </style>
    <script>
{leaflet_js}
    </script>
    <style>
        #{map_id} {{
            width: 100%;
            height: {self.height}px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .map-container {{
            position: relative;
        }}
        .map-controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 8px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .map-controls button {{
            display: block;
            width: 100%;
            margin: 2px 0;
            padding: 5px 10px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
        }}
        .map-controls button:hover {{
            background: #f0f0f0;
        }}
        .map-controls button.active {{
            background: #0d6efd;
            color: white;
            border-color: #0d6efd;
        }}
        .map-info {{
            margin-top: 5px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 5px;
            font-size: 12px;
        }}
        .marker-list {{
            max-height: 150px;
            overflow-y: auto;
            margin-top: 5px;
        }}
        .marker-item {{
            padding: 3px 5px;
            margin: 2px 0;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="map-container">
        <div id="{map_id}"></div>
        <div class="map-controls">
            <button id="{map_id}_marker_btn" onclick="toggleMarkerMode_{map_id}()">Add Marker</button>
            <button id="{map_id}_waypoint_btn" onclick="toggleWaypointMode_{map_id}()">Add Waypoint</button>
            <button onclick="clearMarkers_{map_id}()">Clear All</button>
        </div>
    </div>
    <div class="map-info">
        <div id="{map_id}_coords">Center: ({self.center[0]:.4f}, {self.center[1]:.4f}) | Zoom: {self.zoom}</div>
        <div id="{map_id}_markers" class="marker-list"></div>
    </div>

    <script>
    (function() {{
        // Embedded tiles data
        var tilesData = {tiles_json};

        // Create custom tile layer with embedded tiles
        var OfflineTileLayer = L.TileLayer.extend({{
            getTileUrl: function(coords) {{
                var key = coords.z + '/' + coords.x + '/' + coords.y;
                if (tilesData[key]) {{
                    return 'data:image/png;base64,' + tilesData[key];
                }}
                // Return placeholder tile URL (generated via canvas)
                return this._generatePlaceholderTile(coords);
            }},

            _generatePlaceholderTile: function(coords) {{
                var canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 256;
                var ctx = canvas.getContext('2d');

                // Background
                ctx.fillStyle = '#e8e8e8';
                ctx.fillRect(0, 0, 256, 256);

                // Grid
                ctx.strokeStyle = '#ccc';
                ctx.lineWidth = 1;
                ctx.strokeRect(0, 0, 256, 256);

                // Coordinate text
                ctx.fillStyle = '#999';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('z:' + coords.z + ' x:' + coords.x + ' y:' + coords.y, 128, 128);

                return canvas.toDataURL();
            }},

            createTile: function(coords, done) {{
                var tile = document.createElement('img');
                var key = coords.z + '/' + coords.x + '/' + coords.y;

                if (tilesData[key]) {{
                    tile.src = 'data:image/png;base64,' + tilesData[key];
                }} else {{
                    tile.src = this._generatePlaceholderTile(coords);
                }}

                tile.onload = function() {{ done(null, tile); }};
                tile.onerror = function() {{ done(null, tile); }};

                return tile;
            }}
        }});

        // Initialize map
        var map = L.map('{map_id}').setView([{self.center[0]}, {self.center[1]}], {self.zoom});

        // Add tile layer
        new OfflineTileLayer('', {{
            maxZoom: 18,
            attribution: 'Offline Map'
        }}).addTo(map);

        // Marker management
        var markers = [];
        var waypointCoords = [];
        var waypointLine = null;
        var markerMode = false;
        var waypointMode = false;

        // Add existing markers
        {markers_js}

        // Add existing polylines
        {polylines_js}

        // Add existing polygons
        {polygons_js}

        // Add existing circles
        {circles_js}

        // Add existing GeoJSON
        {geojson_js}

        // Update coordinates display
        function updateCoordsDisplay() {{
            var center = map.getCenter();
            var zoom = map.getZoom();
            document.getElementById('{map_id}_coords').innerHTML =
                'Center: (' + center.lat.toFixed(4) + ', ' + center.lng.toFixed(4) + ') | Zoom: ' + zoom;
        }}

        map.on('moveend', updateCoordsDisplay);
        map.on('zoomend', updateCoordsDisplay);

        // Update marker list display
        function updateMarkerList() {{
            var html = '<strong>Markers/Waypoints:</strong><br>';
            markers.forEach(function(m, i) {{
                var ll = m.getLatLng();
                html += '<div class="marker-item">M' + (i+1) + ': (' + ll.lat.toFixed(4) + ', ' + ll.lng.toFixed(4) + ')</div>';
            }});
            if (waypointCoords.length > 0) {{
                html += '<div class="marker-item"><strong>Waypoints: ' + waypointCoords.length + ' points</strong></div>';
            }}
            document.getElementById('{map_id}_markers').innerHTML = html;
        }}

        // Toggle marker mode
        window.toggleMarkerMode_{map_id} = function() {{
            markerMode = !markerMode;
            waypointMode = false;
            document.getElementById('{map_id}_marker_btn').classList.toggle('active', markerMode);
            document.getElementById('{map_id}_waypoint_btn').classList.remove('active');
        }};

        // Toggle waypoint mode
        window.toggleWaypointMode_{map_id} = function() {{
            waypointMode = !waypointMode;
            markerMode = false;
            document.getElementById('{map_id}_waypoint_btn').classList.toggle('active', waypointMode);
            document.getElementById('{map_id}_marker_btn').classList.remove('active');
        }};

        // Clear all markers
        window.clearMarkers_{map_id} = function() {{
            markers.forEach(function(m) {{ map.removeLayer(m); }});
            markers = [];
            waypointCoords = [];
            if (waypointLine) {{
                map.removeLayer(waypointLine);
                waypointLine = null;
            }}
            updateMarkerList();
        }};

        // Map click handler
        map.on('click', function(e) {{
            if (markerMode) {{
                var marker = L.circleMarker([e.latlng.lat, e.latlng.lng], {{
                    radius: 8,
                    fillColor: '#dc3545',
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).addTo(map);
                marker.bindPopup('Marker at (' + e.latlng.lat.toFixed(4) + ', ' + e.latlng.lng.toFixed(4) + ')');
                markers.push(marker);
                updateMarkerList();
            }} else if (waypointMode) {{
                waypointCoords.push([e.latlng.lat, e.latlng.lng]);

                // Add waypoint marker
                var wpMarker = L.circleMarker([e.latlng.lat, e.latlng.lng], {{
                    radius: 6,
                    fillColor: '#198754',
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).addTo(map);
                wpMarker.bindPopup('Waypoint ' + waypointCoords.length);
                markers.push(wpMarker);

                // Update polyline
                if (waypointLine) {{
                    map.removeLayer(waypointLine);
                }}
                if (waypointCoords.length > 1) {{
                    waypointLine = L.polyline(waypointCoords, {{
                        color: '#198754',
                        weight: 3,
                        opacity: 0.8
                    }}).addTo(map);
                }}

                updateMarkerList();
            }}
        }});

        // Make map instance globally accessible for debugging
        window['{map_id}'] = map;
    }})();
    </script>
</body>
</html>
        """
        return html

    def save(self, path: str | Path) -> None:
        """Save the map to a file.

        Args:
            path: Output file path (.html)
        """
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.to_html())

    def get_state(self) -> dict[str, Any]:
        """Get the current map state as a dictionary."""
        return {
            "center": self.center,
            "zoom": self.zoom,
            "markers": self.markers.copy(),
            "polylines": self.polylines.copy(),
            "polygons": self.polygons.copy(),
            "circles": self.circles.copy(),
            "geojson_layers": [
                {"style": layer["style"], "popup_property": layer["popup_property"]}
                for layer in self.geojson_layers
            ],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load map state from a dictionary."""
        self.center = tuple(state.get("center", self.center))
        self.zoom = state.get("zoom", self.zoom)
        self.markers = state.get("markers", [])
        self.polylines = state.get("polylines", [])
        self.polygons = state.get("polygons", [])
        self.circles = state.get("circles", [])


# For backward compatibility - static image rendering
def render_static_map(
    center: tuple[float, float],
    zoom: int,
    width: int,
    height: int,
    tiles_dir: Optional[Path],
    markers: list,
    polylines: list,
    polygons: list,
    circles: list,
) -> Image.Image:
    """Render a static map image using PIL.

    This is a fallback for environments where JavaScript is not available.
    """
    TILE_SIZE = 256

    def get_tile(z: int, x: int, y: int) -> Optional[Image.Image]:
        if not tiles_dir:
            return None
        tile_path = tiles_dir / str(z) / str(x) / f"{y}.png"
        if tile_path.exists():
            return Image.open(tile_path).convert("RGBA")
        return None

    def create_placeholder() -> Image.Image:
        return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (220, 220, 220, 255))

    def latlon_to_pixel(lat: float, lon: float) -> tuple[int, int]:
        center_x, center_y = deg2num(center[0], center[1], zoom)
        point_x, point_y = deg2num(lat, lon, zoom)
        px = int((point_x - center_x) * TILE_SIZE + width / 2)
        py = int((point_y - center_y) * TILE_SIZE + height / 2)
        return (px, py)

    # Create base image
    img = Image.new("RGBA", (width, height), (240, 240, 240, 255))

    # Calculate tiles
    center_x, center_y = deg2num(center[0], center[1], zoom)
    tiles_x = math.ceil(width / TILE_SIZE) + 1
    tiles_y = math.ceil(height / TILE_SIZE) + 1
    start_tile_x = int(center_x - tiles_x / 2)
    start_tile_y = int(center_y - tiles_y / 2)
    offset_x = int((center_x - start_tile_x - tiles_x / 2) * TILE_SIZE + width / 2)
    offset_y = int((center_y - start_tile_y - tiles_y / 2) * TILE_SIZE + height / 2)

    # Render tiles
    for ty in range(tiles_y + 1):
        for tx in range(tiles_x + 1):
            tile = get_tile(zoom, start_tile_x + tx, start_tile_y + ty)
            if tile is None:
                tile = create_placeholder()
            px = offset_x + tx * TILE_SIZE
            py = offset_y + ty * TILE_SIZE
            img.paste(tile, (px, py))

    # Draw overlays
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw markers
    for marker in markers:
        mx, my = latlon_to_pixel(*marker["location"])
        draw.ellipse([mx - 10, my - 10, mx + 10, my + 10], fill=(220, 53, 69, 255), outline=(255, 255, 255, 255), width=2)

    return img.convert("RGB")
