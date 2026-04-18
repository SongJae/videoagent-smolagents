"""War game map for military operations planning.

This module provides a specialized map class for war game applications,
supporting military unit symbols, movement planning, and tactical overlays.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from .map_utils import OfflineMap, _get_leaflet_assets
from .military_symbols import (
    Affiliation,
    Echelon,
    MilitaryUnit,
    UnitType,
    generate_sidc,
    get_milsymbol_css,
    get_milsymbol_js_integration,
)


# Cache for milsymbol JS
_MILSYMBOL_JS_CACHE: Optional[str] = None


def _get_milsymbol_js() -> str:
    """Load milsymbol JS from bundled assets."""
    global _MILSYMBOL_JS_CACHE

    if _MILSYMBOL_JS_CACHE is None:
        assets_dir = Path(__file__).parent / "assets"
        js_path = assets_dir / "milsymbol.min.js"

        if js_path.exists():
            _MILSYMBOL_JS_CACHE = js_path.read_text(encoding="utf-8")
        else:
            _MILSYMBOL_JS_CACHE = ""

    return _MILSYMBOL_JS_CACHE


class WarGameMap(OfflineMap):
    """A map specialized for war game operations with military symbols.

    Extends OfflineMap with support for:
    - Military unit symbols (NATO APP-6/MIL-STD-2525)
    - Unit movement planning with waypoints
    - Friendly and hostile force visualization
    - Tactical graphics and control measures
    """

    def __init__(
        self,
        center: tuple[float, float] = (37.5665, 126.9780),
        zoom: int = 10,
        tiles_dir: Optional[str | Path] = None,
        width: int = 1000,
        height: int = 700,
    ):
        """Initialize a war game map.

        Args:
            center: Initial map center as (latitude, longitude)
            zoom: Initial zoom level
            tiles_dir: Directory containing offline tiles
            width: Map width in pixels
            height: Map height in pixels
        """
        super().__init__(center, zoom, tiles_dir, width, height)
        self.military_units: list[MilitaryUnit] = []
        self._selected_unit_id: Optional[str] = None

    def add_military_unit(
        self,
        name: str,
        affiliation: Affiliation | str,
        unit_type: UnitType | str,
        location: tuple[float, float],
        echelon: Echelon | str = Echelon.PLATOON,
        designation: Optional[str] = None,
        strength: Optional[int] = None,
        direction: Optional[float] = None,
        is_headquarters: bool = False,
        waypoints: Optional[list[tuple[float, float]]] = None,
    ) -> MilitaryUnit:
        """Add a military unit to the map.

        Args:
            name: Unit name (e.g., "1st Platoon, Alpha Company")
            affiliation: FRIEND, HOSTILE, NEUTRAL, or UNKNOWN
            unit_type: Type of unit (INFANTRY, ARMOR, etc.)
            location: (latitude, longitude) position
            echelon: Unit size (TEAM, SQUAD, PLATOON, etc.)
            designation: Unit designation code (e.g., "1-1-A")
            strength: Personnel count
            direction: Heading in degrees (0-360)
            is_headquarters: Whether this is a HQ unit
            waypoints: List of planned movement positions

        Returns:
            The created MilitaryUnit object
        """
        # Convert string arguments to enums if needed
        if isinstance(affiliation, str):
            affiliation = Affiliation[affiliation.upper()]
        if isinstance(unit_type, str):
            unit_type = UnitType[unit_type.upper()]
        if isinstance(echelon, str):
            echelon = Echelon[echelon.upper()]

        unit = MilitaryUnit(
            id=f"unit_{uuid.uuid4().hex[:8]}",
            name=name,
            affiliation=affiliation,
            unit_type=unit_type,
            echelon=echelon,
            location=location,
            designation=designation,
            strength=strength,
            direction=direction,
            is_headquarters=is_headquarters,
            waypoints=waypoints or [],
        )

        self.military_units.append(unit)
        return unit

    def add_friendly_unit(
        self,
        name: str,
        unit_type: UnitType | str,
        location: tuple[float, float],
        echelon: Echelon | str = Echelon.PLATOON,
        **kwargs,
    ) -> MilitaryUnit:
        """Add a friendly unit (convenience method)."""
        return self.add_military_unit(
            name=name,
            affiliation=Affiliation.FRIEND,
            unit_type=unit_type,
            location=location,
            echelon=echelon,
            **kwargs,
        )

    def add_hostile_unit(
        self,
        name: str,
        unit_type: UnitType | str,
        location: tuple[float, float],
        echelon: Echelon | str = Echelon.PLATOON,
        **kwargs,
    ) -> MilitaryUnit:
        """Add a hostile unit (convenience method)."""
        return self.add_military_unit(
            name=name,
            affiliation=Affiliation.HOSTILE,
            unit_type=unit_type,
            location=location,
            echelon=echelon,
            **kwargs,
        )

    def get_unit(self, unit_id: str) -> Optional[MilitaryUnit]:
        """Get a unit by ID."""
        for unit in self.military_units:
            if unit.id == unit_id:
                return unit
        return None

    def get_units_by_affiliation(self, affiliation: Affiliation) -> list[MilitaryUnit]:
        """Get all units of a specific affiliation."""
        return [u for u in self.military_units if u.affiliation == affiliation]

    def get_friendly_units(self) -> list[MilitaryUnit]:
        """Get all friendly units."""
        return self.get_units_by_affiliation(Affiliation.FRIEND)

    def get_hostile_units(self) -> list[MilitaryUnit]:
        """Get all hostile units."""
        return self.get_units_by_affiliation(Affiliation.HOSTILE)

    def remove_unit(self, unit_id: str) -> bool:
        """Remove a unit from the map."""
        for i, unit in enumerate(self.military_units):
            if unit.id == unit_id:
                self.military_units.pop(i)
                return True
        return False

    def update_unit_location(
        self, unit_id: str, location: tuple[float, float]
    ) -> bool:
        """Update a unit's location."""
        unit = self.get_unit(unit_id)
        if unit:
            unit.location = location
            return True
        return False

    def add_waypoint(
        self, unit_id: str, waypoint: tuple[float, float]
    ) -> bool:
        """Add a waypoint to a unit's movement plan."""
        unit = self.get_unit(unit_id)
        if unit:
            unit.waypoints.append(waypoint)
            return True
        return False

    def clear_waypoints(self, unit_id: str) -> bool:
        """Clear all waypoints for a unit."""
        unit = self.get_unit(unit_id)
        if unit:
            unit.waypoints.clear()
            return True
        return False

    def set_waypoints(
        self, unit_id: str, waypoints: list[tuple[float, float]]
    ) -> bool:
        """Set all waypoints for a unit."""
        unit = self.get_unit(unit_id)
        if unit:
            unit.waypoints = waypoints
            return True
        return False

    def clear_all_units(self) -> None:
        """Remove all military units from the map."""
        self.military_units.clear()

    def clear(self) -> None:
        """Clear all map elements including military units."""
        super().clear()
        self.military_units.clear()

    def _generate_units_json(self) -> str:
        """Generate JSON for military units."""
        units_data = []
        for unit in self.military_units:
            unit_data = unit.to_dict()
            # Add display properties
            unit_data["size"] = 35  # Symbol size
            unit_data["draggable"] = True
            # Mark API-created units as protected (cannot be deleted by map click)
            unit_data["protected"] = True
            units_data.append(unit_data)
        return json.dumps(units_data)

    def to_html(self, milsymbol_url: Optional[str] = None) -> str:
        """Render the war game map as interactive HTML.

        Args:
            milsymbol_url: Optional URL to load milsymbol.js from instead of embedding.
                          Use this for Gradio UI to reduce HTML size (milsymbol.js is ~850KB).
                          When None, milsymbol.js is embedded inline.

        Returns:
            HTML string with embedded interactive map and military symbols
        """
        map_id = self._map_id
        tiles_json = self._generate_visible_tiles_json()
        units_json = self._generate_units_json()

        # Load embedded assets
        leaflet_js, leaflet_css = _get_leaflet_assets()

        # milsymbol.js can be loaded from URL or embedded
        if milsymbol_url:
            milsymbol_js = ""  # Will use external script tag
        else:
            milsymbol_js = _get_milsymbol_js()

        milsymbol_integration_js = get_milsymbol_js_integration()
        milsymbol_css = get_milsymbol_css()

        # Generate standard map elements (markers, polylines, etc.)
        markers_js = self._generate_markers_js()
        polylines_js = self._generate_polylines_js()
        polygons_js = self._generate_polygons_js()
        circles_js = self._generate_circles_js()
        geojson_js = self._generate_geojson_js()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>War Game Map</title>
    <style>
{leaflet_css}
    </style>
    <style>
{milsymbol_css}
        #{map_id} {{
            width: 100%;
            height: {self.height}px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .map-container {{
            position: relative;
        }}
        .map-info {{
            margin-top: 5px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 5px;
            font-size: 12px;
        }}
        .affiliation-legend {{
            display: flex;
            gap: 15px;
            margin-top: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .friendly-color {{ background: #00D0D0; }}
        .hostile-color {{ background: #FF3333; }}
        .neutral-color {{ background: #00CC00; }}
        .unknown-color {{ background: #FFFF00; }}
        /* Collapsible panel styles */
        .wargame-controls .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }}
        .wargame-controls .panel-header h3 {{
            margin: 0;
        }}
        .wargame-controls .toggle-btn {{
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            padding: 0 5px;
            color: #333;
        }}
        .wargame-controls .toggle-btn:hover {{
            color: #007bff;
        }}
        .wargame-controls .panel-content {{
            transition: max-height 0.3s ease-out, opacity 0.3s ease-out;
            overflow: hidden;
        }}
        .wargame-controls .panel-content.collapsed {{
            max-height: 0;
            opacity: 0;
            padding-top: 0;
            padding-bottom: 0;
        }}
        .wargame-controls .panel-content.expanded {{
            max-height: 1000px;
            opacity: 1;
        }}
        /* Waypoint marker styles for draggable */
        .waypoint-draggable {{
            cursor: move;
        }}
        .waypoint-marker {{
            cursor: grab;
        }}
        .waypoint-marker:active {{
            cursor: grabbing;
        }}
    </style>
</head>
<body>
    <div class="map-container">
        <div id="{map_id}"></div>
        <div class="wargame-controls" id="{map_id}_controls">
            <div class="panel-header" onclick="togglePanel_{map_id}()">
                <h3>War Game Controls</h3>
                <button class="toggle-btn" id="{map_id}_toggle_btn">&#9660;</button>
            </div>

            <div class="panel-content expanded" id="{map_id}_panel_content">
                <div style="margin-bottom: 10px; margin-top: 10px;">
                    <strong>Add Unit:</strong>
                    <button class="friendly" id="{map_id}_add_friendly_btn" onclick="toggleAddMode_{map_id}('friendly')">+ Friendly</button>
                    <button class="hostile" id="{map_id}_add_hostile_btn" onclick="toggleAddMode_{map_id}('hostile')">+ Hostile</button>
                </div>

                <div style="margin-bottom: 10px;">
                    <label>Unit Type:
                    <select id="{map_id}_unit_type">
                        <option value="INFANTRY">Infantry</option>
                        <option value="MECHANIZED_INFANTRY">Mechanized Infantry</option>
                        <option value="ARMOR">Armor</option>
                        <option value="RECONNAISSANCE">Reconnaissance</option>
                        <option value="ARTILLERY">Artillery</option>
                        <option value="AIR_DEFENSE">Air Defense</option>
                        <option value="ENGINEER">Engineer</option>
                        <option value="SIGNAL">Signal</option>
                        <option value="MEDICAL">Medical</option>
                        <option value="SUPPLY">Supply</option>
                        <option value="HEADQUARTERS">Headquarters</option>
                        <option value="SPECIAL_OPERATIONS">Special Ops</option>
                    </select>
                    </label>
                </div>

                <div style="margin-bottom: 10px;">
                    <label>Echelon:
                    <select id="{map_id}_echelon">
                        <option value="TEAM">Team</option>
                        <option value="SQUAD">Squad</option>
                        <option value="SECTION">Section</option>
                        <option value="PLATOON" selected>Platoon</option>
                        <option value="COMPANY">Company/Battery</option>
                        <option value="BATTALION">Battalion</option>
                        <option value="REGIMENT">Regiment</option>
                        <option value="BRIGADE">Brigade</option>
                        <option value="DIVISION">Division</option>
                        <option value="CORPS">Corps</option>
                    </select>
                    </label>
                </div>

                <div style="margin-bottom: 10px;">
                    <strong>Waypoints:</strong>
                    <button id="{map_id}_waypoint_btn" onclick="toggleWaypointMode_{map_id}()">Add Waypoints</button>
                    <div style="font-size: 11px; color: #666; margin-top: 3px;">Drag to move, right-click to delete</div>
                </div>

                <div style="margin-bottom: 10px;">
                    <button onclick="clearAllUnits_{map_id}()">Clear All Units</button>
                </div>

                <div style="margin-bottom: 10px; padding: 8px; background: #fff3cd; border-radius: 4px; font-size: 11px;">
                    <strong>Tip:</strong> Left-click unit to view info, right-click to delete
                </div>

                <div class="unit-list" id="{map_id}_unit_list">
                    <strong>Units:</strong>
                    <div id="{map_id}_units_display"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="map-info">
        <div id="{map_id}_coords">Center: ({self.center[0]:.4f}, {self.center[1]:.4f}) | Zoom: {self.zoom}</div>
        <div id="{map_id}_selection">No unit selected</div>
        <div class="affiliation-legend">
            <div class="legend-item"><div class="legend-color friendly-color"></div>Friendly</div>
            <div class="legend-item"><div class="legend-color hostile-color"></div>Hostile</div>
            <div class="legend-item"><div class="legend-color neutral-color"></div>Neutral</div>
            <div class="legend-item"><div class="legend-color unknown-color"></div>Unknown</div>
        </div>
    </div>

    <script>
{leaflet_js}
    </script>
    {f'<script src="{milsymbol_url}"></script>' if milsymbol_url else f'<script>{milsymbol_js}</script>'}
    <script>
{milsymbol_integration_js}
    </script>
    <script>
    (function() {{
        // Embedded tiles data
        var tilesData = {tiles_json};

        // Initial units data
        var initialUnits = {units_json};

        // Create custom tile layer with embedded tiles
        var OfflineTileLayer = L.TileLayer.extend({{
            getTileUrl: function(coords) {{
                var key = coords.z + '/' + coords.x + '/' + coords.y;
                if (tilesData[key]) {{
                    return 'data:image/png;base64,' + tilesData[key];
                }}
                return this._generatePlaceholderTile(coords);
            }},

            _generatePlaceholderTile: function(coords) {{
                var canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 256;
                var ctx = canvas.getContext('2d');
                ctx.fillStyle = '#e8e8e8';
                ctx.fillRect(0, 0, 256, 256);
                ctx.strokeStyle = '#ccc';
                ctx.lineWidth = 1;
                ctx.strokeRect(0, 0, 256, 256);
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
            attribution: 'Offline War Game Map'
        }}).addTo(map);

        // State management
        var addMode = null;  // 'friendly', 'hostile', or null
        var waypointMode = false;
        var selectedUnitId = null;
        var unitCounter = 1;

        // Military symbol layer
        var militaryLayer = L.militarySymbolLayer({{
            onUnitMove: function(unit) {{
                updateUnitsList();
                updateSelectionDisplay();
            }}
        }}).addTo(map);

        // Unit markers storage (for click handlers)
        var unitMarkers = {{}};
        var waypointMarkers = {{}};
        var waypointLines = {{}};
        var waypointArrows = {{}};

        // Calculate angle between two points for arrow rotation
        function getAngle(p1, p2) {{
            var dy = p2[0] - p1[0];
            var dx = Math.cos(Math.PI / 180 * p1[0]) * (p2[1] - p1[1]);
            var angle = Math.atan2(dy, dx) * 180 / Math.PI;
            return angle;
        }}

        // Get midpoint between two points
        function getMidpoint(p1, p2) {{
            return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
        }}

        // SIDC generation function
        function generateSIDC(affiliation, unitType, echelon, isHQ) {{
            var version = "10";
            var context = "0";

            var affiliationCodes = {{
                'FRIEND': '3',
                'HOSTILE': '6',
                'NEUTRAL': '4',
                'UNKNOWN': '0'
            }};
            var affCode = affiliationCodes[affiliation] || '0';

            var symbolSet = "10"; // Land unit

            var status = "0"; // Present
            var hqIndicator = isHQ ? "1" : "0";

            var echelonCodes = {{
                'NONE': '00', 'TEAM': '11', 'SQUAD': '12', 'SECTION': '13',
                'PLATOON': '14', 'COMPANY': '15', 'BATTERY': '15',
                'BATTALION': '16', 'REGIMENT': '17', 'BRIGADE': '18',
                'DIVISION': '19', 'CORPS': '20', 'ARMY': '21'
            }};
            var echelonCode = echelonCodes[echelon] || '14';

            var unitTypeCodes = {{
                'INFANTRY': '121100',
                'MECHANIZED_INFANTRY': '121102',
                'MOTORIZED_INFANTRY': '121103',
                'ARMOR': '120500',
                'RECONNAISSANCE': '121200',
                'CAVALRY': '120600',
                'ARTILLERY': '130300',
                'AIR_DEFENSE': '130200',
                'ENGINEER': '140700',
                'SIGNAL': '141100',
                'MEDICAL': '160800',
                'SUPPLY': '160000',
                'HEADQUARTERS': '110000',
                'SPECIAL_OPERATIONS': '121500',
                'AVIATION': '110100'
            }};
            var funcId = unitTypeCodes[unitType] || '121100';

            return version + context + affCode + symbolSet + status + hqIndicator + echelonCode + funcId + '0000';
        }}

        // Get current map state (all units including map-clicked ones)
        function getMapState() {{
            var units = [];
            for (var id in unitMarkers) {{
                var unit = unitMarkers[id].unitData;
                units.push({{
                    id: unit.id,
                    name: unit.name,
                    sidc: unit.sidc,
                    affiliation: unit.affiliation,
                    unit_type: unit.unit_type,
                    echelon: unit.echelon,
                    location: unit.location,
                    designation: unit.designation,
                    waypoints: unit.waypoints || [],
                    protected: unit.protected || false
                }});
            }}
            return {{
                center: [map.getCenter().lat, map.getCenter().lng],
                zoom: map.getZoom(),
                military_units: units
            }};
        }}

        // Notify parent window of state changes via postMessage (cross-origin safe)
        function notifyStateChange(action, data) {{
            try {{
                var state = getMapState();
                // Use postMessage for cross-origin safe communication
                window.parent.postMessage({{
                    type: 'wargame_state_change',
                    action: action,
                    state: state,
                    stateJson: JSON.stringify(state),
                    timestamp: Date.now()
                }}, '*');
            }} catch (e) {{
                // Ignore errors if not in iframe
                console.log('postMessage error:', e);
            }}
        }}

        // Periodically send state to parent via postMessage (for initial load and redundancy)
        setInterval(function() {{
            try {{
                var state = getMapState();
                window.parent.postMessage({{
                    type: 'wargame_state_update',
                    state: state,
                    stateJson: JSON.stringify(state),
                    timestamp: Date.now()
                }}, '*');
            }} catch (e) {{}}
        }}, 500);

        // Expose getMapState globally for parent window access
        window.getWargameState_{map_id} = getMapState;

        // Add a military unit to the map
        function addUnit(latlng, affiliation) {{
            var unitType = document.getElementById('{map_id}_unit_type').value;
            var echelon = document.getElementById('{map_id}_echelon').value;
            var isHQ = unitType === 'HEADQUARTERS';

            var unitId = 'unit_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            var unitName = affiliation.charAt(0) + '-' + unitCounter;
            unitCounter++;

            var sidc = generateSIDC(affiliation, unitType, echelon, isHQ);

            var unit = {{
                id: unitId,
                name: unitName,
                sidc: sidc,
                affiliation: affiliation,
                unit_type: unitType,
                echelon: echelon,
                location: [latlng.lat, latlng.lng],
                designation: unitName,
                waypoints: [],
                draggable: true,
                size: 35,
                protected: false  // Map-created units are NOT protected
            }};

            createUnitMarker(unit);
            updateUnitsList();
            notifyStateChange('unit_added', unit);

            return unit;
        }}

        // Create a marker for a unit
        function createUnitMarker(unit) {{
            var symbol = new ms.Symbol(unit.sidc, {{
                size: unit.size || 35,
                uniqueDesignation: unit.designation || unit.name
            }});

            var icon = L.divIcon({{
                className: 'military-symbol',
                html: symbol.asSVG(),
                iconAnchor: [symbol.getAnchor().x, symbol.getAnchor().y],
                iconSize: [symbol.getSize().width, symbol.getSize().height]
            }});

            var marker = L.marker(unit.location, {{
                icon: icon,
                draggable: true
            }});

            marker.unitData = unit;

            // Popup content
            var popupHtml = '<div class="unit-popup">' +
                '<strong>' + unit.name + '</strong><br>' +
                'Type: ' + unit.unit_type + '<br>' +
                'Echelon: ' + unit.echelon + '<br>' +
                'Affiliation: ' + unit.affiliation + '<br>' +
                'Location: ' + unit.location[0].toFixed(4) + ', ' + unit.location[1].toFixed(4) +
                '</div>';
            marker.bindPopup(popupHtml);

            marker.bindTooltip(unit.name, {{
                permanent: false,
                direction: 'top',
                offset: [0, -20]
            }});

            // Left-click handler: show unit information popup and select unit
            marker.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                selectUnit(unit.id);
                marker.openPopup();
            }});

            // Right-click handler: delete unit (only non-protected units)
            marker.on('contextmenu', function(e) {{
                L.DomEvent.stopPropagation(e);
                L.DomEvent.preventDefault(e);
                if (unit.protected) {{
                    alert('This unit was created via settings panel and cannot be deleted by right-clicking.\\nUse "Clear Units (Settings)" button to remove it.');
                    return;
                }}
                if (confirm('Delete unit "' + unit.name + '"?')) {{
                    deleteUnit(unit.id);
                }}
            }});

            // Drag handler
            marker.on('dragend', function(e) {{
                var pos = e.target.getLatLng();
                unit.location = [pos.lat, pos.lng];
                updateWaypointLine(unit.id);
                updateUnitsList();
                notifyStateChange('unit_moved', unit);
            }});

            marker.addTo(map);
            unitMarkers[unit.id] = marker;

            // Draw initial waypoints if any
            if (unit.waypoints && unit.waypoints.length > 0) {{
                updateWaypointLine(unit.id);
            }}
        }}

        // Select a unit
        function selectUnit(unitId) {{
            selectedUnitId = unitId;

            // Update visual selection
            document.querySelectorAll('.unit-item').forEach(function(el) {{
                el.classList.remove('selected');
            }});
            var selectedEl = document.querySelector('.unit-item[data-unit-id="' + unitId + '"]');
            if (selectedEl) {{
                selectedEl.classList.add('selected');
            }}

            updateSelectionDisplay();
        }}

        // Update selection display
        function updateSelectionDisplay() {{
            var displayEl = document.getElementById('{map_id}_selection');
            if (selectedUnitId && unitMarkers[selectedUnitId]) {{
                var unit = unitMarkers[selectedUnitId].unitData;
                displayEl.innerHTML = 'Selected: <strong>' + unit.name + '</strong> (' +
                    unit.unit_type + ', ' + unit.affiliation + ')' +
                    ' | Waypoints: ' + (unit.waypoints ? unit.waypoints.length : 0);
            }} else {{
                displayEl.innerHTML = 'No unit selected (click a unit to select)';
            }}
        }}

        // Update units list display
        function updateUnitsList() {{
            var container = document.getElementById('{map_id}_units_display');
            var html = '';

            var friendlyUnits = [];
            var hostileUnits = [];

            for (var id in unitMarkers) {{
                var unit = unitMarkers[id].unitData;
                if (unit.affiliation === 'FRIEND') {{
                    friendlyUnits.push(unit);
                }} else {{
                    hostileUnits.push(unit);
                }}
            }}

            if (friendlyUnits.length > 0) {{
                html += '<div style="color:#00A0A0; font-weight:bold; margin-top:5px;">Friendly (' + friendlyUnits.length + ')</div>';
                friendlyUnits.forEach(function(u) {{
                    var selected = selectedUnitId === u.id ? ' selected' : '';
                    html += '<div class="unit-item' + selected + '" data-unit-id="' + u.id + '" onclick="selectUnit_' + '{map_id}' + '(\\'' + u.id + '\\')">' +
                        '<div class="unit-name">' + u.name + '</div>' +
                        '<div class="unit-info">' + u.unit_type + ' | ' + u.echelon + '</div>' +
                        '</div>';
                }});
            }}

            if (hostileUnits.length > 0) {{
                html += '<div style="color:#CC0000; font-weight:bold; margin-top:5px;">Hostile (' + hostileUnits.length + ')</div>';
                hostileUnits.forEach(function(u) {{
                    var selected = selectedUnitId === u.id ? ' selected' : '';
                    html += '<div class="unit-item' + selected + '" data-unit-id="' + u.id + '" onclick="selectUnit_' + '{map_id}' + '(\\'' + u.id + '\\')">' +
                        '<div class="unit-name">' + u.name + '</div>' +
                        '<div class="unit-info">' + u.unit_type + ' | ' + u.echelon + '</div>' +
                        '</div>';
                }});
            }}

            if (html === '') {{
                html = '<div style="color:#999; font-style:italic;">No units placed yet</div>';
            }}

            container.innerHTML = html;
        }}

        // Add waypoint to selected unit
        function addWaypoint(latlng) {{
            if (!selectedUnitId || !unitMarkers[selectedUnitId]) {{
                alert('Please select a unit first');
                return;
            }}

            var unit = unitMarkers[selectedUnitId].unitData;
            if (!unit.waypoints) {{
                unit.waypoints = [];
            }}
            unit.waypoints.push([latlng.lat, latlng.lng]);

            updateWaypointLine(selectedUnitId);
            updateSelectionDisplay();
            notifyStateChange('waypoint_added', {{unit_id: selectedUnitId, waypoint: [latlng.lat, latlng.lng]}});
        }}

        // Update waypoint line for a unit
        function updateWaypointLine(unitId) {{
            var marker = unitMarkers[unitId];
            if (!marker) return;

            var unit = marker.unitData;

            // Remove existing waypoint markers, line, and arrows
            if (waypointMarkers[unitId]) {{
                waypointMarkers[unitId].forEach(function(m) {{
                    map.removeLayer(m);
                }});
            }}
            if (waypointLines[unitId]) {{
                map.removeLayer(waypointLines[unitId]);
            }}
            if (waypointArrows[unitId]) {{
                waypointArrows[unitId].forEach(function(a) {{
                    map.removeLayer(a);
                }});
            }}

            if (!unit.waypoints || unit.waypoints.length === 0) return;

            // Determine color based on affiliation
            var color = unit.affiliation === 'HOSTILE' ? '#FF3333' : '#00D0D0';

            // Create path from unit location through waypoints
            var path = [unit.location].concat(unit.waypoints);

            // Draw line (solid line for better visibility)
            var line = L.polyline(path, {{
                color: color,
                weight: 3,
                opacity: 0.8
            }}).addTo(map);
            waypointLines[unitId] = line;

            // Add arrow markers at midpoint of each segment to indicate direction
            waypointArrows[unitId] = [];
            for (var i = 0; i < path.length - 1; i++) {{
                var p1 = path[i];
                var p2 = path[i + 1];
                var midpoint = getMidpoint(p1, p2);
                var angle = getAngle(p1, p2);

                // Create arrow icon using SVG
                var arrowIcon = L.divIcon({{
                    className: 'waypoint-arrow',
                    html: '<div style="transform: rotate(' + (90 - angle) + 'deg); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-bottom: 14px solid ' + color + '; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.3));"></div>',
                    iconSize: [16, 14],
                    iconAnchor: [8, 7]
                }});

                var arrowMarker = L.marker(midpoint, {{
                    icon: arrowIcon,
                    interactive: false
                }}).addTo(map);

                waypointArrows[unitId].push(arrowMarker);
            }}

            // Add waypoint markers (draggable with right-click delete)
            waypointMarkers[unitId] = [];
            unit.waypoints.forEach(function(wp, idx) {{
                // Create a draggable marker icon for waypoints
                var wpIcon = L.divIcon({{
                    className: 'waypoint-marker',
                    html: '<div style="width: 16px; height: 16px; border-radius: 50%; background: ' + color + '; border: 2px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.4); cursor: grab;"></div>',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                }});

                var wpMarker = L.marker(wp, {{
                    icon: wpIcon,
                    draggable: true
                }});

                wpMarker.waypointIndex = idx;
                wpMarker.unitId = unitId;

                wpMarker.bindPopup('Waypoint ' + (idx + 1) + '<br>(' + wp[0].toFixed(4) + ', ' + wp[1].toFixed(4) + ')<br><em>Drag to move, right-click to delete</em>');
                wpMarker.bindTooltip('WP' + (idx + 1), {{permanent: false, direction: 'top', offset: [0, -10]}});

                // Drag handler to update waypoint position
                wpMarker.on('dragend', function(e) {{
                    var pos = e.target.getLatLng();
                    var wpIdx = e.target.waypointIndex;
                    var uId = e.target.unitId;

                    if (unitMarkers[uId]) {{
                        var u = unitMarkers[uId].unitData;
                        u.waypoints[wpIdx] = [pos.lat, pos.lng];
                        updateWaypointLine(uId);
                        updateSelectionDisplay();
                        notifyStateChange('waypoint_moved', {{unit_id: uId, waypoint_index: wpIdx}});
                    }}
                }});

                // Right-click handler to delete waypoint
                wpMarker.on('contextmenu', function(e) {{
                    L.DomEvent.stopPropagation(e);
                    L.DomEvent.preventDefault(e);

                    var wpIdx = e.target.waypointIndex;
                    var uId = e.target.unitId;

                    if (unitMarkers[uId]) {{
                        var u = unitMarkers[uId].unitData;
                        u.waypoints.splice(wpIdx, 1);
                        updateWaypointLine(uId);
                        updateSelectionDisplay();
                        notifyStateChange('waypoint_deleted', {{unit_id: uId, waypoint_index: wpIdx}});
                    }}
                }});

                // Prevent map click when clicking waypoint
                wpMarker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                wpMarker.addTo(map);
                waypointMarkers[unitId].push(wpMarker);
            }});
        }}

        // Clear all units
        function clearAllUnits() {{
            for (var id in unitMarkers) {{
                map.removeLayer(unitMarkers[id]);
            }}
            for (var id in waypointMarkers) {{
                waypointMarkers[id].forEach(function(m) {{
                    map.removeLayer(m);
                }});
            }}
            for (var id in waypointLines) {{
                map.removeLayer(waypointLines[id]);
            }}
            for (var id in waypointArrows) {{
                waypointArrows[id].forEach(function(a) {{
                    map.removeLayer(a);
                }});
            }}
            unitMarkers = {{}};
            waypointMarkers = {{}};
            waypointLines = {{}};
            waypointArrows = {{}};
            selectedUnitId = null;
            unitCounter = 1;
            updateUnitsList();
            updateSelectionDisplay();
        }}

        // Delete a single unit
        function deleteUnit(unitId) {{
            var deletedUnit = unitMarkers[unitId] ? unitMarkers[unitId].unitData : null;
            if (unitMarkers[unitId]) {{
                map.removeLayer(unitMarkers[unitId]);
                delete unitMarkers[unitId];
            }}
            if (waypointMarkers[unitId]) {{
                waypointMarkers[unitId].forEach(function(m) {{
                    map.removeLayer(m);
                }});
                delete waypointMarkers[unitId];
            }}
            if (waypointLines[unitId]) {{
                map.removeLayer(waypointLines[unitId]);
                delete waypointLines[unitId];
            }}
            if (waypointArrows[unitId]) {{
                waypointArrows[unitId].forEach(function(a) {{
                    map.removeLayer(a);
                }});
                delete waypointArrows[unitId];
            }}
            if (selectedUnitId === unitId) {{
                selectedUnitId = null;
            }}
            updateUnitsList();
            updateSelectionDisplay();
            if (deletedUnit) {{
                notifyStateChange('unit_deleted', deletedUnit);
            }}
        }}

        // Toggle add mode
        window.toggleAddMode_{map_id} = function(affiliation) {{
            var friendlyBtn = document.getElementById('{map_id}_add_friendly_btn');
            var hostileBtn = document.getElementById('{map_id}_add_hostile_btn');
            var waypointBtn = document.getElementById('{map_id}_waypoint_btn');

            if (addMode === affiliation) {{
                addMode = null;
                friendlyBtn.classList.remove('active');
                hostileBtn.classList.remove('active');
            }} else {{
                addMode = affiliation;
                waypointMode = false;
                waypointBtn.classList.remove('active');
                friendlyBtn.classList.toggle('active', affiliation === 'friendly');
                hostileBtn.classList.toggle('active', affiliation === 'hostile');
            }}
        }};

        // Toggle waypoint mode
        window.toggleWaypointMode_{map_id} = function() {{
            var waypointBtn = document.getElementById('{map_id}_waypoint_btn');
            var friendlyBtn = document.getElementById('{map_id}_add_friendly_btn');
            var hostileBtn = document.getElementById('{map_id}_add_hostile_btn');

            waypointMode = !waypointMode;
            if (waypointMode) {{
                addMode = null;
                friendlyBtn.classList.remove('active');
                hostileBtn.classList.remove('active');
            }}
            waypointBtn.classList.toggle('active', waypointMode);
        }};

        // Toggle panel collapse/expand
        window.togglePanel_{map_id} = function() {{
            var panelContent = document.getElementById('{map_id}_panel_content');
            var toggleBtn = document.getElementById('{map_id}_toggle_btn');

            if (panelContent.classList.contains('expanded')) {{
                panelContent.classList.remove('expanded');
                panelContent.classList.add('collapsed');
                toggleBtn.innerHTML = '&#9654;';  // Right arrow
            }} else {{
                panelContent.classList.remove('collapsed');
                panelContent.classList.add('expanded');
                toggleBtn.innerHTML = '&#9660;';  // Down arrow
            }}
        }};

        // Clear all units (exposed)
        window.clearAllUnits_{map_id} = clearAllUnits;

        // Select unit (exposed)
        window.selectUnit_{map_id} = selectUnit;

        // Map click handler
        map.on('click', function(e) {{
            if (addMode) {{
                var affiliation = addMode === 'friendly' ? 'FRIEND' : 'HOSTILE';
                addUnit(e.latlng, affiliation);
            }} else if (waypointMode) {{
                addWaypoint(e.latlng);
            }}
        }});

        // Update coordinates display
        function updateCoordsDisplay() {{
            var center = map.getCenter();
            var zoom = map.getZoom();
            document.getElementById('{map_id}_coords').innerHTML =
                'Center: (' + center.lat.toFixed(4) + ', ' + center.lng.toFixed(4) + ') | Zoom: ' + zoom;
        }}

        map.on('moveend', updateCoordsDisplay);
        map.on('zoomend', updateCoordsDisplay);

        // Add existing markers/polylines/etc.
        {markers_js}
        {polylines_js}
        {polygons_js}
        {circles_js}
        {geojson_js}

        // Load initial units
        initialUnits.forEach(function(unit) {{
            createUnitMarker(unit);
        }});
        updateUnitsList();

        // Make map instance globally accessible
        window['{map_id}'] = map;
        window['{map_id}_militaryLayer'] = militaryLayer;
    }})();
    </script>
</body>
</html>
        """
        return html

    def _generate_markers_js(self) -> str:
        """Generate JavaScript for standard markers."""
        js = ""
        for marker in self.markers:
            lat, lon = marker["location"]
            color = self._get_color_value(marker["color"])
            popup_js = f'.bindPopup("{marker["popup"]}")' if marker.get("popup") else ""
            tooltip_js = f'.bindTooltip("{marker["tooltip"]}")' if marker.get("tooltip") else ""
            js += f"""
            L.circleMarker([{lat}, {lon}], {{
                radius: 8,
                fillColor: '{color}',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}){popup_js}{tooltip_js}.addTo(map);
            """
        return js

    def _generate_polylines_js(self) -> str:
        """Generate JavaScript for polylines."""
        js = ""
        for polyline in self.polylines:
            coords = [[lat, lon] for lat, lon in polyline["locations"]]
            color = self._get_color_value(polyline["color"])
            popup_js = f'.bindPopup("{polyline["popup"]}")' if polyline.get("popup") else ""
            js += f"""
            L.polyline({json.dumps(coords)}, {{
                color: '{color}',
                weight: {polyline["weight"]},
                opacity: {polyline["opacity"]}
            }}){popup_js}.addTo(map);
            """
        return js

    def _generate_polygons_js(self) -> str:
        """Generate JavaScript for polygons."""
        js = ""
        for polygon in self.polygons:
            coords = [[lat, lon] for lat, lon in polygon["locations"]]
            color = self._get_color_value(polygon["color"])
            fill_color = self._get_color_value(polygon["fill_color"])
            popup_js = f'.bindPopup("{polygon["popup"]}")' if polygon.get("popup") else ""
            js += f"""
            L.polygon({json.dumps(coords)}, {{
                color: '{color}',
                fillColor: '{fill_color}',
                fillOpacity: {polygon["fill_opacity"]},
                weight: {polygon["weight"]}
            }}){popup_js}.addTo(map);
            """
        return js

    def _generate_circles_js(self) -> str:
        """Generate JavaScript for circles."""
        js = ""
        for circle in self.circles:
            lat, lon = circle["location"]
            color = self._get_color_value(circle["color"])
            fill_color = self._get_color_value(circle["fill_color"])
            popup_js = f'.bindPopup("{circle["popup"]}")' if circle.get("popup") else ""
            js += f"""
            L.circle([{lat}, {lon}], {{
                radius: {circle["radius"]},
                color: '{color}',
                fillColor: '{fill_color}',
                fillOpacity: {circle["fill_opacity"]}
            }}){popup_js}.addTo(map);
            """
        return js

    def _generate_geojson_js(self) -> str:
        """Generate JavaScript for GeoJSON layers."""
        js = ""
        for layer in self.geojson_layers:
            style = layer["style"]
            style_js = json.dumps(style) if style else "{}"
            js += f"""
            L.geoJSON({json.dumps(layer["data"])}, {{
                style: function(feature) {{ return {style_js}; }}
            }}).addTo(map);
            """
        return js

    def get_state(self) -> dict[str, Any]:
        """Get the current map state including military units."""
        state = super().get_state()
        state["military_units"] = [unit.to_dict() for unit in self.military_units]
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load map state including military units."""
        super().load_state(state)
        self.military_units = []
        for unit_data in state.get("military_units", []):
            try:
                unit = MilitaryUnit.from_dict(unit_data)
                self.military_units.append(unit)
            except (KeyError, ValueError):
                pass  # Skip invalid unit data
