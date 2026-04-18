"""Military symbol utilities for war game applications.

This module provides SIDC (Symbol Identification Code) generation helpers
and military unit symbol support for offline map rendering using milsymbol.js.

Supports MIL-STD-2525 and STANAG APP-6 symbology standards.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json


class Affiliation(Enum):
    """Unit affiliation (friend/hostile/neutral/unknown)."""
    UNKNOWN = "0"
    FRIEND = "3"
    NEUTRAL = "4"
    HOSTILE = "6"

    @property
    def color(self) -> str:
        """Return the default color for this affiliation."""
        colors = {
            "0": "#FFFF00",  # Yellow for unknown
            "3": "#00D0D0",  # Cyan for friend
            "4": "#00CC00",  # Green for neutral
            "6": "#FF3333",  # Red for hostile
        }
        return colors[self.value]

    @property
    def name_display(self) -> str:
        """Return display name."""
        names = {
            "0": "Unknown",
            "3": "Friendly",
            "4": "Neutral",
            "6": "Hostile",
        }
        return names[self.value]


class SymbolSet(Enum):
    """Symbol set codes for different unit types."""
    UNKNOWN = "00"
    AIR = "01"
    AIR_MISSILE = "02"
    SPACE = "05"
    LAND_UNIT = "10"
    LAND_CIVILIAN = "11"
    LAND_EQUIPMENT = "15"
    LAND_INSTALLATION = "20"
    CONTROL_MEASURE = "25"
    DISMOUNTED = "27"
    SEA_SURFACE = "30"
    SEA_SUBSURFACE = "35"
    MINE_WARFARE = "36"
    ACTIVITIES = "40"
    CYBERSPACE = "60"


class Echelon(Enum):
    """Unit echelon/size indicators."""
    NONE = "00"
    TEAM = "11"
    SQUAD = "12"
    SECTION = "13"
    PLATOON = "14"
    COMPANY = "15"
    BATTERY = "15"  # Same as company for artillery
    BATTALION = "16"
    REGIMENT = "17"
    BRIGADE = "18"
    DIVISION = "19"
    CORPS = "20"
    ARMY = "21"
    ARMY_GROUP = "22"
    REGION = "23"


class UnitType(Enum):
    """Common unit function IDs for land units."""
    # Infantry
    INFANTRY = "121100"
    MECHANIZED_INFANTRY = "121102"
    MOTORIZED_INFANTRY = "121103"
    AIRBORNE_INFANTRY = "121104"
    AIR_ASSAULT_INFANTRY = "121105"
    MOUNTAIN_INFANTRY = "121106"
    LIGHT_INFANTRY = "121107"

    # Armor
    ARMOR = "120500"
    LIGHT_ARMOR = "120501"
    MEDIUM_ARMOR = "120502"
    HEAVY_ARMOR = "120503"
    AMPHIBIOUS_ARMOR = "120504"

    # Cavalry/Reconnaissance
    CAVALRY = "120600"
    RECONNAISSANCE = "121200"

    # Artillery
    ARTILLERY = "130300"
    SELF_PROPELLED_ARTILLERY = "130301"
    ROCKET_ARTILLERY = "130700"
    AIR_DEFENSE = "130200"

    # Combat Support
    ENGINEER = "140700"
    SIGNAL = "141100"
    MILITARY_INTELLIGENCE = "140900"
    MILITARY_POLICE = "141000"

    # Combat Service Support
    SUPPLY = "160000"
    TRANSPORTATION = "161200"
    MAINTENANCE = "160600"
    MEDICAL = "160800"

    # Aviation
    AVIATION = "110100"
    ATTACK_AVIATION = "110102"
    RECONNAISSANCE_AVIATION = "110103"
    UTILITY_AVIATION = "110104"

    # Special Operations
    SPECIAL_OPERATIONS = "121500"
    RANGER = "121501"

    # Headquarters
    HEADQUARTERS = "110000"
    COMMAND_POST = "111000"


@dataclass
class MilitaryUnit:
    """Represents a military unit on the map."""

    id: str
    name: str
    affiliation: Affiliation
    unit_type: UnitType
    echelon: Echelon = Echelon.PLATOON
    location: tuple[float, float] = (0.0, 0.0)

    # Optional properties
    strength: Optional[int] = None  # Personnel count
    designation: Optional[str] = None  # Unit designation (e.g., "1-1-A")
    higher_formation: Optional[str] = None  # Parent unit
    direction: Optional[float] = None  # Heading in degrees
    speed: Optional[float] = None  # Speed in km/h
    status: str = "present"  # present, planned, anticipated

    # Waypoints for movement planning
    waypoints: list[tuple[float, float]] = field(default_factory=list)

    # Additional modifiers
    reinforced: bool = False
    reduced: bool = False
    is_headquarters: bool = False

    def generate_sidc(self) -> str:
        """Generate a 20-character SIDC code for this unit.

        Format (2525D/E number-based):
        Positions 0-1: Version (10)
        Position 2: Context (0=Reality)
        Position 3: Standard Identity/Affiliation
        Positions 4-5: Symbol Set
        Position 6: Status (0=Present, 1=Planned)
        Position 7: HQ/Task Force/Dummy (0=None, 1=HQ)
        Positions 8-9: Echelon
        Positions 10-15: Function ID (entity/type/subtype)
        Positions 16-17: Modifier 1
        Positions 18-19: Modifier 2
        """
        version = "10"  # 2525D
        context = "0"   # Reality
        affiliation = self.affiliation.value
        symbol_set = SymbolSet.LAND_UNIT.value
        status = "0" if self.status == "present" else "1"
        hq_indicator = "1" if self.is_headquarters else "0"
        echelon = self.echelon.value
        function_id = self.unit_type.value
        modifier1 = "00"
        modifier2 = "00"

        # Handle reinforced/reduced
        if self.reinforced:
            modifier1 = "01"
        elif self.reduced:
            modifier1 = "02"

        sidc = f"{version}{context}{affiliation}{symbol_set}{status}{hq_indicator}{echelon}{function_id}{modifier1}{modifier2}"
        return sidc

    def to_dict(self) -> dict[str, Any]:
        """Convert unit to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "sidc": self.generate_sidc(),
            "affiliation": self.affiliation.name,
            "unit_type": self.unit_type.name,
            "echelon": self.echelon.name,
            "location": self.location,
            "strength": self.strength,
            "designation": self.designation,
            "higher_formation": self.higher_formation,
            "direction": self.direction,
            "speed": self.speed,
            "status": self.status,
            "waypoints": self.waypoints,
            "reinforced": self.reinforced,
            "reduced": self.reduced,
            "is_headquarters": self.is_headquarters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MilitaryUnit":
        """Create a MilitaryUnit from a dictionary."""
        echelon_name = data.get("echelon", "PLATOON")
        try:
            echelon = Echelon[echelon_name]
        except KeyError:
            echelon = Echelon.PLATOON

        return cls(
            id=data["id"],
            name=data["name"],
            affiliation=Affiliation[data["affiliation"]],
            unit_type=UnitType[data["unit_type"]],
            echelon=echelon,
            location=tuple(data.get("location", (0.0, 0.0))),
            strength=data.get("strength"),
            designation=data.get("designation"),
            higher_formation=data.get("higher_formation"),
            direction=data.get("direction"),
            speed=data.get("speed"),
            status=data.get("status", "present"),
            waypoints=data.get("waypoints", []),
            reinforced=data.get("reinforced", False),
            reduced=data.get("reduced", False),
            is_headquarters=data.get("is_headquarters", False),
        )


def generate_sidc(
    affiliation: Affiliation = Affiliation.FRIEND,
    unit_type: UnitType = UnitType.INFANTRY,
    echelon: Echelon = Echelon.PLATOON,
    is_headquarters: bool = False,
    status: str = "present",
) -> str:
    """Generate a SIDC code from component parts.

    Args:
        affiliation: Friend, hostile, neutral, or unknown
        unit_type: Type of unit (infantry, armor, etc.)
        echelon: Unit size (team, squad, platoon, etc.)
        is_headquarters: Whether this is a headquarters unit
        status: "present" or "planned"

    Returns:
        20-character SIDC code
    """
    version = "10"  # 2525D
    context = "0"   # Reality
    symbol_set = SymbolSet.LAND_UNIT.value
    status_code = "0" if status == "present" else "1"
    hq_indicator = "1" if is_headquarters else "0"

    sidc = f"{version}{context}{affiliation.value}{symbol_set}{status_code}{hq_indicator}{echelon.value}{unit_type.value}0000"
    return sidc


def parse_sidc(sidc: str) -> dict[str, Any]:
    """Parse a SIDC code and return its components.

    Args:
        sidc: 20-character SIDC code

    Returns:
        Dictionary with parsed components
    """
    if len(sidc) < 20:
        sidc = sidc.ljust(20, "0")

    affiliation_map = {
        "0": "Unknown",
        "1": "Unknown",
        "2": "Friend",
        "3": "Friend",
        "4": "Neutral",
        "5": "Hostile",
        "6": "Hostile",
    }

    status_map = {
        "0": "Present",
        "1": "Planned",
        "2": "Anticipated",
    }

    return {
        "version": sidc[0:2],
        "context": sidc[2],
        "affiliation": affiliation_map.get(sidc[3], "Unknown"),
        "symbol_set": sidc[4:6],
        "status": status_map.get(sidc[6], "Present"),
        "hq_indicator": sidc[7] == "1",
        "echelon": sidc[8:10],
        "function_id": sidc[10:16],
        "modifier1": sidc[16:18],
        "modifier2": sidc[18:20],
    }


# Quick SIDC generators for common unit types
class QuickSIDC:
    """Quick SIDC generation for common scenarios."""

    @staticmethod
    def friendly_infantry(echelon: Echelon = Echelon.PLATOON) -> str:
        return generate_sidc(Affiliation.FRIEND, UnitType.INFANTRY, echelon)

    @staticmethod
    def friendly_armor(echelon: Echelon = Echelon.COMPANY) -> str:
        return generate_sidc(Affiliation.FRIEND, UnitType.ARMOR, echelon)

    @staticmethod
    def friendly_artillery(echelon: Echelon = Echelon.BATTERY) -> str:
        return generate_sidc(Affiliation.FRIEND, UnitType.ARTILLERY, echelon)

    @staticmethod
    def hostile_infantry(echelon: Echelon = Echelon.PLATOON) -> str:
        return generate_sidc(Affiliation.HOSTILE, UnitType.INFANTRY, echelon)

    @staticmethod
    def hostile_armor(echelon: Echelon = Echelon.COMPANY) -> str:
        return generate_sidc(Affiliation.HOSTILE, UnitType.ARMOR, echelon)

    @staticmethod
    def hostile_artillery(echelon: Echelon = Echelon.BATTERY) -> str:
        return generate_sidc(Affiliation.HOSTILE, UnitType.ARTILLERY, echelon)

    @staticmethod
    def unknown_unit() -> str:
        return generate_sidc(Affiliation.UNKNOWN, UnitType.INFANTRY, Echelon.NONE)


# JavaScript code for milsymbol integration in Leaflet
def get_milsymbol_js_integration() -> str:
    """Get JavaScript code for milsymbol integration with Leaflet.

    This code should be included in the HTML after milsymbol is loaded.
    """
    return """
    // Military Symbol Layer for Leaflet using milsymbol
    var MilitarySymbolLayer = L.Layer.extend({
        initialize: function(options) {
            this._units = [];
            this._markers = {};
            this._waypointLines = {};
            L.setOptions(this, options);
        },

        onAdd: function(map) {
            this._map = map;
            this._updateSymbols();
        },

        onRemove: function(map) {
            this.clearAll();
            this._map = null;
        },

        addUnit: function(unit) {
            this._units.push(unit);
            if (this._map) {
                this._createMarker(unit);
            }
            return this;
        },

        removeUnit: function(unitId) {
            var idx = this._units.findIndex(u => u.id === unitId);
            if (idx > -1) {
                this._units.splice(idx, 1);
                if (this._markers[unitId]) {
                    this._map.removeLayer(this._markers[unitId]);
                    delete this._markers[unitId];
                }
                if (this._waypointLines[unitId]) {
                    this._map.removeLayer(this._waypointLines[unitId]);
                    delete this._waypointLines[unitId];
                }
            }
            return this;
        },

        updateUnit: function(unitId, updates) {
            var unit = this._units.find(u => u.id === unitId);
            if (unit) {
                Object.assign(unit, updates);
                this._updateMarker(unit);
            }
            return this;
        },

        getUnit: function(unitId) {
            return this._units.find(u => u.id === unitId);
        },

        getAllUnits: function() {
            return this._units.slice();
        },

        clearAll: function() {
            for (var id in this._markers) {
                this._map.removeLayer(this._markers[id]);
            }
            for (var id in this._waypointLines) {
                this._map.removeLayer(this._waypointLines[id]);
            }
            this._markers = {};
            this._waypointLines = {};
            this._units = [];
            return this;
        },

        _createMarker: function(unit) {
            var symbol = new ms.Symbol(unit.sidc, {
                size: unit.size || 35,
                uniqueDesignation: unit.designation || unit.name,
                direction: unit.direction,
                speed: unit.speed,
                staffComments: unit.staffComments,
                additionalInformation: unit.additionalInformation,
                quantity: unit.strength
            });

            var icon = L.divIcon({
                className: 'military-symbol',
                html: symbol.asSVG(),
                iconAnchor: [symbol.getAnchor().x, symbol.getAnchor().y],
                iconSize: [symbol.getSize().width, symbol.getSize().height]
            });

            var marker = L.marker(unit.location, {
                icon: icon,
                draggable: unit.draggable || false,
                unitId: unit.id
            });

            // Add popup with unit info
            var popupContent = this._createPopupContent(unit);
            marker.bindPopup(popupContent);

            // Add tooltip with unit name
            if (unit.name) {
                marker.bindTooltip(unit.name, {
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10]
                });
            }

            marker.addTo(this._map);
            this._markers[unit.id] = marker;

            // Draw waypoints if present
            if (unit.waypoints && unit.waypoints.length > 0) {
                this._drawWaypoints(unit);
            }

            // Setup drag events if draggable
            if (unit.draggable) {
                marker.on('dragend', function(e) {
                    var latlng = e.target.getLatLng();
                    unit.location = [latlng.lat, latlng.lng];
                    if (this.options.onUnitMove) {
                        this.options.onUnitMove(unit);
                    }
                }.bind(this));
            }
        },

        _updateMarker: function(unit) {
            // Remove old marker and recreate
            if (this._markers[unit.id]) {
                this._map.removeLayer(this._markers[unit.id]);
            }
            if (this._waypointLines[unit.id]) {
                this._map.removeLayer(this._waypointLines[unit.id]);
            }
            this._createMarker(unit);
        },

        _createPopupContent: function(unit) {
            var html = '<div class="unit-popup">';
            html += '<strong>' + (unit.name || 'Unknown Unit') + '</strong><br>';
            if (unit.designation) html += 'Designation: ' + unit.designation + '<br>';
            if (unit.affiliation) html += 'Affiliation: ' + unit.affiliation + '<br>';
            if (unit.strength) html += 'Strength: ' + unit.strength + '<br>';
            if (unit.status) html += 'Status: ' + unit.status + '<br>';
            html += 'Location: ' + unit.location[0].toFixed(4) + ', ' + unit.location[1].toFixed(4);
            if (unit.waypoints && unit.waypoints.length > 0) {
                html += '<br>Waypoints: ' + unit.waypoints.length;
            }
            html += '</div>';
            return html;
        },

        _drawWaypoints: function(unit) {
            var affiliation = unit.affiliation || 'FRIEND';
            var color = affiliation === 'HOSTILE' ? '#FF3333' : '#00D0D0';

            // Include current location as first point
            var path = [unit.location].concat(unit.waypoints);

            var line = L.polyline(path, {
                color: color,
                weight: 2,
                opacity: 0.7,
                dashArray: '5, 10'
            });

            // Add arrow decorations at waypoints
            for (var i = 0; i < unit.waypoints.length; i++) {
                var wp = unit.waypoints[i];
                var wpMarker = L.circleMarker(wp, {
                    radius: 5,
                    fillColor: color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                });
                wpMarker.bindPopup('Waypoint ' + (i + 1) + '<br>(' + wp[0].toFixed(4) + ', ' + wp[1].toFixed(4) + ')');
                wpMarker.addTo(this._map);
            }

            line.addTo(this._map);
            this._waypointLines[unit.id] = line;
        },

        _updateSymbols: function() {
            for (var i = 0; i < this._units.length; i++) {
                this._createMarker(this._units[i]);
            }
        }
    });

    // Factory function
    L.militarySymbolLayer = function(options) {
        return new MilitarySymbolLayer(options);
    };
    """


# Get CSS for military symbol styling
def get_milsymbol_css() -> str:
    """Get CSS styles for military symbols."""
    return """
    .military-symbol {
        background: transparent !important;
        border: none !important;
    }

    .unit-popup {
        font-family: Arial, sans-serif;
        font-size: 12px;
        min-width: 150px;
    }

    .unit-popup strong {
        font-size: 14px;
        color: #333;
    }

    .wargame-controls {
        position: absolute;
        top: 10px;
        left: 50px;
        z-index: 1000;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        max-height: 80vh;
        overflow-y: auto;
    }

    .wargame-controls h3 {
        margin: 0 0 10px 0;
        font-size: 14px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
    }

    .wargame-controls button {
        display: block;
        width: 100%;
        margin: 3px 0;
        padding: 6px 10px;
        cursor: pointer;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: #fff;
        font-size: 12px;
        text-align: left;
    }

    .wargame-controls button:hover {
        background: #f0f0f0;
    }

    .wargame-controls button.active {
        background: #0d6efd;
        color: white;
        border-color: #0d6efd;
    }

    .wargame-controls button.friendly {
        border-left: 4px solid #00D0D0;
    }

    .wargame-controls button.hostile {
        border-left: 4px solid #FF3333;
    }

    .unit-list {
        margin-top: 10px;
        max-height: 200px;
        overflow-y: auto;
    }

    .unit-item {
        padding: 5px;
        margin: 2px 0;
        border: 1px solid #ddd;
        border-radius: 3px;
        font-size: 11px;
        cursor: pointer;
    }

    .unit-item:hover {
        background: #f5f5f5;
    }

    .unit-item.selected {
        background: #e0e0ff;
        border-color: #0d6efd;
    }

    .unit-item .unit-name {
        font-weight: bold;
    }

    .unit-item .unit-info {
        color: #666;
        font-size: 10px;
    }
    """
