"""Tests for the war game functionality."""

import json
import pytest
from pathlib import Path

from map_mcp import (
    WarGameMap,
    Affiliation,
    Echelon,
    UnitType,
    MilitaryUnit,
    generate_sidc,
    parse_sidc,
)
from map_mcp.military_symbols import QuickSIDC


class TestSIDCGeneration:
    """Tests for SIDC code generation and parsing."""

    def test_generate_friendly_infantry_sidc(self):
        """Test generating SIDC for friendly infantry."""
        sidc = generate_sidc(
            Affiliation.FRIEND,
            UnitType.INFANTRY,
            Echelon.PLATOON,
        )
        assert len(sidc) == 20
        assert sidc.startswith("10")  # Version
        assert sidc[3] == "3"  # Friend affiliation

    def test_generate_hostile_armor_sidc(self):
        """Test generating SIDC for hostile armor."""
        sidc = generate_sidc(
            Affiliation.HOSTILE,
            UnitType.ARMOR,
            Echelon.COMPANY,
        )
        assert len(sidc) == 20
        assert sidc[3] == "6"  # Hostile affiliation

    def test_parse_sidc(self):
        """Test parsing SIDC code."""
        sidc = "10031001141211000000"
        parsed = parse_sidc(sidc)

        assert parsed["version"] == "10"
        assert parsed["affiliation"] == "Friend"
        assert parsed["symbol_set"] == "10"
        assert parsed["status"] == "Present"
        assert parsed["hq_indicator"] == True

    def test_quick_sidc_friendly_infantry(self):
        """Test QuickSIDC helper for friendly infantry."""
        sidc = QuickSIDC.friendly_infantry()
        parsed = parse_sidc(sidc)
        assert parsed["affiliation"] == "Friend"

    def test_quick_sidc_hostile_armor(self):
        """Test QuickSIDC helper for hostile armor."""
        sidc = QuickSIDC.hostile_armor()
        parsed = parse_sidc(sidc)
        assert parsed["affiliation"] == "Hostile"


class TestMilitaryUnit:
    """Tests for MilitaryUnit class."""

    def test_create_unit(self):
        """Test creating a military unit."""
        unit = MilitaryUnit(
            id="test_unit_1",
            name="Test Platoon",
            affiliation=Affiliation.FRIEND,
            unit_type=UnitType.INFANTRY,
            echelon=Echelon.PLATOON,
            location=(37.5, 126.9),
        )

        assert unit.id == "test_unit_1"
        assert unit.name == "Test Platoon"
        assert unit.affiliation == Affiliation.FRIEND
        assert unit.location == (37.5, 126.9)

    def test_unit_sidc_generation(self):
        """Test that unit generates correct SIDC."""
        unit = MilitaryUnit(
            id="test_unit_2",
            name="Test Company",
            affiliation=Affiliation.HOSTILE,
            unit_type=UnitType.ARMOR,
            echelon=Echelon.COMPANY,
            location=(37.5, 126.9),
        )

        sidc = unit.generate_sidc()
        assert len(sidc) == 20
        assert sidc[3] == "6"  # Hostile

    def test_unit_to_dict(self):
        """Test unit serialization to dict."""
        unit = MilitaryUnit(
            id="test_unit_3",
            name="HQ Unit",
            affiliation=Affiliation.FRIEND,
            unit_type=UnitType.HEADQUARTERS,
            echelon=Echelon.BATTALION,
            location=(37.5, 126.9),
            is_headquarters=True,
            waypoints=[(37.51, 126.91), (37.52, 126.92)],
        )

        data = unit.to_dict()
        assert data["id"] == "test_unit_3"
        assert data["name"] == "HQ Unit"
        assert data["is_headquarters"] == True
        assert len(data["waypoints"]) == 2


class TestWarGameMap:
    """Tests for WarGameMap class."""

    def test_create_map(self):
        """Test creating a war game map."""
        wmap = WarGameMap(center=(37.5, 126.9), zoom=12)

        assert wmap.center == (37.5, 126.9)
        assert wmap.zoom == 12
        assert len(wmap.military_units) == 0

    def test_add_friendly_unit(self):
        """Test adding a friendly unit."""
        wmap = WarGameMap()
        unit = wmap.add_friendly_unit(
            name="Alpha Platoon",
            unit_type=UnitType.INFANTRY,
            location=(37.5, 126.9),
        )

        assert unit.affiliation == Affiliation.FRIEND
        assert len(wmap.get_friendly_units()) == 1
        assert len(wmap.get_hostile_units()) == 0

    def test_add_hostile_unit(self):
        """Test adding a hostile unit."""
        wmap = WarGameMap()
        unit = wmap.add_hostile_unit(
            name="Enemy Force",
            unit_type=UnitType.ARMOR,
            location=(37.5, 126.9),
        )

        assert unit.affiliation == Affiliation.HOSTILE
        assert len(wmap.get_hostile_units()) == 1
        assert len(wmap.get_friendly_units()) == 0

    def test_add_waypoints(self):
        """Test adding waypoints to a unit."""
        wmap = WarGameMap()
        unit = wmap.add_friendly_unit(
            name="Recon",
            unit_type=UnitType.RECONNAISSANCE,
            location=(37.5, 126.9),
        )

        assert wmap.add_waypoint(unit.id, (37.51, 126.91))
        assert wmap.add_waypoint(unit.id, (37.52, 126.92))
        assert len(unit.waypoints) == 2

    def test_clear_waypoints(self):
        """Test clearing waypoints."""
        wmap = WarGameMap()
        unit = wmap.add_friendly_unit(
            name="Test",
            unit_type=UnitType.INFANTRY,
            location=(37.5, 126.9),
            waypoints=[(37.51, 126.91)],
        )

        assert wmap.clear_waypoints(unit.id)
        assert len(unit.waypoints) == 0

    def test_remove_unit(self):
        """Test removing a unit."""
        wmap = WarGameMap()
        unit = wmap.add_friendly_unit(
            name="Test",
            unit_type=UnitType.INFANTRY,
            location=(37.5, 126.9),
        )

        assert len(wmap.military_units) == 1
        assert wmap.remove_unit(unit.id)
        assert len(wmap.military_units) == 0

    def test_update_unit_location(self):
        """Test updating unit location."""
        wmap = WarGameMap()
        unit = wmap.add_friendly_unit(
            name="Mobile",
            unit_type=UnitType.CAVALRY,
            location=(37.5, 126.9),
        )

        new_location = (37.55, 126.95)
        assert wmap.update_unit_location(unit.id, new_location)
        assert unit.location == new_location

    def test_get_state(self):
        """Test getting map state."""
        wmap = WarGameMap(center=(37.5, 126.9), zoom=10)
        wmap.add_friendly_unit(
            name="Unit 1",
            unit_type=UnitType.INFANTRY,
            location=(37.51, 126.91),
        )
        wmap.add_hostile_unit(
            name="Enemy 1",
            unit_type=UnitType.ARMOR,
            location=(37.52, 126.92),
        )

        state = wmap.get_state()
        assert state["center"] == (37.5, 126.9)
        assert state["zoom"] == 10
        assert len(state["military_units"]) == 2

    def test_load_state(self):
        """Test loading map state."""
        state = {
            "center": (38.0, 127.0),
            "zoom": 11,
            "markers": [],
            "polylines": [],
            "polygons": [],
            "circles": [],
            "military_units": [
                {
                    "id": "unit_1",
                    "name": "Test Unit",
                    "affiliation": "FRIEND",
                    "unit_type": "INFANTRY",
                    "echelon": "PLATOON",
                    "location": (38.1, 127.1),
                    "waypoints": [],
                }
            ],
        }

        wmap = WarGameMap()
        wmap.load_state(state)

        assert wmap.center == (38.0, 127.0)
        assert wmap.zoom == 11
        assert len(wmap.military_units) == 1

    def test_to_html(self):
        """Test HTML generation includes milsymbol."""
        wmap = WarGameMap()
        wmap.add_friendly_unit(
            name="Alpha",
            unit_type=UnitType.INFANTRY,
            location=(37.5, 126.9),
        )

        html = wmap.to_html()

        assert "milsymbol" in html.lower() or "ms.Symbol" in html
        assert "wargame-controls" in html
        assert "INFANTRY" in html

    def test_clear_all_units(self):
        """Test clearing all units."""
        wmap = WarGameMap()
        wmap.add_friendly_unit("F1", UnitType.INFANTRY, (37.5, 126.9))
        wmap.add_friendly_unit("F2", UnitType.ARMOR, (37.5, 126.9))
        wmap.add_hostile_unit("H1", UnitType.ARTILLERY, (37.5, 126.9))

        assert len(wmap.military_units) == 3
        wmap.clear_all_units()
        assert len(wmap.military_units) == 0


class TestAffiliationEnum:
    """Tests for Affiliation enum."""

    def test_affiliation_colors(self):
        """Test affiliation colors."""
        assert Affiliation.FRIEND.color == "#00D0D0"
        assert Affiliation.HOSTILE.color == "#FF3333"
        assert Affiliation.NEUTRAL.color == "#00CC00"
        assert Affiliation.UNKNOWN.color == "#FFFF00"

    def test_affiliation_display_names(self):
        """Test affiliation display names."""
        assert Affiliation.FRIEND.name_display == "Friendly"
        assert Affiliation.HOSTILE.name_display == "Hostile"
        assert Affiliation.NEUTRAL.name_display == "Neutral"
        assert Affiliation.UNKNOWN.name_display == "Unknown"


class TestUnitTypeEnum:
    """Tests for UnitType enum."""

    def test_common_unit_types_exist(self):
        """Test that common unit types are defined."""
        assert UnitType.INFANTRY
        assert UnitType.ARMOR
        assert UnitType.ARTILLERY
        assert UnitType.RECONNAISSANCE
        assert UnitType.ENGINEER
        assert UnitType.HEADQUARTERS


class TestEchelonEnum:
    """Tests for Echelon enum."""

    def test_echelon_values(self):
        """Test echelon code values."""
        assert Echelon.TEAM.value == "11"
        assert Echelon.SQUAD.value == "12"
        assert Echelon.PLATOON.value == "14"
        assert Echelon.COMPANY.value == "15"
        assert Echelon.BATTALION.value == "16"
        assert Echelon.BRIGADE.value == "18"
        assert Echelon.DIVISION.value == "19"
