"""War Game Gradio UI for military operations planning.

This module provides a Gradio-based web interface for war game applications
with military unit symbols and movement planning.
"""

import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .military_symbols import Affiliation, Echelon, UnitType
from .wargame_map import WarGameMap

# Get the assets directory path
_ASSETS_DIR = Path(__file__).parent / "assets"

# Temp directory for HTML files served by Gradio
_TEMP_DIR: Optional[Path] = None


def _get_temp_dir() -> Path:
    """Get or create the temporary directory for HTML files."""
    global _TEMP_DIR

    if _TEMP_DIR is None:
        _TEMP_DIR = Path(tempfile.mkdtemp(prefix="wargame_"))

    return _TEMP_DIR


def _setup_static_paths() -> None:
    """Set up Gradio static file serving for assets and temp directory."""
    temp_dir = _get_temp_dir()
    # Register both assets and temp directories for static file serving
    gr.set_static_paths(paths=[str(_ASSETS_DIR), str(temp_dir)])


class WarGameUI:
    """Gradio-based war game user interface."""

    def __init__(self, tiles_dir: Optional[str | Path] = None):
        """Initialize the war game UI.

        Args:
            tiles_dir: Directory containing offline map tiles
        """
        self.tiles_dir = Path(tiles_dir) if tiles_dir else None
        self.map = WarGameMap(tiles_dir=self.tiles_dir, width=1000, height=700)
        self._html_counter = 0
        # Auto-save file path for scenario state
        self._auto_save_path = _get_temp_dir() / "wargame_state.json"

    def auto_save_state(self, map_state_json: str = "") -> str:
        """Auto-save the current state to JSON file and return formatted JSON.

        Combines Python state with map-clicked units from JavaScript.

        Args:
            map_state_json: JSON string from map JavaScript state

        Returns:
            Formatted JSON string of the combined state
        """
        # Get Python state
        state = self.map.get_state()

        # Merge map-clicked units from JavaScript
        if map_state_json and map_state_json != "{}":
            try:
                js_state = json.loads(map_state_json)
                js_units = js_state.get("military_units", [])
                # Add non-protected (map-clicked) units to the state
                map_clicked_units = [u for u in js_units if not u.get("protected", False)]
                if map_clicked_units:
                    if "map_clicked_units" not in state:
                        state["map_clicked_units"] = []
                    state["map_clicked_units"] = map_clicked_units
            except json.JSONDecodeError:
                pass

        # Save to file
        try:
            with open(self._auto_save_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass  # Silently fail if save fails

        # Return formatted JSON for display
        return json.dumps(state, indent=2)

    def _create_iframe_from_file(self, map_html: str, height: int = 750) -> str:
        """Create an iframe that loads HTML from a served file.

        Saves HTML to a temp file and serves it via Gradio's static file serving.
        This avoids the data URI null origin issue that blocks loading external scripts.

        Args:
            map_html: The full HTML document for the map
            height: Height of the iframe in pixels

        Returns:
            HTML with iframe loading from served file
        """
        # Generate unique filename
        self._html_counter += 1
        temp_dir = _get_temp_dir()
        html_file = temp_dir / f"wargame_map_{self._html_counter}.html"

        # Write HTML to temp file
        html_file.write_text(map_html, encoding="utf-8")

        # Create iframe with file URL
        file_url = f"/gradio_api/file={html_file}"
        return f"""
        <iframe
            src="{file_url}"
            style="width: 100%; height: {height}px; border: none; border-radius: 5px;"
        ></iframe>
        """

    def _get_map_html(self) -> str:
        """Get map HTML with milsymbol.js loaded from static file URL.

        Returns:
            HTML string with milsymbol.js loaded from static file URL
        """
        # Generate URL for milsymbol.js served from assets
        milsymbol_url = f"/gradio_api/file={_ASSETS_DIR}/milsymbol.min.js"
        return self.map.to_html(milsymbol_url=milsymbol_url)

    def _create_data_uri_iframe(self, map_html: str, height: int = 750) -> str:
        """Create an iframe using file serving for the map HTML.

        This method name is kept for compatibility but now uses file serving.

        Args:
            map_html: The full HTML document for the map
            height: Height of the iframe in pixels

        Returns:
            HTML with iframe
        """
        return self._create_iframe_from_file(map_html, height)

    def create_map(
        self,
        lat: float,
        lon: float,
        zoom: int,
    ) -> str:
        """Create a new map.

        Args:
            lat: Center latitude
            lon: Center longitude
            zoom: Zoom level

        Returns:
            HTML representation of the map
        """
        self.map = WarGameMap(
            center=(lat, lon),
            zoom=zoom,
            tiles_dir=self.tiles_dir,
            width=1000,
            height=700,
        )
        return self._create_data_uri_iframe(self._get_map_html())

    def add_friendly_unit(
        self,
        name: str,
        unit_type: str,
        echelon: str,
        lat: float,
        lon: float,
        designation: str,
        strength: int,
    ) -> tuple[str, str]:
        """Add a friendly unit to the map.

        Args:
            name: Unit name
            unit_type: Type of unit
            echelon: Unit size/echelon
            lat: Latitude
            lon: Longitude
            designation: Unit designation
            strength: Personnel strength

        Returns:
            Tuple of (map HTML, status message)
        """
        try:
            unit = self.map.add_friendly_unit(
                name=name,
                unit_type=unit_type,
                echelon=echelon,
                location=(lat, lon),
                designation=designation if designation else None,
                strength=int(strength) if strength else None,
            )
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Added friendly unit: {unit.name} (ID: {unit.id})",
            )
        except Exception as e:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Error adding unit: {e}",
            )

    def add_hostile_unit(
        self,
        name: str,
        unit_type: str,
        echelon: str,
        lat: float,
        lon: float,
        designation: str,
        strength: int,
    ) -> tuple[str, str]:
        """Add a hostile unit to the map.

        Args:
            name: Unit name
            unit_type: Type of unit
            echelon: Unit size/echelon
            lat: Latitude
            lon: Longitude
            designation: Unit designation
            strength: Personnel strength

        Returns:
            Tuple of (map HTML, status message)
        """
        try:
            unit = self.map.add_hostile_unit(
                name=name,
                unit_type=unit_type,
                echelon=echelon,
                location=(lat, lon),
                designation=designation if designation else None,
                strength=int(strength) if strength else None,
            )
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Added hostile unit: {unit.name} (ID: {unit.id})",
            )
        except Exception as e:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Error adding unit: {e}",
            )

    def set_unit_waypoints(
        self,
        unit_id: str,
        waypoints_json: str,
    ) -> tuple[str, str]:
        """Set waypoints for a unit.

        Args:
            unit_id: The unit ID
            waypoints_json: JSON array of [lat, lon] waypoints

        Returns:
            Tuple of (map HTML, status message)
        """
        try:
            waypoints = json.loads(waypoints_json)
            waypoints = [tuple(wp) for wp in waypoints]
            if self.map.set_waypoints(unit_id, waypoints):
                return (
                    self._create_data_uri_iframe(self._get_map_html()),
                    f"Set {len(waypoints)} waypoints for unit {unit_id}",
                )
            else:
                return (
                    self._create_data_uri_iframe(self._get_map_html()),
                    f"Unit {unit_id} not found",
                )
        except json.JSONDecodeError as e:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Error parsing waypoints JSON: {e}",
            )

    def clear_all_units(self) -> tuple[str, str]:
        """Clear all military units from the map.

        Returns:
            Tuple of (map HTML, status message)
        """
        self.map.clear_all_units()
        return (
            self._create_data_uri_iframe(self._get_map_html()),
            "All units cleared",
        )

    def clear_map(self) -> tuple[str, str]:
        """Clear all elements from the map.

        Returns:
            Tuple of (map HTML, status message)
        """
        self.map.clear()
        return (
            self._create_data_uri_iframe(self._get_map_html()),
            "Map cleared",
        )

    def get_state(self) -> str:
        """Get current map state as JSON.

        Returns:
            JSON string of map state
        """
        return json.dumps(self.map.get_state(), indent=2)

    def get_units_summary(self) -> str:
        """Get a summary of all units on the map (Python state only).

        Returns:
            Summary text
        """
        friendly = self.map.get_friendly_units()
        hostile = self.map.get_hostile_units()

        lines = ["=== War Game Status (Settings Panel Units) ===\n"]
        lines.append(f"Friendly Units: {len(friendly)}")
        for u in friendly:
            wp_count = len(u.waypoints) if u.waypoints else 0
            lines.append(
                f"  - {u.name} ({u.unit_type.name}, {u.echelon.name}) @ "
                f"({u.location[0]:.4f}, {u.location[1]:.4f}) | Waypoints: {wp_count}"
            )
            if u.waypoints:
                wp_json = json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in u.waypoints])
                lines.append(f"    Waypoints: {wp_json}")

        lines.append(f"\nHostile Units: {len(hostile)}")
        for u in hostile:
            wp_count = len(u.waypoints) if u.waypoints else 0
            lines.append(
                f"  - {u.name} ({u.unit_type.name}, {u.echelon.name}) @ "
                f"({u.location[0]:.4f}, {u.location[1]:.4f}) | Waypoints: {wp_count}"
            )
            if u.waypoints:
                wp_json = json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in u.waypoints])
                lines.append(f"    Waypoints: {wp_json}")

        return "\n".join(lines)

    def get_full_status(self, map_state_json: str = "") -> str:
        """Get combined status of Python units and map-clicked units.

        Args:
            map_state_json: JSON string from map JavaScript state

        Returns:
            Combined status text with all units
        """
        lines = []

        # First, add Python state units (from Settings Panel)
        friendly = self.map.get_friendly_units()
        hostile = self.map.get_hostile_units()

        lines.append("=" * 50)
        lines.append("SETTINGS PANEL UNITS (Protected)")
        lines.append("=" * 50)

        lines.append(f"\n[Friendly: {len(friendly)}]")
        for u in friendly:
            wp_count = len(u.waypoints) if u.waypoints else 0
            lines.append(f"  ID: {u.id}")
            lines.append(f"  Name: {u.name}")
            lines.append(f"  Type: {u.unit_type.name} | Echelon: {u.echelon.name}")
            lines.append(f"  Location: [{u.location[0]:.4f}, {u.location[1]:.4f}]")
            lines.append(f"  Waypoints ({wp_count}):")
            if u.waypoints:
                lines.append(f"    {json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in u.waypoints])}")
            else:
                lines.append("    []")
            lines.append("")

        lines.append(f"[Hostile: {len(hostile)}]")
        for u in hostile:
            wp_count = len(u.waypoints) if u.waypoints else 0
            lines.append(f"  ID: {u.id}")
            lines.append(f"  Name: {u.name}")
            lines.append(f"  Type: {u.unit_type.name} | Echelon: {u.echelon.name}")
            lines.append(f"  Location: [{u.location[0]:.4f}, {u.location[1]:.4f}]")
            lines.append(f"  Waypoints ({wp_count}):")
            if u.waypoints:
                lines.append(f"    {json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in u.waypoints])}")
            else:
                lines.append("    []")
            lines.append("")

        # Now add map-clicked units (from JavaScript state)
        if map_state_json and map_state_json != "{}":
            try:
                state = json.loads(map_state_json)
                map_units = state.get("military_units", [])

                # Filter to only non-protected (map-clicked) units
                map_clicked_units = [u for u in map_units if not u.get("protected", False)]

                if map_clicked_units:
                    lines.append("=" * 50)
                    lines.append("MAP-CLICKED UNITS (Click 'Sync' to save)")
                    lines.append("=" * 50)

                    map_friendly = [u for u in map_clicked_units if u.get("affiliation") == "FRIEND"]
                    map_hostile = [u for u in map_clicked_units if u.get("affiliation") == "HOSTILE"]

                    lines.append(f"\n[Friendly: {len(map_friendly)}]")
                    for u in map_friendly:
                        loc = u.get("location", [0, 0])
                        wps = u.get("waypoints", [])
                        lines.append(f"  ID: {u.get('id', 'N/A')}")
                        lines.append(f"  Name: {u.get('name', 'Unknown')}")
                        lines.append(f"  Type: {u.get('unit_type', 'N/A')} | Echelon: {u.get('echelon', 'N/A')}")
                        lines.append(f"  Location: [{loc[0]:.4f}, {loc[1]:.4f}]")
                        lines.append(f"  Waypoints ({len(wps)}):")
                        if wps:
                            lines.append(f"    {json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in wps])}")
                        else:
                            lines.append("    []")
                        lines.append("")

                    lines.append(f"[Hostile: {len(map_hostile)}]")
                    for u in map_hostile:
                        loc = u.get("location", [0, 0])
                        wps = u.get("waypoints", [])
                        lines.append(f"  ID: {u.get('id', 'N/A')}")
                        lines.append(f"  Name: {u.get('name', 'Unknown')}")
                        lines.append(f"  Type: {u.get('unit_type', 'N/A')} | Echelon: {u.get('echelon', 'N/A')}")
                        lines.append(f"  Location: [{loc[0]:.4f}, {loc[1]:.4f}]")
                        lines.append(f"  Waypoints ({len(wps)}):")
                        if wps:
                            lines.append(f"    {json.dumps([[round(wp[0], 4), round(wp[1], 4)] for wp in wps])}")
                        else:
                            lines.append("    []")
                        lines.append("")

            except json.JSONDecodeError:
                pass

        if not lines or all(line.startswith("=") or not line.strip() for line in lines):
            lines.append("\nNo units placed yet.")

        return "\n".join(lines)

    def get_brief_status(self, map_state_json: str = "") -> str:
        """Get a brief status summary for the Status textbox.

        Args:
            map_state_json: JSON string from map JavaScript state

        Returns:
            Brief status string
        """
        # Count Python state units
        py_friendly = len(self.map.get_friendly_units())
        py_hostile = len(self.map.get_hostile_units())

        # Count map-clicked units
        map_friendly = 0
        map_hostile = 0

        if map_state_json and map_state_json != "{}":
            try:
                state = json.loads(map_state_json)
                map_units = state.get("military_units", [])
                map_clicked = [u for u in map_units if not u.get("protected", False)]
                map_friendly = len([u for u in map_clicked if u.get("affiliation") == "FRIEND"])
                map_hostile = len([u for u in map_clicked if u.get("affiliation") == "HOSTILE"])
            except json.JSONDecodeError:
                pass

        total_friendly = py_friendly + map_friendly
        total_hostile = py_hostile + map_hostile

        if total_friendly == 0 and total_hostile == 0:
            return "No units placed"

        parts = []
        if total_friendly > 0:
            parts.append(f"Friendly: {total_friendly}")
        if total_hostile > 0:
            parts.append(f"Hostile: {total_hostile}")

        status = " | ".join(parts)

        if map_friendly > 0 or map_hostile > 0:
            status += f" (Map: {map_friendly + map_hostile} unsaved)"

        return status

    def save_map(self, filename: str) -> str:
        """Save map to HTML file.

        Args:
            filename: Output filename

        Returns:
            Status message
        """
        if not filename:
            return "Please provide a filename"

        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".html")

        self.map.save(output_path)
        return f"Map saved to {output_path}"

    def load_scenario(self, file) -> tuple[str, str]:
        """Load a scenario from a JSON file.

        Args:
            file: Uploaded file object

        Returns:
            Tuple of (map HTML, status message)
        """
        if file is None:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                "No file selected",
            )

        try:
            with open(file.name, "r") as f:
                state = json.load(f)
            self.map.load_state(state)
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Loaded scenario with {len(self.map.military_units)} units",
            )
        except Exception as e:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Error loading scenario: {e}",
            )

    def save_scenario(self, filename: str) -> str:
        """Save current scenario to JSON file.

        Args:
            filename: Output filename

        Returns:
            Status message
        """
        if not filename:
            return "Please provide a filename"

        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        try:
            state = self.map.get_state()
            with open(output_path, "w") as f:
                json.dump(state, f, indent=2)
            return f"Scenario saved to {output_path}"
        except Exception as e:
            return f"Error saving scenario: {e}"

    def sync_from_map(self, state_json: str) -> tuple[str, str]:
        """Sync units created on the map back to Python state.

        Args:
            state_json: JSON string of map state from JavaScript

        Returns:
            Tuple of (map HTML, status message)
        """
        if not state_json or state_json == "{}":
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                "No map state to sync",
            )

        try:
            state = json.loads(state_json)
            units = state.get("military_units", [])

            # Get IDs of existing Python units
            existing_ids = {u.id for u in self.map.military_units}

            # Add only new (non-protected) units from map
            added_count = 0
            for unit_data in units:
                if unit_data.get("id") not in existing_ids and not unit_data.get("protected", False):
                    # This is a new unit created on the map
                    try:
                        affiliation = Affiliation[unit_data.get("affiliation", "FRIEND")]
                        unit_type = UnitType[unit_data.get("unit_type", "INFANTRY")]
                        echelon_name = unit_data.get("echelon", "PLATOON")
                        try:
                            echelon = Echelon[echelon_name]
                        except KeyError:
                            echelon = Echelon.PLATOON

                        from .military_symbols import MilitaryUnit
                        new_unit = MilitaryUnit(
                            id=unit_data["id"],
                            name=unit_data.get("name", "Unknown"),
                            affiliation=affiliation,
                            unit_type=unit_type,
                            echelon=echelon,
                            location=tuple(unit_data.get("location", [0, 0])),
                            designation=unit_data.get("designation"),
                            waypoints=[tuple(wp) for wp in unit_data.get("waypoints", [])],
                        )
                        self.map.military_units.append(new_unit)
                        added_count += 1
                    except Exception:
                        pass  # Skip invalid unit data

            # Update existing units with new waypoints/locations
            updated_count = 0
            for unit_data in units:
                if unit_data.get("id") in existing_ids:
                    unit = self.map.get_unit(unit_data["id"])
                    if unit:
                        # Update location and waypoints
                        unit.location = tuple(unit_data.get("location", unit.location))
                        unit.waypoints = [tuple(wp) for wp in unit_data.get("waypoints", [])]
                        updated_count += 1

            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Synced: {added_count} units added, {updated_count} units updated",
            )
        except json.JSONDecodeError as e:
            return (
                self._create_data_uri_iframe(self._get_map_html()),
                f"Error parsing map state: {e}",
            )

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        # Set up static file serving for assets and temp HTML files
        # This enables loading milsymbol.js externally, reducing HTML from ~1MB to ~200KB
        _setup_static_paths()

        # Unit type choices
        unit_types = [e.name for e in UnitType]
        echelon_choices = [e.name for e in Echelon if e.name != "BATTERY"]

        # JavaScript to set up message listener for iframe state sync
        # Uses hidden textbox bridge pattern for reliable JS-to-Python communication
        # Reference: https://stackoverflow.com/questions/77586262
        # Note: visible="hidden" is required for Gradio 5.x compatibility (see GitHub issue #11974)
        state_sync_js = """
        function() {
            // Listen for postMessage from map iframe
            window.addEventListener('message', function(event) {
                if (event.data && (event.data.type === 'wargame_state_change' || event.data.type === 'wargame_state_update')) {
                    var stateJson = event.data.stateJson || JSON.stringify(event.data.state);

                    // Find the hidden textbox by elem_id and update its value
                    // This triggers Gradio's change event to call Python
                    // Try multiple selectors for compatibility across Gradio versions
                    var textbox = document.querySelector('#map_state_holder textarea') ||
                                  document.querySelector('#map_state_holder input') ||
                                  document.getElementById('map_state_holder');

                    if (textbox) {
                        // For textarea/input elements
                        if (textbox.tagName === 'TEXTAREA' || textbox.tagName === 'INPUT') {
                            textbox.value = stateJson;
                        } else {
                            // For container elements, find the input inside
                            var input = textbox.querySelector('textarea') || textbox.querySelector('input');
                            if (input) {
                                input.value = stateJson;
                                textbox = input;
                            }
                        }

                        // Dispatch input event to trigger Gradio's change detection
                        var inputEvent = new Event('input', { bubbles: true });
                        textbox.dispatchEvent(inputEvent);

                        // Also dispatch change event for redundancy
                        var changeEvent = new Event('change', { bubbles: true });
                        textbox.dispatchEvent(changeEvent);
                    } else {
                        console.warn('Could not find map_state_holder textbox');
                    }
                }
            });
            console.log('Wargame message listener initialized - using textbox bridge');
        }
        """

        with gr.Blocks(title="War Game Map") as demo:
            gr.Markdown(
                """
            # War Game Command Map

            Interactive map for military operations planning and war gaming.
            Place friendly (blue) and hostile (red) units, plan movement routes with waypoints.

            **Map Controls:**
            - **+ Friendly / + Hostile buttons**: Toggle placement mode, then click on map
            - **Add Waypoints button**: Select a unit first, then click to add waypoints
            - **Drag units**: Click and drag any unit to reposition
            - **Click unit icon**: Delete unit (map-created only)
            - **Scroll wheel**: Zoom in/out
            - **Click & drag map**: Pan the view
            """
            )

            # Hidden textbox for map state sync (bridge between JS and Python)
            # The postMessage listener updates this textbox, triggering change events
            # Note: Use visible="hidden" instead of visible=False for Gradio 5.x compatibility
            # visible=False doesn't render in DOM, visible="hidden" keeps it in DOM but hidden
            map_state_holder = gr.Textbox(visible="hidden", elem_id="map_state_holder")

            with gr.Row():
                # Left panel - Controls (all in one collapsible group)
                with gr.Column(scale=1, min_width=350):
                    with gr.Accordion("Settings Panel", open=True):
                        with gr.Accordion("Map Settings", open=True):
                            with gr.Row():
                                lat_input = gr.Number(label="Latitude", value=37.5665)
                                lon_input = gr.Number(label="Longitude", value=126.9780)
                            zoom_input = gr.Slider(
                                label="Zoom Level",
                                minimum=1,
                                maximum=18,
                                value=10,
                                step=1,
                            )
                            create_btn = gr.Button("Create/Reset Map", variant="primary")

                        with gr.Accordion("Add Friendly Unit", open=False):
                            friendly_name = gr.Textbox(
                                label="Unit Name",
                                value="Alpha Platoon",
                                placeholder="e.g., Alpha Platoon",
                            )
                            friendly_type = gr.Dropdown(
                                label="Unit Type",
                                choices=unit_types,
                                value="INFANTRY",
                            )
                            friendly_echelon = gr.Dropdown(
                                label="Echelon",
                                choices=echelon_choices,
                                value="PLATOON",
                            )
                            with gr.Row():
                                friendly_lat = gr.Number(label="Lat", value=37.57)
                                friendly_lon = gr.Number(label="Lon", value=126.98)
                            friendly_designation = gr.Textbox(
                                label="Designation",
                                placeholder="e.g., 1-1-A",
                            )
                            friendly_strength = gr.Number(label="Strength", value=30)
                            add_friendly_btn = gr.Button(
                                "Add Friendly Unit", variant="primary"
                            )

                        with gr.Accordion("Add Hostile Unit", open=False):
                            hostile_name = gr.Textbox(
                                label="Unit Name",
                                value="Enemy Force 1",
                                placeholder="e.g., Enemy Tank Co",
                            )
                            hostile_type = gr.Dropdown(
                                label="Unit Type",
                                choices=unit_types,
                                value="ARMOR",
                            )
                            hostile_echelon = gr.Dropdown(
                                label="Echelon",
                                choices=echelon_choices,
                                value="COMPANY",
                            )
                            with gr.Row():
                                hostile_lat = gr.Number(label="Lat", value=37.55)
                                hostile_lon = gr.Number(label="Lon", value=126.99)
                            hostile_designation = gr.Textbox(
                                label="Designation",
                                placeholder="e.g., RED-1",
                            )
                            hostile_strength = gr.Number(label="Strength", value=0)
                            add_hostile_btn = gr.Button(
                                "Add Hostile Unit", variant="stop"
                            )

                        with gr.Accordion("Set Waypoints (API)", open=False):
                            waypoint_unit_id = gr.Textbox(
                                label="Unit ID",
                                placeholder="unit_xxxxxxxx",
                            )
                            waypoints_json = gr.Textbox(
                                label="Waypoints (JSON)",
                                placeholder='[[37.58, 126.98], [37.59, 126.99]]',
                                lines=2,
                            )
                            set_waypoints_btn = gr.Button("Set Waypoints")

                        with gr.Accordion("Scenario Management", open=False):
                            scenario_file = gr.File(
                                label="Load Scenario",
                                file_types=[".json"],
                            )
                            load_scenario_btn = gr.Button("Load Scenario")
                            save_scenario_name = gr.Textbox(
                                label="Save Scenario As",
                                placeholder="scenario.json",
                            )
                            save_scenario_btn = gr.Button("Save Scenario")
                            save_html_name = gr.Textbox(
                                label="Export HTML As",
                                placeholder="wargame.html",
                            )
                            save_html_btn = gr.Button("Export as HTML")

                        with gr.Accordion("Actions", open=True):
                            sync_btn = gr.Button("Sync from Map", variant="secondary")
                            gr.Markdown(
                                "*Sync saves map-clicked units to scenario*",
                                elem_classes=["hint-text"]
                            )
                            with gr.Row():
                                clear_units_btn = gr.Button("Clear Units (Settings)")
                                clear_all_btn = gr.Button("Clear All", variant="secondary")
                            refresh_summary_btn = gr.Button("Refresh Status")

                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                        )

                # Right panel - Map display
                with gr.Column(scale=2):
                    map_display = gr.HTML(
                        value=self._create_data_uri_iframe(self._get_map_html()),
                    )

            with gr.Accordion("War Game Status", open=True):
                gr.Markdown(
                    f"*Auto-updates and saves to JSON when map state changes. File: `{self._auto_save_path}`*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        units_summary = gr.Textbox(
                            label="Units Summary",
                            value=self.get_full_status(""),
                            lines=15,
                            interactive=False,
                            elem_id="units_summary_display",
                        )
                    with gr.Column(scale=1):
                        state_json_display = gr.Code(
                            label="Auto-saved JSON State",
                            language="json",
                            value=self.auto_save_state(""),
                            lines=15,
                        )

            with gr.Accordion("Scenario State (Manual)", open=False):
                state_display = gr.Code(
                    label="Current Python State",
                    language="json",
                    value=self.get_state(),
                )
                refresh_state_btn = gr.Button("Refresh State")

            # Helper function to update full status and auto-save JSON
            def update_full_status():
                full_status = self.get_full_status("")
                json_state = self.auto_save_state("")
                return full_status, json_state

            # Event handlers
            create_btn.click(
                fn=self.create_map,
                inputs=[lat_input, lon_input, zoom_input],
                outputs=[map_display],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            add_friendly_btn.click(
                fn=self.add_friendly_unit,
                inputs=[
                    friendly_name,
                    friendly_type,
                    friendly_echelon,
                    friendly_lat,
                    friendly_lon,
                    friendly_designation,
                    friendly_strength,
                ],
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            add_hostile_btn.click(
                fn=self.add_hostile_unit,
                inputs=[
                    hostile_name,
                    hostile_type,
                    hostile_echelon,
                    hostile_lat,
                    hostile_lon,
                    hostile_designation,
                    hostile_strength,
                ],
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            set_waypoints_btn.click(
                fn=self.set_unit_waypoints,
                inputs=[waypoint_unit_id, waypoints_json],
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            clear_units_btn.click(
                fn=self.clear_all_units,
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            clear_all_btn.click(
                fn=self.clear_map,
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            load_scenario_btn.click(
                fn=self.load_scenario,
                inputs=[scenario_file],
                outputs=[map_display, status_output],
            ).then(
                fn=update_full_status,
                outputs=[units_summary, state_json_display],
            )

            save_scenario_btn.click(
                fn=self.save_scenario,
                inputs=[save_scenario_name],
                outputs=[status_output],
            )

            save_html_btn.click(
                fn=self.save_map,
                inputs=[save_html_name],
                outputs=[status_output],
            )

            # Helper to update status with map state
            def update_status_with_map_state(map_state_json: str):
                full_status = self.get_full_status(map_state_json)
                json_state = self.auto_save_state(map_state_json)
                return full_status, json_state

            # Refresh Status: get current map state from stored variable
            refresh_summary_btn.click(
                fn=None,
                inputs=None,
                outputs=[map_state_holder],
                js="""
                () => {
                    if (window._wargameMapState) {
                        return window._wargameMapState;
                    }
                    return '{}';
                }
                """,
            ).then(
                fn=update_status_with_map_state,
                inputs=[map_state_holder],
                outputs=[units_summary, state_json_display],
            )

            refresh_state_btn.click(
                fn=self.get_state,
                outputs=[state_display],
            )

            # Sync button: get state from stored variable and sync to Python
            sync_btn.click(
                fn=None,
                inputs=None,
                outputs=[map_state_holder],
                js="""
                () => {
                    if (window._wargameMapState) {
                        return window._wargameMapState;
                    }
                    return '{}';
                }
                """,
            ).then(
                fn=self.sync_from_map,
                inputs=[map_state_holder],
                outputs=[map_display, status_output],
            ).then(
                fn=update_status_with_map_state,
                inputs=[map_state_holder],
                outputs=[units_summary, state_json_display],
            )

            # Helper function to update all status displays and auto-save JSON
            def update_all_status(map_state_json: str):
                full_status = self.get_full_status(map_state_json)
                brief_status = self.get_brief_status(map_state_json)
                json_state = self.auto_save_state(map_state_json)
                return full_status, brief_status, json_state

            # Event-driven status update: when JS updates the hidden textbox,
            # Gradio's change event triggers this Python callback
            # This auto-saves to JSON and updates all status displays
            map_state_holder.change(
                fn=update_all_status,
                inputs=[map_state_holder],
                outputs=[units_summary, status_output, state_json_display],
            )

            # Load JavaScript for message listener on page load
            # The JS listens for postMessage from iframe and updates the hidden textbox
            demo.load(fn=None, js=state_sync_js)

        return demo


def create_wargame_app(tiles_dir: Optional[str] = None) -> gr.Blocks:
    """Create the War Game Gradio app.

    Args:
        tiles_dir: Directory containing offline map tiles

    Returns:
        Gradio Blocks application
    """
    ui = WarGameUI(tiles_dir=tiles_dir)
    return ui.build_interface()


def main() -> None:
    """Main entry point for the War Game UI."""
    import argparse

    parser = argparse.ArgumentParser(description="War Game Map")
    parser.add_argument(
        "--tiles-dir",
        type=str,
        help="Directory containing offline map tiles",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    args = parser.parse_args()

    app = create_wargame_app(tiles_dir=args.tiles_dir)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
