"""Gradio Web UI for offline map rendering and interaction.

This module provides a Gradio-based web interface for creating
and manipulating maps in an offline environment.
"""

import base64
import json
from pathlib import Path
from typing import Any, Optional
import gradio as gr

from .map_utils import OfflineMap


class MapUI:
    """Gradio-based map user interface."""

    def __init__(self, tiles_dir: Optional[str | Path] = None):
        """Initialize the map UI.

        Args:
            tiles_dir: Directory containing offline map tiles
        """
        self.tiles_dir = Path(tiles_dir) if tiles_dir else None
        self.map = OfflineMap(tiles_dir=self.tiles_dir)

    def _create_iframe_html(self, map_html: str) -> str:
        """Wrap map HTML in an iframe for proper rendering in Gradio.

        Args:
            map_html: The full HTML document for the map

        Returns:
            HTML with iframe embedding the map
        """
        # Encode the HTML as base64 for use in iframe src
        encoded = base64.b64encode(map_html.encode('utf-8')).decode('utf-8')
        return f'''
        <iframe
            srcdoc="{map_html.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')}"
            style="width: 100%; height: 650px; border: none; border-radius: 5px;"
            sandbox="allow-scripts allow-same-origin"
        ></iframe>
        '''

    def _create_data_uri_iframe(self, map_html: str) -> str:
        """Create an iframe using data URI for the map HTML.

        Args:
            map_html: The full HTML document for the map

        Returns:
            HTML with iframe using data URI
        """
        encoded = base64.b64encode(map_html.encode('utf-8')).decode('utf-8')
        return f'''
        <iframe
            src="data:text/html;base64,{encoded}"
            style="width: 100%; height: 650px; border: none; border-radius: 5px;"
            sandbox="allow-scripts allow-same-origin"
        ></iframe>
        '''

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
        self.map = OfflineMap(
            center=(lat, lon),
            zoom=zoom,
            tiles_dir=self.tiles_dir,
        )
        return self._create_data_uri_iframe(self.map.to_html())

    def add_marker(
        self,
        lat: float,
        lon: float,
        popup: str,
        tooltip: str,
        color: str,
    ) -> str:
        """Add a marker to the map.

        Args:
            lat: Marker latitude
            lon: Marker longitude
            popup: Popup content
            tooltip: Tooltip text
            color: Marker color

        Returns:
            Updated map HTML
        """
        self.map.add_marker(
            location=(lat, lon),
            popup=popup if popup else None,
            tooltip=tooltip if tooltip else None,
            color=color,
        )
        return self._create_data_uri_iframe(self.map.to_html())

    def add_polyline(
        self,
        points_json: str,
        color: str,
        weight: int,
    ) -> str:
        """Add a polyline to the map.

        Args:
            points_json: JSON array of [lat, lon] points
            color: Line color
            weight: Line weight

        Returns:
            Updated map HTML
        """
        try:
            points = json.loads(points_json)
            self.map.add_polyline(
                locations=[tuple(p) for p in points],
                color=color,
                weight=weight,
            )
        except json.JSONDecodeError as e:
            return f"<p style='color: red;'>Error parsing points: {e}</p>"
        return self._create_data_uri_iframe(self.map.to_html())

    def add_polygon(
        self,
        points_json: str,
        color: str,
        fill_color: str,
        fill_opacity: float,
    ) -> str:
        """Add a polygon to the map.

        Args:
            points_json: JSON array of [lat, lon] vertices
            color: Border color
            fill_color: Fill color
            fill_opacity: Fill opacity

        Returns:
            Updated map HTML
        """
        try:
            points = json.loads(points_json)
            self.map.add_polygon(
                locations=[tuple(p) for p in points],
                color=color,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
            )
        except json.JSONDecodeError as e:
            return f"<p style='color: red;'>Error parsing points: {e}</p>"
        return self._create_data_uri_iframe(self.map.to_html())

    def add_circle(
        self,
        lat: float,
        lon: float,
        radius: float,
        color: str,
        fill_opacity: float,
    ) -> str:
        """Add a circle to the map.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius: Radius in meters
            color: Circle color
            fill_opacity: Fill opacity

        Returns:
            Updated map HTML
        """
        self.map.add_circle(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill_opacity=fill_opacity,
        )
        return self._create_data_uri_iframe(self.map.to_html())

    def load_geojson(
        self,
        file,
        color: str,
        fill_opacity: float,
    ) -> str:
        """Load a GeoJSON file onto the map.

        Args:
            file: Uploaded file object
            color: Feature color
            fill_opacity: Fill opacity

        Returns:
            Updated map HTML
        """
        if file is None:
            return self._create_data_uri_iframe(self.map.to_html())

        try:
            with open(file.name, 'r') as f:
                data = json.load(f)
            self.map.add_geojson(
                data=data,
                style={
                    "color": color,
                    "fillColor": color,
                    "fillOpacity": fill_opacity,
                    "weight": 2,
                },
            )
        except Exception as e:
            return f"<p style='color: red;'>Error loading GeoJSON: {e}</p>"
        return self._create_data_uri_iframe(self.map.to_html())

    def clear_map(self) -> str:
        """Clear all map elements.

        Returns:
            Updated map HTML
        """
        self.map.clear()
        return self._create_data_uri_iframe(self.map.to_html())

    def get_state(self) -> str:
        """Get current map state as JSON.

        Returns:
            JSON string of map state
        """
        return json.dumps(self.map.get_state(), indent=2)

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

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Offline Map Viewer") as demo:
            gr.Markdown("# Offline Interactive Map Viewer")
            gr.Markdown("""
            Create and interact with maps without internet connectivity.
            - **Mouse scroll**: Zoom in/out
            - **Click & drag**: Pan the map
            - **Add Marker button**: Click on map to place markers
            - **Add Waypoint button**: Click to create a route with connected waypoints
            """)

            with gr.Row():
                # Left panel - Controls
                with gr.Column(scale=1):
                    with gr.Accordion("Map Settings", open=True):
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

                    with gr.Accordion("Add Marker (via form)", open=False):
                        marker_lat = gr.Number(label="Latitude", value=37.5665)
                        marker_lon = gr.Number(label="Longitude", value=126.9780)
                        marker_popup = gr.Textbox(label="Popup Text")
                        marker_tooltip = gr.Textbox(label="Tooltip")
                        marker_color = gr.Dropdown(
                            label="Color",
                            choices=["blue", "red", "green", "orange", "purple", "darkblue", "darkgreen", "cadetblue", "darkred", "lightred", "beige", "darkpurple", "pink", "lightblue", "lightgreen", "gray", "black", "lightgray"],
                            value="blue",
                        )
                        add_marker_btn = gr.Button("Add Marker")

                    with gr.Accordion("Add Polyline", open=False):
                        polyline_points = gr.Textbox(
                            label="Points (JSON)",
                            placeholder='[[37.56, 126.97], [37.57, 126.98], [37.58, 126.99]]',
                            lines=3,
                        )
                        polyline_color = gr.ColorPicker(label="Color", value="#0000FF")
                        polyline_weight = gr.Slider(
                            label="Weight",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                        )
                        add_polyline_btn = gr.Button("Add Polyline")

                    with gr.Accordion("Add Polygon", open=False):
                        polygon_points = gr.Textbox(
                            label="Points (JSON)",
                            placeholder='[[37.56, 126.97], [37.57, 126.98], [37.56, 126.99]]',
                            lines=3,
                        )
                        polygon_color = gr.ColorPicker(label="Border Color", value="#0000FF")
                        polygon_fill_color = gr.ColorPicker(label="Fill Color", value="#0000FF")
                        polygon_fill_opacity = gr.Slider(
                            label="Fill Opacity",
                            minimum=0,
                            maximum=1,
                            value=0.3,
                            step=0.1,
                        )
                        add_polygon_btn = gr.Button("Add Polygon")

                    with gr.Accordion("Add Circle", open=False):
                        circle_lat = gr.Number(label="Center Latitude", value=37.5665)
                        circle_lon = gr.Number(label="Center Longitude", value=126.9780)
                        circle_radius = gr.Number(label="Radius (meters)", value=1000)
                        circle_color = gr.ColorPicker(label="Color", value="#0000FF")
                        circle_fill_opacity = gr.Slider(
                            label="Fill Opacity",
                            minimum=0,
                            maximum=1,
                            value=0.3,
                            step=0.1,
                        )
                        add_circle_btn = gr.Button("Add Circle")

                    with gr.Accordion("Load GeoJSON", open=False):
                        geojson_file = gr.File(
                            label="GeoJSON File",
                            file_types=[".json", ".geojson"],
                        )
                        geojson_color = gr.ColorPicker(label="Color", value="#0000FF")
                        geojson_fill_opacity = gr.Slider(
                            label="Fill Opacity",
                            minimum=0,
                            maximum=1,
                            value=0.3,
                            step=0.1,
                        )
                        load_geojson_btn = gr.Button("Load GeoJSON")

                    with gr.Accordion("Actions", open=True):
                        clear_btn = gr.Button("Clear All Shapes", variant="secondary")
                        with gr.Row():
                            save_filename = gr.Textbox(
                                label="Filename",
                                placeholder="map.html",
                            )
                            save_btn = gr.Button("Save Map")
                        save_status = gr.Textbox(label="Status", interactive=False)

                # Right panel - Map display
                with gr.Column(scale=2):
                    map_display = gr.HTML(
                        label="Map",
                        value=self._create_data_uri_iframe(self.map.to_html()),
                    )

            with gr.Accordion("Map State (JSON)", open=False):
                state_display = gr.Code(
                    label="Current State",
                    language="json",
                    value=self.get_state(),
                )
                refresh_state_btn = gr.Button("Refresh State")

            # Event handlers
            create_btn.click(
                fn=self.create_map,
                inputs=[lat_input, lon_input, zoom_input],
                outputs=[map_display],
            )

            add_marker_btn.click(
                fn=self.add_marker,
                inputs=[marker_lat, marker_lon, marker_popup, marker_tooltip, marker_color],
                outputs=[map_display],
            )

            add_polyline_btn.click(
                fn=self.add_polyline,
                inputs=[polyline_points, polyline_color, polyline_weight],
                outputs=[map_display],
            )

            add_polygon_btn.click(
                fn=self.add_polygon,
                inputs=[polygon_points, polygon_color, polygon_fill_color, polygon_fill_opacity],
                outputs=[map_display],
            )

            add_circle_btn.click(
                fn=self.add_circle,
                inputs=[circle_lat, circle_lon, circle_radius, circle_color, circle_fill_opacity],
                outputs=[map_display],
            )

            load_geojson_btn.click(
                fn=self.load_geojson,
                inputs=[geojson_file, geojson_color, geojson_fill_opacity],
                outputs=[map_display],
            )

            clear_btn.click(
                fn=self.clear_map,
                outputs=[map_display],
            )

            save_btn.click(
                fn=self.save_map,
                inputs=[save_filename],
                outputs=[save_status],
            )

            refresh_state_btn.click(
                fn=self.get_state,
                outputs=[state_display],
            )

        return demo


def create_app(tiles_dir: Optional[str] = None) -> gr.Blocks:
    """Create the Gradio app.

    Args:
        tiles_dir: Directory containing offline map tiles

    Returns:
        Gradio Blocks application
    """
    ui = MapUI(tiles_dir=tiles_dir)
    return ui.build_interface()


def main() -> None:
    """Main entry point for the Gradio UI."""
    import argparse

    parser = argparse.ArgumentParser(description="Offline Map Viewer")
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
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    args = parser.parse_args()

    app = create_app(tiles_dir=args.tiles_dir)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
