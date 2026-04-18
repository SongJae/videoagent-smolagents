"""Offline Map MCP Server with Gradio UI.

This package provides:
- Offline map rendering using Leaflet.js
- War game support with NATO military symbols (milsymbol)
- MCP server for map and war game operations
- Gradio web UI for interactive map manipulation
"""

__version__ = "0.1.0"

from .map_utils import OfflineMap
from .wargame_map import WarGameMap
from .military_symbols import (
    Affiliation,
    Echelon,
    UnitType,
    MilitaryUnit,
    generate_sidc,
    parse_sidc,
)

__all__ = [
    "OfflineMap",
    "WarGameMap",
    "Affiliation",
    "Echelon",
    "UnitType",
    "MilitaryUnit",
    "generate_sidc",
    "parse_sidc",
]
