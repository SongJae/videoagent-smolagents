#!/bin/bash
# Download map tiles for entire Korean Peninsula
# This covers both North and South Korea with surrounding areas
#
# Geographic bounds:
# - Latitude: 33°N to 43°N (entire peninsula)
# - Longitude: 124°E to 132°E (west to east coast)
#
# Zoom levels: 6-12
# - Level 6: Country overview
# - Level 8: Regional view
# - Level 10: City level detail
# - Level 12: Street level detail
#
# Note: This will download approximately 50,000-100,000 tiles (1-2 GB)
# For higher detail (zoom 13-15), increase --zoom-max but expect 10-50 GB

echo "======================================"
echo "Korean Peninsula Tile Download"
echo "======================================"
echo ""
echo "Coverage Area:"
echo "  North Boundary: 43°N (China border)"
echo "  South Boundary: 33°N (Jeju Island)"
echo "  West Boundary: 124°E (Yellow Sea)"
echo "  East Boundary: 132°E (East Sea)"
echo ""
echo "Zoom Levels: 6-12"
echo "Estimated tiles: 50,000-100,000"
echo "Estimated size: 1-2 GB"
echo ""
echo "Starting download..."
echo ""

python scripts/download_map_tiles.py \
  --lat-min 33.0 \
  --lat-max 43.0 \
  --lon-min 124.0 \
  --lon-max 132.0 \
  --zoom-min 6 \
  --zoom-max 14 \
  --output-dir ./data/map_tiles \
  --workers 16

echo ""
echo "======================================"
echo "Download Complete!"
echo "======================================"
echo ""
echo "Tiles saved to: ./data/map_tiles"
echo ""
echo "To download higher detail (zoom 13-15):"
echo "  ./scripts/download_korean_peninsula_tiles.sh --high-detail"
echo ""
