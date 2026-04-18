#!/bin/bash
#
# Test script for offline map functionality
# Verifies that offline map system is properly configured and working
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  Offline Map System Test"
echo "=================================================="
echo ""

# Test 1: Check configuration file
echo "[1/6] Checking configuration..."
CONFIG_FILE="${PROJECT_ROOT}/config/map_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Configuration file exists: $CONFIG_FILE"

    # Check if offline mode is enabled
    if grep -q "enabled: true" "$CONFIG_FILE"; then
        echo "✅ Offline mode is enabled"
    else
        echo "⚠️  Offline mode is disabled (set enabled: true in config)"
    fi
else
    echo "❌ Configuration file missing: $CONFIG_FILE"
    exit 1
fi
echo ""

# Test 2: Check tiles directory
echo "[2/6] Checking cached tiles..."
TILES_DIR="${PROJECT_ROOT}/data/map_tiles"
if [ -d "$TILES_DIR" ]; then
    tile_count=$(find "$TILES_DIR" -name "*.png" 2>/dev/null | wc -l)
    tile_size=$(du -sh "$TILES_DIR" 2>/dev/null | cut -f1)

    if [ "$tile_count" -gt 0 ]; then
        echo "✅ Cached tiles found: $tile_count tiles ($tile_size)"

        # List zoom levels
        zoom_levels=$(ls -1 "$TILES_DIR" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
        echo "   Zoom levels: $zoom_levels"
    else
        echo "⚠️  No tiles cached yet"
        echo "   Run: python scripts/download_map_tiles.py"
    fi
else
    echo "⚠️  Tiles directory doesn't exist: $TILES_DIR"
    echo "   Run: python scripts/download_map_tiles.py"
fi
echo ""

# Test 3: Check if tile server is running
echo "[3/6] Checking tile server..."
if curl -f -s http://localhost:8080/ > /dev/null 2>&1; then
    echo "✅ Tile server is running on port 8080"
else
    echo "❌ Tile server not running on port 8080"
    echo "   Start with: ./scripts/start_tile_server.sh"
fi
echo ""

# Test 4: Test tile accessibility
echo "[4/6] Testing tile access..."
if [ "$tile_count" -gt 0 ]; then
    # Find a random tile to test
    sample_tile=$(find "$TILES_DIR" -name "*.png" | head -1)
    if [ -n "$sample_tile" ]; then
        # Extract z/x/y from path
        rel_path="${sample_tile#$TILES_DIR/}"

        # Try to access via server
        if curl -f -s http://localhost:8080/$rel_path > /dev/null 2>&1; then
            echo "✅ Sample tile accessible via server"
        else
            echo "⚠️  Cannot access tile via server (but file exists)"
            echo "   Tile: $rel_path"
        fi
    fi
else
    echo "⏭️  Skipped (no tiles to test)"
fi
echo ""

# Test 5: Check Python dependencies
echo "[5/6] Checking Python dependencies..."
missing_deps=0

for package in folium yaml tqdm requests; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package installed"
    else
        echo "❌ $package not found"
        missing_deps=$((missing_deps + 1))
    fi
done

if [ $missing_deps -gt 0 ]; then
    echo ""
    echo "Install missing dependencies:"
    echo "  pip install -r requirements.txt"
fi
echo ""

# Test 6: Check map generator initialization
echo "[6/6] Testing map generator..."
cd "$PROJECT_ROOT"
python3 << 'EOF'
import sys
try:
    from core_src.battlefield_map import BattlefieldMapGenerator

    # Try to initialize
    gen = BattlefieldMapGenerator()

    print(f"✅ Map generator initialized")
    print(f"   Offline mode: {gen.offline_mode}")
    print(f"   Local tiles available: {gen.local_tiles_available}")

    # Try to create a simple map
    m = gen.create_base_map()
    print(f"✅ Base map created successfully")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
else
    echo ""
    echo "Check Python errors above"
fi

# Summary
echo "=================================================="
echo "  Test Summary"
echo "=================================================="
echo ""

# Determine overall status
if [ "$tile_count" -gt 0 ] && curl -f -s http://localhost:8080/ > /dev/null 2>&1; then
    echo "✅ System ready for offline operation"
    echo ""
    echo "Next steps:"
    echo "  1. python main.py ui"
    echo "  2. Navigate to Query Agent tab"
    echo "  3. Click 'Generate Map'"
    echo ""
elif [ "$tile_count" -gt 0 ]; then
    echo "⚠️  Tiles cached but server not running"
    echo ""
    echo "Start tile server:"
    echo "  ./scripts/start_tile_server.sh"
    echo ""
else
    echo "⚠️  System not ready - tiles need to be downloaded"
    echo ""
    echo "Download tiles:"
    echo "  python scripts/download_map_tiles.py \\"
    echo "    --lat-min 37.50 --lat-max 37.65 \\"
    echo "    --lon-min 126.85 --lon-max 127.10 \\"
    echo "    --zoom-min 10 --zoom-max 16"
    echo ""
    echo "Then start tile server:"
    echo "  ./scripts/start_tile_server.sh"
    echo ""
fi

echo "For detailed guide, see: OFFLINE_MAP_SETUP.md"
echo "For quick start, see: OFFLINE_MAP_QUICKSTART.md"
echo ""
