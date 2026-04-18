#!/bin/bash
#
# Start Local Tile Server for Offline Map Operation
# Serves cached map tiles on localhost:8080
#

# Don't exit on error immediately - we'll handle errors ourselves
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TILES_DIR="${PROJECT_ROOT}/data/map_tiles"
DEFAULT_PORT=8080
SERVER_PORT=${1:-$DEFAULT_PORT}  # Allow port override via argument

echo "=================================================="
echo "  Local Tile Server for Battlefield Map"
echo "=================================================="
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -i :$port &> /dev/null
        return $?
    elif command -v netstat &> /dev/null; then
        netstat -an | grep ":$port " | grep LISTEN &> /dev/null
        return $?
    elif command -v ss &> /dev/null; then
        ss -ln | grep ":$port " &> /dev/null
        return $?
    else
        # Can't check - assume port is free
        return 1
    fi
}

# Function to get process using port
get_port_process() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -i :$port | tail -n +2 | awk '{print $1, $2}' | head -1
    elif command -v netstat &> /dev/null; then
        netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | head -1
    else
        echo "unknown unknown"
    fi
}

# Function to kill process on port
kill_port_process() {
    local port=$1
    echo "Attempting to stop process on port $port..."

    if command -v lsof &> /dev/null; then
        local pid=$(lsof -ti :$port)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null
            sleep 1
            if check_port $port; then
                echo "❌ Failed to stop process"
                return 1
            else
                echo "✅ Process stopped successfully"
                return 0
            fi
        fi
    fi

    return 1
}

# Check if port is in use
if check_port $SERVER_PORT; then
    echo "⚠️  Port $SERVER_PORT is already in use!"
    echo ""

    # Get process info
    process_info=$(get_port_process $SERVER_PORT)
    if [ -n "$process_info" ]; then
        echo "Process using port $SERVER_PORT:"
        echo "  $process_info"
    fi
    echo ""

    # Ask user what to do
    echo "Options:"
    echo "  1) Kill the existing process and restart"
    echo "  2) Use a different port (8081)"
    echo "  3) Cancel"
    echo ""
    read -p "Choose an option (1-3): " choice

    case $choice in
        1)
            if kill_port_process $SERVER_PORT; then
                echo ""
                echo "Continuing with port $SERVER_PORT..."
                echo ""
            else
                echo ""
                echo "❌ Could not stop the process. Try manually:"
                echo "   lsof -ti :$SERVER_PORT | xargs kill -9"
                echo ""
                exit 1
            fi
            ;;
        2)
            SERVER_PORT=8081
            echo ""
            echo "Using alternative port: $SERVER_PORT"
            echo "⚠️  Remember to update config/map_config.yaml:"
            echo "   url: \"http://localhost:$SERVER_PORT/{z}/{x}/{y}.png\""
            echo ""
            # Check if alternative port is also in use
            if check_port $SERVER_PORT; then
                echo "❌ Port $SERVER_PORT is also in use!"
                echo "   Please stop the process manually or use a different port"
                exit 1
            fi
            ;;
        3|*)
            echo ""
            echo "Cancelled."
            exit 0
            ;;
    esac
fi

echo "Using port: $SERVER_PORT"
echo ""

# Check if tiles directory exists
if [ ! -d "$TILES_DIR" ]; then
    echo "❌ Error: Tiles directory not found: $TILES_DIR"
    echo ""
    echo "To cache tiles for offline use, run:"
    echo "  python scripts/download_map_tiles.py"
    echo ""
    exit 1
fi

# Check if tiles exist
tile_count=$(find "$TILES_DIR" -name "*.png" 2>/dev/null | wc -l)
if [ "$tile_count" -eq 0 ]; then
    echo "⚠️  Warning: No tiles found in $TILES_DIR"
    echo ""
    echo "To download tiles, run:"
    echo "  python scripts/download_map_tiles.py"
    echo ""
    echo "Continuing anyway..."
    echo ""
fi

echo "📁 Tiles directory: $TILES_DIR"
echo "🗺️  Cached tiles: $tile_count"
echo ""

# Method 1: Try Docker with TileServer GL (best option)
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected - trying TileServer GL..."
    echo ""
    echo "Starting TileServer GL on http://localhost:$SERVER_PORT"
    echo "Press Ctrl+C to stop the server"
    echo ""

    # Check if tileserver-gl image exists
    if docker images | grep -q "maptiler/tileserver-gl"; then
        :  # Image exists
    else
        echo "Pulling TileServer GL image..."
        docker pull maptiler/tileserver-gl
    fi

    # Start TileServer GL
    docker run --rm \
        -it \
        -v "${TILES_DIR}:/data" \
        -p ${SERVER_PORT}:8080 \
        maptiler/tileserver-gl \
        --verbose
    exit 0
fi

# Method 2: Try Node.js with http-server
if command -v node &> /dev/null && command -v npx &> /dev/null; then
    echo "📦 Node.js detected - using http-server..."
    echo ""
    echo "Starting HTTP server on http://localhost:$SERVER_PORT"
    echo "Press Ctrl+C to stop the server"
    echo ""

    cd "$TILES_DIR"
    npx http-server -p $SERVER_PORT --cors
    exit 0
fi

# Method 3: Python HTTP server (fallback)
if command -v python3 &> /dev/null; then
    echo "🐍 Python detected - using built-in HTTP server..."
    echo ""
    echo "Starting Python HTTP server on http://localhost:$SERVER_PORT"
    echo "Press Ctrl+C to stop the server"
    echo ""

    cd "$TILES_DIR"
    python3 -m http.server $SERVER_PORT
    exit 0
fi

# No suitable server found
echo "❌ Error: No suitable HTTP server found"
echo ""
echo "Please install one of the following:"
echo "  • Docker (recommended): https://docs.docker.com/get-docker/"
echo "  • Node.js: https://nodejs.org/"
echo "  • Python 3: https://www.python.org/"
echo ""
exit 1
