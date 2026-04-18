#!/bin/bash
#
# Manage Tile Server Port
# Helper script to check, stop, or change the tile server port
#

PORT=${1:-8080}

echo "=================================================="
echo "  Tile Server Port Manager"
echo "=================================================="
echo ""
echo "Port: $PORT"
echo ""

# Function to check if port is in use
check_port() {
    if command -v lsof &> /dev/null; then
        lsof -i :$PORT &> /dev/null
        return $?
    elif command -v netstat &> /dev/null; then
        netstat -an | grep ":$PORT " | grep LISTEN &> /dev/null
        return $?
    else
        echo "⚠️  Cannot check port (lsof/netstat not found)"
        return 1
    fi
}

# Function to show process details
show_process() {
    echo "Process using port $PORT:"
    echo ""

    if command -v lsof &> /dev/null; then
        echo "=== lsof output ==="
        lsof -i :$PORT

        echo ""
        echo "=== Process details ==="
        local pid=$(lsof -ti :$PORT)
        if [ -n "$pid" ]; then
            echo "PID: $pid"
            ps -p $pid -o pid,ppid,user,comm,args
        fi
    elif command -v netstat &> /dev/null; then
        echo "=== netstat output ==="
        netstat -tulpn 2>/dev/null | grep ":$PORT "
    else
        echo "❌ No tools available to show process info"
        echo "   Install lsof or netstat"
    fi
}

# Function to kill process
kill_process() {
    echo "Stopping process on port $PORT..."
    echo ""

    if command -v lsof &> /dev/null; then
        local pid=$(lsof -ti :$PORT)
        if [ -n "$pid" ]; then
            echo "Killing PID: $pid"
            kill -9 $pid
            sleep 1

            if check_port; then
                echo "❌ Process still running"
                return 1
            else
                echo "✅ Process stopped successfully"
                return 0
            fi
        else
            echo "❌ Could not find PID"
            return 1
        fi
    else
        echo "❌ lsof not found - cannot kill process automatically"
        echo ""
        echo "Try manually:"
        echo "  sudo netstat -tulpn | grep :$PORT"
        echo "  sudo kill -9 <PID>"
        return 1
    fi
}

# Main menu
while true; do
    if check_port; then
        echo "✅ Port $PORT is IN USE"
        echo ""
        echo "Options:"
        echo "  1) Show process details"
        echo "  2) Stop the process"
        echo "  3) Exit"
        echo ""
        read -p "Choose an option (1-3): " choice

        case $choice in
            1)
                echo ""
                show_process
                echo ""
                ;;
            2)
                echo ""
                kill_process
                echo ""
                ;;
            3|*)
                exit 0
                ;;
        esac
    else
        echo "✅ Port $PORT is FREE"
        echo ""
        echo "You can now start the tile server:"
        echo "  ./scripts/start_tile_server.sh"
        echo ""
        exit 0
    fi
done
