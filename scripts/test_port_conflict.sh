#!/bin/bash
#
# Test script for port conflict resolution
# Demonstrates the fix for EADDRINUSE error
#

echo "=================================================="
echo "  Port Conflict Test"
echo "=================================================="
echo ""

# Check if port 8080 is in use
if lsof -i :8080 &> /dev/null; then
    echo "✅ Port 8080 is currently IN USE"
    echo ""
    echo "Process using port 8080:"
    lsof -i :8080 | tail -n +2
    echo ""
    echo "This is perfect for testing the fix!"
    echo ""
    echo "When you run ./scripts/start_tile_server.sh, you should see:"
    echo "  - Detection of port conflict"
    echo "  - Process information"
    echo "  - Options to resolve (kill/change port/cancel)"
    echo ""
else
    echo "ℹ️  Port 8080 is currently FREE"
    echo ""
    echo "To test the port conflict fix, first start something on port 8080:"
    echo ""
    echo "# In Terminal 1:"
    echo "python3 -m http.server 8080"
    echo ""
    echo "# Then in Terminal 2:"
    echo "./scripts/start_tile_server.sh"
    echo ""
    echo "You should see port conflict detection and resolution options."
    echo ""
fi

echo "=================================================="
echo "  Testing Port Check Functions"
echo "=================================================="
echo ""

# Test different port checking methods
echo "Available port checking tools:"
echo ""

if command -v lsof &> /dev/null; then
    echo "✅ lsof - available"
else
    echo "❌ lsof - not found"
fi

if command -v netstat &> /dev/null; then
    echo "✅ netstat - available"
else
    echo "❌ netstat - not found"
fi

if command -v ss &> /dev/null; then
    echo "✅ ss - available"
else
    echo "❌ ss - not found"
fi

echo ""
echo "The script will use the first available tool."
echo ""

echo "=================================================="
echo "  Quick Test Commands"
echo "=================================================="
echo ""
echo "1. Check if port is in use:"
echo "   lsof -i :8080"
echo ""
echo "2. Start tile server (with automatic conflict resolution):"
echo "   ./scripts/start_tile_server.sh"
echo ""
echo "3. Start on alternative port:"
echo "   ./scripts/start_tile_server.sh 8081"
echo ""
echo "4. Interactive port management:"
echo "   ./scripts/manage_tile_server_port.sh"
echo ""
echo "5. Find and kill process:"
echo "   kill -9 \$(lsof -ti :8080)"
echo ""
