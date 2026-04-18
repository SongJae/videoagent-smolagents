#!/bin/bash
# Download Leaflet library for offline use

LIBS_DIR="./data/map_libs"
LEAFLET_VERSION="1.9.4"

echo "Creating libraries directory..."
mkdir -p "$LIBS_DIR"

echo "Downloading Leaflet ${LEAFLET_VERSION}..."
curl -L "https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.js" -o "$LIBS_DIR/leaflet.js"
curl -L "https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.css" -o "$LIBS_DIR/leaflet.css"

echo "Downloading Leaflet plugins..."
curl -L "https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js" -o "$LIBS_DIR/leaflet.markercluster.js"
curl -L "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" -o "$LIBS_DIR/MarkerCluster.css"
curl -L "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" -o "$LIBS_DIR/MarkerCluster.Default.css"

echo "Done! Leaflet libraries downloaded to $LIBS_DIR"
echo ""
echo "To use offline mode:"
echo "1. Ensure offline_mode is enabled in config/map_config.yaml"
echo "2. The system will automatically use these local files"
