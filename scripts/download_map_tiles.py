#!/usr/bin/env python3
"""
Download and cache map tiles for offline use
Downloads tiles for a specified geographic area and zoom levels
"""

import os
import sys
import math
import time
import requests
from pathlib import Path
from typing import Tuple, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class TileDownloader:
    """Download map tiles for offline use"""

    def __init__(self, tile_url: str, output_dir: str, max_workers: int = 4):
        """
        Initialize tile downloader

        Args:
            tile_url: Tile server URL template with {z}, {x}, {y} placeholders
            output_dir: Directory to save tiles
            max_workers: Number of parallel download threads
        """
        self.tile_url = tile_url
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Offline Map Cache)'
        })

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """
        Convert lat/lon to tile coordinates

        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level

        Returns:
            Tuple of (tile_x, tile_y)
        """
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return tile_x, tile_y

    def get_tile_bounds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        zoom: int
    ) -> Tuple[int, int, int, int]:
        """
        Get tile coordinate bounds for a geographic area

        Args:
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude
            zoom: Zoom level

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        x_min, y_max = self.lat_lon_to_tile(lat_min, lon_min, zoom)
        x_max, y_min = self.lat_lon_to_tile(lat_max, lon_max, zoom)
        return x_min, x_max, y_min, y_max

    def download_tile(self, z: int, x: int, y: int) -> bool:
        """
        Download a single tile

        Args:
            z: Zoom level
            x: Tile X coordinate
            y: Tile Y coordinate

        Returns:
            True if successful, False otherwise
        """
        # Create output directory
        tile_dir = self.output_dir / str(z) / str(x)
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Check if tile already exists
        tile_path = tile_dir / f"{y}.png"
        if tile_path.exists():
            return True

        # Download tile
        url = self.tile_url.format(z=z, x=x, y=y)

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Save tile
            with open(tile_path, 'wb') as f:
                f.write(response.content)

            # Rate limiting - be nice to tile servers
            time.sleep(0.1)
            return True

        except Exception as e:
            print(f"Error downloading tile {z}/{x}/{y}: {e}", file=sys.stderr)
            return False

    def download_area(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        zoom_levels: List[int]
    ) -> Tuple[int, int]:
        """
        Download tiles for a geographic area

        Args:
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude
            zoom_levels: List of zoom levels to download

        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        total_tiles = 0
        tiles_to_download = []

        # Calculate tiles needed
        for zoom in zoom_levels:
            x_min, x_max, y_min, y_max = self.get_tile_bounds(
                lat_min, lat_max, lon_min, lon_max, zoom
            )

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tiles_to_download.append((zoom, x, y))
                    total_tiles += 1

        print(f"Downloading {total_tiles} tiles across {len(zoom_levels)} zoom levels...")
        print(f"Tile bounds: {lat_min}°N to {lat_max}°N, {lon_min}°E to {lon_max}°E")

        # Download tiles in parallel
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_tile, z, x, y): (z, x, y)
                for z, x, y in tiles_to_download
            }

            with tqdm(total=total_tiles, desc="Downloading tiles") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                    pbar.update(1)

        return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download map tiles for offline use"
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=37.50,
        help="Minimum latitude (default: 37.50 - Seoul area)"
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=37.65,
        help="Maximum latitude (default: 37.65 - Seoul area)"
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=126.85,
        help="Minimum longitude (default: 126.85 - Seoul area)"
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=127.10,
        help="Maximum longitude (default: 127.10 - Seoul area)"
    )
    parser.add_argument(
        "--zoom-min",
        type=int,
        default=10,
        help="Minimum zoom level (default: 10)"
    )
    parser.add_argument(
        "--zoom-max",
        type=int,
        default=16,
        help="Maximum zoom level (default: 16)"
    )
    parser.add_argument(
        "--tile-url",
        type=str,
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="Tile server URL template"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/map_tiles",
        help="Output directory for tiles"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads (default: 4)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create downloader
    downloader = TileDownloader(
        tile_url=args.tile_url,
        output_dir=output_dir,
        max_workers=args.workers
    )

    # Generate zoom levels
    zoom_levels = list(range(args.zoom_min, args.zoom_max + 1))

    # Download tiles
    print(f"Tile server: {args.tile_url}")
    print(f"Output directory: {output_dir}")
    print(f"Zoom levels: {zoom_levels}")

    start_time = time.time()
    successful, failed = downloader.download_area(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        zoom_levels=zoom_levels
    )
    elapsed = time.time() - start_time

    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"\nTiles saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
