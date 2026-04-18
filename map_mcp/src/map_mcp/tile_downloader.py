"""Map tile downloader for offline use.

This module provides utilities to download map tiles for a given
geographic area and zoom levels, enabling offline map usage.

IMPORTANT: This utility should be run while you have internet connectivity
to prepare tiles for offline use later.

Note: Tile servers have usage policies. This downloader includes rate limiting
to respect server limits. OpenStreetMap allows max ~1-2 requests/second.

Performance optimizations:
- Uses aiohttp for async concurrent downloads (much faster than requests)
- Supports multiple tile server subdomains for parallel downloads
- Connection pooling for TCP connection reuse
- Semaphore-based rate limiting for async code
"""

import asyncio
import math
import random
import time
import threading
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convert latitude/longitude to tile numbers.

    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        zoom: Zoom level

    Returns:
        Tuple of (x, y) tile numbers
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Convert tile numbers to latitude/longitude.

    Args:
        xtile: X tile number
        ytile: Y tile number
        zoom: Zoom level

    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_bounds(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
) -> tuple[int, int, int, int]:
    """Get tile bounds for a geographic bounding box.

    Args:
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude
        zoom: Zoom level

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) tile numbers
    """
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    return (min_x, max_x, min_y, max_y)


def count_tiles(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom_min: int,
    zoom_max: int,
) -> int:
    """Count the total number of tiles in a region.

    Args:
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude
        zoom_min: Minimum zoom level
        zoom_max: Maximum zoom level

    Returns:
        Total tile count
    """
    total = 0
    for zoom in range(zoom_min, zoom_max + 1):
        min_x, max_x, min_y, max_y = get_tile_bounds(
            min_lat, max_lat, min_lon, max_lon, zoom
        )
        total += (max_x - min_x + 1) * (max_y - min_y + 1)
    return total


class RateLimiter:
    """Thread-safe rate limiter for controlling request frequency."""

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait until the next request is allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


class TileDownloader:
    """Downloads map tiles for offline use."""

    # Common tile server URLs
    TILE_SERVERS = {
        "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "carto-light": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        "carto-dark": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "stamen-terrain": "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        "stamen-toner": "https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
    }

    def __init__(
        self,
        output_dir: str | Path,
        tile_server: str = "osm",
        user_agent: str = "OfflineMapDownloader/1.0 (offline map preparation)",
        requests_per_second: float = 1.0,
    ):
        """Initialize the tile downloader.

        Args:
            output_dir: Directory to save downloaded tiles
            tile_server: Name of tile server or custom URL template
            user_agent: User agent string for HTTP requests
            requests_per_second: Maximum requests per second (default: 1.0)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if tile_server in self.TILE_SERVERS:
            self.tile_url = self.TILE_SERVERS[tile_server]
        else:
            self.tile_url = tile_server

        self.user_agent = user_agent
        self._session = None
        self._rate_limiter = RateLimiter(requests_per_second)

    def _get_session(self):
        """Get or create a requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = self.user_agent
            # Add headers to be a polite client
            self._session.headers["Accept"] = "image/png,image/*;q=0.9,*/*;q=0.8"
            self._session.headers["Accept-Language"] = "en-US,en;q=0.5"
        return self._session

    def _download_tile_with_retry(
        self,
        z: int,
        x: int,
        y: int,
        overwrite: bool = False,
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """Download a single tile with retry logic.

        Args:
            z: Zoom level
            x: X tile number
            y: Y tile number
            overwrite: Whether to overwrite existing tiles
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, message)
        """
        tile_path = self.output_dir / str(z) / str(x) / f"{y}.png"

        if tile_path.exists() and not overwrite:
            return (True, f"Skipped {z}/{x}/{y} (exists)")

        tile_path.parent.mkdir(parents=True, exist_ok=True)

        url = self.tile_url.format(z=z, x=x, y=y)

        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self._rate_limiter.wait()

                session = self._get_session()
                response = session.get(url, timeout=30)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    wait_time = min(retry_after, 120)  # Cap at 2 minutes
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        return (False, f"Failed {z}/{x}/{y}: Rate limited after {max_retries} retries")

                # Handle server errors with retry
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()

                with open(tile_path, "wb") as f:
                    f.write(response.content)

                return (True, f"Downloaded {z}/{x}/{y}")

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff for other errors
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                return (False, f"Failed {z}/{x}/{y}: {str(e)}")

        return (False, f"Failed {z}/{x}/{y}: Max retries exceeded")

    def download_region(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        zoom_min: int = 1,
        zoom_max: int = 15,
        max_workers: int = 1,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Download all tiles for a geographic region.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            zoom_min: Minimum zoom level
            zoom_max: Maximum zoom level
            max_workers: Number of concurrent download threads (default: 1 for rate limiting)
            overwrite: Whether to overwrite existing tiles
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with download statistics
        """
        total_tiles = count_tiles(
            min_lat, max_lat, min_lon, max_lon, zoom_min, zoom_max
        )

        stats = {
            "total": total_tiles,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        # Generate all tile coordinates
        tiles = []
        for zoom in range(zoom_min, zoom_max + 1):
            min_x, max_x, min_y, max_y = get_tile_bounds(
                min_lat, max_lat, min_lon, max_lon, zoom
            )
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    tiles.append((zoom, x, y))

        # Download tiles (use single worker by default to respect rate limits)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_tile_with_retry, z, x, y, overwrite
                ): (z, x, y)
                for z, x, y in tiles
            }

            completed = 0
            for future in as_completed(futures):
                success, message = future.result()
                completed += 1

                if success:
                    if "Skipped" in message:
                        stats["skipped"] += 1
                    else:
                        stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(message)

                if progress_callback:
                    progress_callback(completed, total_tiles, message)

        return stats

    def download_around_point(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10,
        zoom_min: int = 1,
        zoom_max: int = 15,
        max_workers: int = 1,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Download tiles in a radius around a point.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Radius in kilometers
            zoom_min: Minimum zoom level
            zoom_max: Maximum zoom level
            max_workers: Number of concurrent download threads (default: 1 for rate limiting)
            overwrite: Whether to overwrite existing tiles
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with download statistics
        """
        # Approximate conversion from km to degrees
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

        return self.download_region(
            min_lat=lat - lat_offset,
            max_lat=lat + lat_offset,
            min_lon=lon - lon_offset,
            max_lon=lon + lon_offset,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            max_workers=max_workers,
            overwrite=overwrite,
            progress_callback=progress_callback,
        )


class AsyncTileDownloader:
    """Fast async tile downloader using aiohttp.

    This downloader is significantly faster than the sync version because:
    - Uses aiohttp for non-blocking I/O
    - Supports high concurrency with connection pooling
    - Uses multiple tile server subdomains for parallel downloads
    - Efficient semaphore-based rate limiting
    """

    # Tile servers with subdomain support ({s} = a, b, c)
    TILE_SERVERS = {
        "osm": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "carto-light": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        "carto-dark": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "stamen-terrain": "https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        "stamen-toner": "https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
    }

    SUBDOMAINS = ["a", "b", "c"]

    def __init__(
        self,
        output_dir: str | Path,
        tile_server: str = "osm",
        user_agent: str = "OfflineMapDownloader/1.0 (offline map preparation)",
        max_concurrent: int = 50,
        requests_per_second: float = 50.0,
    ):
        """Initialize the async tile downloader.

        Args:
            output_dir: Directory to save downloaded tiles
            tile_server: Name of tile server or custom URL template
            user_agent: User agent string for HTTP requests
            max_concurrent: Maximum concurrent connections (default: 50)
            requests_per_second: Maximum requests per second (default: 50.0)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if tile_server in self.TILE_SERVERS:
            self.tile_url_template = self.TILE_SERVERS[tile_server]
        else:
            self.tile_url_template = tile_server

        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second

    def _get_tile_url(self, z: int, x: int, y: int) -> str:
        """Get tile URL with subdomain rotation for parallel downloads."""
        subdomain = self.SUBDOMAINS[(x + y) % len(self.SUBDOMAINS)]
        return self.tile_url_template.format(s=subdomain, z=z, x=x, y=y)

    async def _download_tile(
        self,
        session,
        semaphore: asyncio.Semaphore,
        z: int,
        x: int,
        y: int,
        overwrite: bool = False,
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """Download a single tile asynchronously.

        Args:
            session: aiohttp ClientSession
            semaphore: Semaphore for rate limiting
            z: Zoom level
            x: X tile number
            y: Y tile number
            overwrite: Whether to overwrite existing tiles
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (success, message)
        """
        tile_path = self.output_dir / str(z) / str(x) / f"{y}.png"

        if tile_path.exists() and not overwrite:
            return (True, f"Skipped {z}/{x}/{y} (exists)")

        tile_path.parent.mkdir(parents=True, exist_ok=True)
        url = self._get_tile_url(z, x, y)

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    # Small delay for rate limiting within semaphore
                    await asyncio.sleep(1.0 / self.requests_per_second)

                    async with session.get(url, timeout=30) as response:
                        if response.status == 429:
                            # Rate limited - wait and retry
                            retry_after = int(response.headers.get("Retry-After", 5))
                            if attempt < max_retries - 1:
                                await asyncio.sleep(min(retry_after, 30))
                                continue
                            return (False, f"Failed {z}/{x}/{y}: Rate limited")

                        if response.status >= 500:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue

                        response.raise_for_status()
                        content = await response.read()

                        with open(tile_path, "wb") as f:
                            f.write(content)

                        return (True, f"Downloaded {z}/{x}/{y}")

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return (False, f"Failed {z}/{x}/{y}: Timeout")
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return (False, f"Failed {z}/{x}/{y}: {str(e)}")

        return (False, f"Failed {z}/{x}/{y}: Max retries exceeded")

    async def download_region_async(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        zoom_min: int = 1,
        zoom_max: int = 15,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Download all tiles for a geographic region asynchronously.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            zoom_min: Minimum zoom level
            zoom_max: Maximum zoom level
            overwrite: Whether to overwrite existing tiles
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with download statistics
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for async downloads. "
                "Install with: pip install aiohttp"
            )

        total_tiles = count_tiles(
            min_lat, max_lat, min_lon, max_lon, zoom_min, zoom_max
        )

        stats = {
            "total": total_tiles,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        # Generate all tile coordinates
        tiles = []
        for zoom in range(zoom_min, zoom_max + 1):
            min_x, max_x, min_y, max_y = get_tile_bounds(
                min_lat, max_lat, min_lon, max_lon, zoom
            )
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    tiles.append((zoom, x, y))

        # Shuffle tiles to distribute load across subdomains
        random.shuffle(tiles)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent // 3,  # Distribute across subdomains
            ttl_dns_cache=300,
        )
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "image/png,image/*;q=0.9,*/*;q=0.8",
        }

        async with aiohttp.ClientSession(
            connector=connector, headers=headers
        ) as session:
            # Create tasks for all tiles
            tasks = [
                self._download_tile(session, semaphore, z, x, y, overwrite)
                for z, x, y in tiles
            ]

            # Process with progress updates
            completed = 0
            for coro in asyncio.as_completed(tasks):
                success, message = await coro
                completed += 1

                if success:
                    if "Skipped" in message:
                        stats["skipped"] += 1
                    else:
                        stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(message)

                if progress_callback:
                    progress_callback(completed, total_tiles, message)

        return stats

    def download_region(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        zoom_min: int = 1,
        zoom_max: int = 15,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Download all tiles for a geographic region (sync wrapper).

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            zoom_min: Minimum zoom level
            zoom_max: Maximum zoom level
            overwrite: Whether to overwrite existing tiles
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with download statistics
        """
        return asyncio.run(
            self.download_region_async(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                zoom_min=zoom_min,
                zoom_max=zoom_max,
                overwrite=overwrite,
                progress_callback=progress_callback,
            )
        )

    def download_around_point(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10,
        zoom_min: int = 1,
        zoom_max: int = 15,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Download tiles in a radius around a point.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Radius in kilometers
            zoom_min: Minimum zoom level
            zoom_max: Maximum zoom level
            overwrite: Whether to overwrite existing tiles
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with download statistics
        """
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

        return self.download_region(
            min_lat=lat - lat_offset,
            max_lat=lat + lat_offset,
            min_lon=lon - lon_offset,
            max_lon=lon + lon_offset,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            overwrite=overwrite,
            progress_callback=progress_callback,
        )


def main() -> None:
    """Main entry point for the tile downloader CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download map tiles for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast download (recommended) - uses async with high concurrency
  map-mcp-download --lat 37.5665 --lon 126.9780 --radius 10 --output ./tiles --fast

  # Fast download with custom concurrency (50 concurrent, 30 req/sec)
  map-mcp-download --lat 37.5665 --lon 126.9780 --radius 100 --output ./tiles --fast --concurrent 50 --rate 30

  # Slow/safe download (respects strict rate limits)
  map-mcp-download --lat 37.5665 --lon 126.9780 --radius 10 --output ./tiles --rate 1.0

  # Download tiles for a specific bounding box
  map-mcp-download --bbox 37.4,37.7,126.8,127.2 --zoom-min 10 --zoom-max 16 --output ./tiles --fast

  # Use a different tile server
  map-mcp-download --lat 37.5665 --lon 126.9780 --server carto-light --output ./tiles --fast

Note: --fast mode uses aiohttp for ~50x faster downloads with multiple
concurrent connections and subdomain rotation. Use --rate to control
speed if you encounter rate limiting (429 errors).
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for tiles",
    )

    # Location options (either point+radius or bbox)
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        "--lat",
        type=float,
        help="Center latitude (use with --lon and --radius)",
    )
    location_group.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as min_lat,max_lat,min_lon,max_lon",
    )

    parser.add_argument(
        "--lon",
        type=float,
        help="Center longitude (use with --lat and --radius)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10,
        help="Radius in kilometers (default: 10)",
    )

    parser.add_argument(
        "--zoom-min",
        type=int,
        default=1,
        help="Minimum zoom level (default: 1)",
    )
    parser.add_argument(
        "--zoom-max",
        type=int,
        default=15,
        help="Maximum zoom level (default: 15)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="osm",
        choices=list(TileDownloader.TILE_SERVERS.keys()),
        help="Tile server to use (default: osm)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast async downloader with high concurrency (recommended)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=50,
        help="Max concurrent connections for --fast mode (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent download workers for sync mode (default: 1)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Max requests per second (default: 50 for --fast, 1 for sync)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tiles",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when using --lat")

    # Set default rate based on mode
    if args.rate is None:
        args.rate = 50.0 if args.fast else 1.0

    # Check aiohttp availability for fast mode
    if args.fast:
        try:
            import aiohttp
        except ImportError:
            print("Error: aiohttp is required for --fast mode.")
            print("Install with: pip install aiohttp")
            print("Falling back to sync mode...")
            args.fast = False

    # Import tqdm for progress bar (optional dependency)
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Create appropriate downloader
    if args.fast:
        downloader = AsyncTileDownloader(
            output_dir=args.output,
            tile_server=args.server,
            max_concurrent=args.concurrent,
            requests_per_second=args.rate,
        )
        mode_str = f"FAST async mode ({args.concurrent} concurrent, {args.rate} req/sec)"
    else:
        downloader = TileDownloader(
            output_dir=args.output,
            tile_server=args.server,
            requests_per_second=args.rate,
        )
        mode_str = f"sync mode ({args.workers} workers, {args.rate} req/sec)"

    # Calculate total tiles first
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        min_lat, max_lat, min_lon, max_lon = parts
    else:
        lat_offset = args.radius / 111.0
        lon_offset = args.radius / (111.0 * math.cos(math.radians(args.lat)))
        min_lat = args.lat - lat_offset
        max_lat = args.lat + lat_offset
        min_lon = args.lon - lon_offset
        max_lon = args.lon + lon_offset

    total = count_tiles(
        min_lat, max_lat, min_lon, max_lon,
        args.zoom_min, args.zoom_max
    )

    # Estimate download time
    effective_rate = min(args.rate, args.concurrent) if args.fast else args.rate
    estimated_seconds = total / effective_rate
    estimated_minutes = estimated_seconds / 60

    print(f"Download mode: {mode_str}")
    print(f"Total tiles to download: {total}")
    print(f"Estimated time: {estimated_minutes:.1f} minutes")
    print()

    # Progress callback
    if use_tqdm:
        pbar = tqdm(total=total, unit="tiles")

        def progress(current, total, message):
            pbar.update(1)
            pbar.set_description(message[:50])
    else:
        def progress(current, total, message):
            if current % 100 == 0 or current == total:
                print(f"Progress: {current}/{total} - {message}")

    # Download tiles
    if args.fast:
        # Async downloader doesn't use max_workers
        if args.bbox:
            stats = downloader.download_region(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                zoom_min=args.zoom_min,
                zoom_max=args.zoom_max,
                overwrite=args.overwrite,
                progress_callback=progress,
            )
        else:
            stats = downloader.download_around_point(
                lat=args.lat,
                lon=args.lon,
                radius_km=args.radius,
                zoom_min=args.zoom_min,
                zoom_max=args.zoom_max,
                overwrite=args.overwrite,
                progress_callback=progress,
            )
    else:
        # Sync downloader uses max_workers
        if args.bbox:
            stats = downloader.download_region(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                zoom_min=args.zoom_min,
                zoom_max=args.zoom_max,
                max_workers=args.workers,
                overwrite=args.overwrite,
                progress_callback=progress,
            )
        else:
            stats = downloader.download_around_point(
                lat=args.lat,
                lon=args.lon,
                radius_km=args.radius,
                zoom_min=args.zoom_min,
                zoom_max=args.zoom_max,
                max_workers=args.workers,
                overwrite=args.overwrite,
                progress_callback=progress,
            )

    if use_tqdm:
        pbar.close()

    print("\nDownload complete!")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")

    if stats["errors"]:
        print("\nErrors:")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")


if __name__ == "__main__":
    main()
