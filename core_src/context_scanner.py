"""
Context Scanner: Discover available video and PDF contexts for multi-context agent querying
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import os


class ContextScanner:
    """
    Scanner for discovering available video and PDF contexts
    """

    def __init__(self,
                 videodb_path: str = "./data/local_videodb",
                 pdf_base_path: str = "./data/pdf_documents"):
        """
        Initialize context scanner

        Args:
            videodb_path: Path to local_videodb storage
            pdf_base_path: Base path for PDF document storage
        """
        self.videodb_path = Path(videodb_path)
        self.pdf_base_path = Path(pdf_base_path)
        self.pdf_metadata_path = self.pdf_base_path / "metadata"
        self.rag_index_path = self.pdf_base_path / "rag_index"

    def list_collections(self) -> List[Dict]:
        """
        List all available collections

        Returns:
            List of collection dictionaries with metadata
        """
        collections = []

        try:
            collections_path = self.videodb_path / "collections"

            if not collections_path.exists():
                print(f"Collections path does not exist: {collections_path}")
                return collections

            # Iterate through collection directories
            for coll_dir in collections_path.iterdir():
                if not coll_dir.is_dir():
                    continue

                # Load collection metadata
                metadata_file = coll_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Count videos
                    video_count = len(metadata.get("videos", []))

                    collection_info = {
                        "id": metadata.get("id", coll_dir.name),
                        "name": metadata.get("name", coll_dir.name),
                        "description": metadata.get("description", ""),
                        "video_count": video_count,
                        "created_at": metadata.get("created_at", "")
                    }

                    collections.append(collection_info)

                except Exception as e:
                    print(f"Error loading collection metadata for {coll_dir.name}: {e}")
                    continue

            # Sort by name
            collections.sort(key=lambda x: x["name"])

            print(f"Found {len(collections)} collections")

        except Exception as e:
            print(f"Error listing collections: {e}")

        return collections

    def scan_available_videos(self, collection_id: str = "battlefield_reconnaissance", collection_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan local_videodb for all analyzed videos

        Args:
            collection_id: Single collection to scan (default: battlefield_reconnaissance)
            collection_ids: Multiple collections to scan (overrides collection_id if provided)

        Returns:
            List of video context dictionaries with metadata
        """
        # Determine which collections to scan
        if collection_ids is not None:
            collections_to_scan = collection_ids
        else:
            collections_to_scan = [collection_id]

        all_videos = []

        for coll_id in collections_to_scan:
            videos = self._scan_collection_videos(coll_id)
            all_videos.extend(videos)

        # Sort by collection then name
        all_videos.sort(key=lambda x: (x["collection_id"], x["name"]))

        return all_videos

    def _scan_collection_videos(self, collection_id: str) -> List[Dict]:
        """
        Scan videos from a single collection

        Args:
            collection_id: Collection to scan

        Returns:
            List of video context dictionaries with metadata
        """
        videos = []

        try:
            # Path to collection videos
            collection_path = self.videodb_path / "collections" / collection_id / "videos"

            if not collection_path.exists():
                print(f"Collection path does not exist: {collection_path}")
                return videos

            # Iterate through video directories
            for video_dir in collection_path.iterdir():
                if not video_dir.is_dir():
                    continue

                # Load video metadata
                metadata_file = video_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract key information
                    video_info = {
                        "video_id": metadata.get("id", video_dir.name),
                        "name": metadata.get("name", "Untitled Video"),
                        "description": metadata.get("description", ""),
                        "length": metadata.get("length", 0.0),
                        "num_segments": len(metadata.get("segments_metadata", {})),
                        "file_path": metadata.get("file_path", ""),
                        "thumbnail_path": self._find_thumbnail(video_dir, metadata),
                        "collection_id": collection_id
                    }

                    # Get segment statistics if available
                    segments_metadata = metadata.get("segments_metadata", {})
                    if segments_metadata:
                        total_objects = sum(
                            len(seg.get("objects", []))
                            for seg in segments_metadata.values()
                        )
                        video_info["total_objects_detected"] = total_objects

                    videos.append(video_info)

                except Exception as e:
                    print(f"Error loading metadata for {video_dir.name}: {e}")
                    continue

            # Sort by name
            videos.sort(key=lambda x: x["name"])

            print(f"Found {len(videos)} videos in collection '{collection_id}'")

        except Exception as e:
            print(f"Error scanning videos in collection {collection_id}: {e}")

        return videos

    def _find_thumbnail(self, video_dir: Path, metadata: Dict) -> Optional[str]:
        """
        Find thumbnail for video

        Args:
            video_dir: Video directory path
            metadata: Video metadata

        Returns:
            Path to thumbnail or None
        """
        # Check if thumbnail_url is set in metadata
        thumbnail_url = metadata.get("thumbnail_url", "")
        if thumbnail_url and Path(thumbnail_url).exists():
            return thumbnail_url

        # Look for thumbnails in frames directory
        frames_dir = video_dir / "frames"
        if frames_dir.exists():
            # Try to find first frame
            for ext in [".jpg", ".jpeg", ".png"]:
                thumbnail_files = list(frames_dir.glob(f"*{ext}"))
                if thumbnail_files:
                    # Return first frame as thumbnail
                    return str(sorted(thumbnail_files)[0])

        # Look for scenes thumbnails
        scenes_dir = video_dir / "scenes"
        if scenes_dir.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                thumbnail_files = list(scenes_dir.glob(f"*{ext}"))
                if thumbnail_files:
                    return str(sorted(thumbnail_files)[0])

        return None

    def scan_available_pdfs(self) -> List[Dict]:
        """
        Scan PDF storage for all indexed PDFs

        Returns:
            List of PDF context dictionaries with metadata
        """
        pdfs = []

        try:
            # Check if metadata directory exists
            if not self.pdf_metadata_path.exists():
                print(f"PDF metadata directory does not exist: {self.pdf_metadata_path}")
                return pdfs

            # Scan individual metadata files
            for metadata_file in self.pdf_metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract relevant information
                    pdf_info = {
                        "filename": metadata.get("filename", metadata_file.stem + ".pdf"),
                        "num_chunks": metadata.get("num_chunks", 0),
                        "total_characters": metadata.get("total_characters", 0),
                        "chunk_ids": metadata.get("chunk_ids", []),
                        "file_size": metadata.get("file_size", 0),
                        "added_at": metadata.get("added_at", ""),
                        "stored_path": metadata.get("stored_path", "")
                    }

                    pdfs.append(pdf_info)

                except Exception as e:
                    print(f"Error loading metadata from {metadata_file}: {e}")
                    continue

            # Sort by filename
            pdfs.sort(key=lambda x: x["filename"])

            print(f"Found {len(pdfs)} PDFs in storage")

        except Exception as e:
            print(f"Error scanning PDFs: {e}")

        return pdfs

    def get_video_summary(self, video_id: str, collection_id: str = "battlefield_reconnaissance") -> Optional[Dict]:
        """
        Get summary for a specific video

        Args:
            video_id: Video ID
            collection_id: Collection ID

        Returns:
            Video summary dictionary or None
        """
        try:
            metadata_file = self.videodb_path / "collections" / collection_id / "videos" / video_id / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract summary information
            segments_metadata = metadata.get("segments_metadata", {})

            # Count objects by type
            object_counts = {}
            total_objects = 0

            for seg in segments_metadata.values():
                objects = seg.get("objects", [])
                for obj in objects:
                    obj_class = obj.get("class", "unknown")
                    object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
                    total_objects += 1

            summary = {
                "video_id": video_id,
                "name": metadata.get("name", ""),
                "length": metadata.get("length", 0.0),
                "num_segments": len(segments_metadata),
                "total_objects": total_objects,
                "object_counts": object_counts
            }

            return summary

        except Exception as e:
            print(f"Error getting video summary: {e}")
            return None

    def get_pdf_summary(self, filename: str) -> Optional[Dict]:
        """
        Get summary for a specific PDF

        Args:
            filename: PDF filename

        Returns:
            PDF summary dictionary or None
        """
        try:
            pdfs = self.scan_available_pdfs()

            for pdf in pdfs:
                if pdf["filename"] == filename:
                    return pdf

            return None

        except Exception as e:
            print(f"Error getting PDF summary: {e}")
            return None


def scan_available_contexts() -> Dict[str, List[Dict]]:
    """
    Convenience function to scan both videos and PDFs

    Returns:
        Dictionary with 'videos' and 'pdfs' keys
    """
    scanner = ContextScanner()

    return {
        "videos": scanner.scan_available_videos(),
        "pdfs": scanner.scan_available_pdfs()
    }


if __name__ == "__main__":
    # Test scanner
    scanner = ContextScanner()

    print("=== Scanning Videos ===")
    videos = scanner.scan_available_videos()
    print(f"\nFound {len(videos)} videos:")
    for video in videos:
        print(f"  - {video['name']} ({video['video_id']}): {video['num_segments']} segments")

    print("\n=== Scanning PDFs ===")
    pdfs = scanner.scan_available_pdfs()
    print(f"\nFound {len(pdfs)} PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf['filename']}: {pdf['num_chunks']} chunks, {pdf['total_characters']} chars")
