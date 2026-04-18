"""
Local Storage Manager for Offline VideoDB
Handles file operations and metadata persistence
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class LocalStorage:
    """
    Manages local file storage and metadata for videos
    """

    def __init__(self, storage_path: str = "./local_videodb"):
        """
        Initialize local storage

        Args:
            storage_path: Root path for storage
        """
        self.storage_path = Path(storage_path)
        self.collections_path = self.storage_path / "collections"
        self.cache_path = self.storage_path / "cache"

        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collections_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)

    def generate_id(self, prefix: str = "m") -> str:
        """
        Generate unique ID

        Args:
            prefix: ID prefix (m=media, c=collection, sc=scene, f=frame)

        Returns:
            Unique ID string
        """
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def get_collection_path(self, collection_id: str) -> Path:
        """Get path for collection"""
        return self.collections_path / collection_id

    def get_video_path(self, collection_id: str, video_id: str) -> Path:
        """Get path for video"""
        return self.get_collection_path(collection_id) / "videos" / video_id

    def collection_exists(self, collection_id: str) -> bool:
        """Check if collection exists"""
        return self.get_collection_path(collection_id).exists()

    def video_exists(self, collection_id: str, video_id: str) -> bool:
        """Check if video exists"""
        return self.get_video_path(collection_id, video_id).exists()

    def create_collection(
        self,
        collection_id: str,
        name: str,
        description: str = ""
    ) -> Dict:
        """
        Create a new collection

        Args:
            collection_id: Collection ID
            name: Collection name
            description: Collection description

        Returns:
            Collection metadata dict
        """
        collection_path = self.get_collection_path(collection_id)
        collection_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (collection_path / "videos").mkdir(exist_ok=True)

        # Create metadata
        metadata = {
            "id": collection_id,
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "videos": []
        }

        # Save metadata
        self.save_collection_metadata(collection_id, metadata)

        return metadata

    def get_collection_metadata(self, collection_id: str) -> Optional[Dict]:
        """Load collection metadata"""
        metadata_file = self.get_collection_path(collection_id) / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def save_collection_metadata(self, collection_id: str, metadata: Dict):
        """Save collection metadata"""
        metadata_file = self.get_collection_path(collection_id) / "metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def add_video_to_collection(self, collection_id: str, video_id: str):
        """Add video ID to collection"""
        metadata = self.get_collection_metadata(collection_id)

        if metadata and video_id not in metadata.get("videos", []):
            metadata["videos"].append(video_id)
            self.save_collection_metadata(collection_id, metadata)

    def create_video(
        self,
        collection_id: str,
        video_id: str,
        source_path: str,
        name: str,
        description: str = ""
    ) -> Dict:
        """
        Create video entry and copy file

        Args:
            collection_id: Collection ID
            video_id: Video ID
            source_path: Path to source video file
            name: Video name
            description: Video description

        Returns:
            Video metadata dict
        """
        video_path = self.get_video_path(collection_id, video_id)
        video_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (video_path / "frames").mkdir(exist_ok=True)
        (video_path / "scenes").mkdir(exist_ok=True)

        # Copy video file
        source = Path(source_path)
        destination = video_path / f"original{source.suffix}"

        if source.exists():
            shutil.copy2(source, destination)
        else:
            raise FileNotFoundError(f"Source video not found: {source_path}")

        # Create metadata
        metadata = {
            "id": video_id,
            "collection_id": collection_id,
            "name": name,
            "description": description,
            "file_path": str(destination),
            "file_extension": source.suffix,
            "length": 0.0,  # Will be updated by video processor
            "thumbnail_url": "",
            "stream_url": f"local://{video_id}",
            "player_url": f"local://{video_id}",
            "created_at": datetime.utcnow().isoformat(),
            "transcript": [],
            "transcript_text": "",
            "scenes": []
        }

        # Save metadata
        self.save_video_metadata(collection_id, video_id, metadata)

        # Add to collection
        self.add_video_to_collection(collection_id, video_id)

        return metadata

    def get_video_metadata(
        self,
        collection_id: str,
        video_id: str
    ) -> Optional[Dict]:
        """Load video metadata"""
        metadata_file = self.get_video_path(collection_id, video_id) / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def save_video_metadata(
        self,
        collection_id: str,
        video_id: str,
        metadata: Dict
    ):
        """Save video metadata"""
        metadata_file = self.get_video_path(collection_id, video_id) / "metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def update_video_metadata(
        self,
        collection_id: str,
        video_id: str,
        updates: Dict
    ) -> Dict:
        """Update video metadata"""
        metadata = self.get_video_metadata(collection_id, video_id)

        if metadata:
            metadata.update(updates)
            self.save_video_metadata(collection_id, video_id, metadata)

        return metadata

    def get_video_file_path(self, collection_id: str, video_id: str) -> Optional[Path]:
        """Get path to video file"""
        metadata = self.get_video_metadata(collection_id, video_id)

        if metadata:
            return Path(metadata["file_path"])

        return None

    def save_frame(
        self,
        collection_id: str,
        video_id: str,
        frame_id: str,
        frame_data: bytes,
        extension: str = ".jpg"
    ) -> str:
        """
        Save frame image

        Returns:
            Path to saved frame
        """
        frame_path = self.get_video_path(collection_id, video_id) / "frames" / f"{frame_id}{extension}"

        with open(frame_path, 'wb') as f:
            f.write(frame_data)

        return str(frame_path)

    def save_scene_metadata(
        self,
        collection_id: str,
        video_id: str,
        scene_id: str,
        metadata: Dict
    ):
        """Save scene metadata"""
        scene_file = self.get_video_path(collection_id, video_id) / "scenes" / f"{scene_id}.json"

        with open(scene_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_scene_metadata(
        self,
        collection_id: str,
        video_id: str,
        scene_id: str
    ) -> Optional[Dict]:
        """Load scene metadata"""
        scene_file = self.get_video_path(collection_id, video_id) / "scenes" / f"{scene_id}.json"

        if not scene_file.exists():
            return None

        with open(scene_file, 'r') as f:
            return json.load(f)

    def list_videos(self, collection_id: str) -> List[str]:
        """List all video IDs in collection"""
        metadata = self.get_collection_metadata(collection_id)

        if metadata:
            return metadata.get("videos", [])

        return []

    def delete_video(self, collection_id: str, video_id: str):
        """Delete video and all associated data"""
        video_path = self.get_video_path(collection_id, video_id)

        if video_path.exists():
            shutil.rmtree(video_path)

        # Remove from collection
        metadata = self.get_collection_metadata(collection_id)
        if metadata and video_id in metadata.get("videos", []):
            metadata["videos"].remove(video_id)
            self.save_collection_metadata(collection_id, metadata)

    def delete_collection(self, collection_id: str):
        """Delete entire collection"""
        collection_path = self.get_collection_path(collection_id)

        if collection_path.exists():
            shutil.rmtree(collection_path)
