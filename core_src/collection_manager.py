"""
Collection Manager: Centralized collection management for video analysis system
Provides clean interface for creating, listing, and managing collections
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class CollectionManager:
    """
    Manages collections for video storage and organization
    Provides high-level interface for collection operations
    """

    def __init__(self, videodb_path: str = "./data/local_videodb"):
        """
        Initialize collection manager

        Args:
            videodb_path: Path to local_videodb storage
        """
        self.videodb_path = Path(videodb_path)
        self.collections_path = self.videodb_path / "collections"

        # Ensure collections directory exists
        self.collections_path.mkdir(parents=True, exist_ok=True)

    def list_collections(self) -> List[Dict]:
        """
        List all available collections with statistics

        Returns:
            List of collection info dictionaries
        """
        collections = []

        try:
            if not self.collections_path.exists():
                return collections

            # Scan collection directories
            for coll_dir in self.collections_path.iterdir():
                if not coll_dir.is_dir():
                    continue

                metadata_file = coll_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Count videos
                    video_count = len(metadata.get("videos", []))

                    # Get total objects across all videos
                    total_objects = 0
                    videos_dir = coll_dir / "videos"
                    if videos_dir.exists():
                        for video_dir in videos_dir.iterdir():
                            if video_dir.is_dir():
                                video_meta_file = video_dir / "metadata.json"
                                if video_meta_file.exists():
                                    with open(video_meta_file, 'r') as vf:
                                        video_meta = json.load(vf)
                                        segments = video_meta.get("segments_metadata", {})
                                        for seg in segments.values():
                                            total_objects += len(seg.get("objects", []))

                    collection_info = {
                        "id": metadata.get("id", coll_dir.name),
                        "name": metadata.get("name", coll_dir.name),
                        "description": metadata.get("description", ""),
                        "video_count": video_count,
                        "total_objects": total_objects,
                        "created_at": metadata.get("created_at", ""),
                        "path": str(coll_dir)
                    }

                    collections.append(collection_info)

                except Exception as e:
                    print(f"Error loading collection {coll_dir.name}: {e}")
                    continue

            # Sort by name
            collections.sort(key=lambda x: x["name"])

        except Exception as e:
            print(f"Error listing collections: {e}")

        return collections

    def create_collection(self, name: str, description: str = "") -> Tuple[bool, str, Optional[str]]:
        """
        Create a new collection

        Args:
            name: Collection name
            description: Optional description

        Returns:
            Tuple of (success: bool, message: str, collection_id: Optional[str])
        """
        try:
            if not name or not name.strip():
                return False, "Collection name cannot be empty", None

            name = name.strip()

            # Sanitize collection ID (lowercase, alphanumeric and underscore only)
            collection_id = re.sub(r'[^a-z0-9_]', '_', name.lower())

            # Remove multiple underscores
            collection_id = re.sub(r'_{2,}', '_', collection_id)

            # Remove leading/trailing underscores
            collection_id = collection_id.strip('_')

            # Check if collection exists
            collection_path = self.collections_path / collection_id
            if collection_path.exists():
                return False, f"Collection '{name}' already exists (ID: {collection_id})", None

            # Create collection directory structure
            collection_path.mkdir(parents=True, exist_ok=True)
            (collection_path / "videos").mkdir(exist_ok=True)

            # Create metadata
            metadata = {
                "id": collection_id,
                "name": name,
                "description": description or f"Collection: {name}",
                "created_at": datetime.now().isoformat(),
                "videos": []
            }

            # Save metadata
            metadata_file = collection_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True, f"Successfully created collection: {name}", collection_id

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error creating collection: {str(e)}", None

    def get_collection_info(self, collection_id: str) -> Optional[Dict]:
        """
        Get detailed information about a collection

        Args:
            collection_id: Collection ID

        Returns:
            Collection info dict or None if not found
        """
        try:
            collection_path = self.collections_path / collection_id
            metadata_file = collection_path / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Add video count
            metadata["video_count"] = len(metadata.get("videos", []))

            return metadata

        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None

    def delete_collection(self, collection_id: str, force: bool = False) -> Tuple[bool, str]:
        """
        Delete a collection

        Args:
            collection_id: Collection ID to delete
            force: If True, delete even if collection has videos

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            collection_path = self.collections_path / collection_id

            if not collection_path.exists():
                return False, f"Collection '{collection_id}' not found"

            # Check if collection has videos
            metadata_file = collection_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    video_count = len(metadata.get("videos", []))

                    if video_count > 0 and not force:
                        return False, f"Collection has {video_count} videos. Use force=True to delete anyway."

            # Delete collection directory
            import shutil
            shutil.rmtree(collection_path)

            return True, f"Successfully deleted collection: {collection_id}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error deleting collection: {str(e)}"

    def get_collection_choices_for_dropdown(self) -> List[Tuple[str, str]]:
        """
        Get collection choices formatted for Gradio dropdown

        Returns:
            List of (label, value) tuples
        """
        collections = self.list_collections()

        if not collections:
            return [("No collections available", "")]

        # Format: "Name (X videos, Y objects)" -> collection_id
        choices = []
        for coll in collections:
            label = f"{coll['name']} ({coll['video_count']} videos, {coll['total_objects']} objects)"
            choices.append((label, coll['id']))

        return choices

    def get_collection_choices_for_checkbox(self) -> List[Tuple[str, str]]:
        """
        Get collection choices formatted for Gradio checkbox group

        Returns:
            List of (label, value) tuples
        """
        collections = self.list_collections()

        if not collections:
            return []

        # Format: "Name (X videos)" -> collection_id
        choices = []
        for coll in collections:
            label = f"{coll['name']} ({coll['video_count']} videos)"
            choices.append((label, coll['id']))

        return choices

    def validate_collection_exists(self, collection_id: str) -> bool:
        """
        Check if a collection exists

        Args:
            collection_id: Collection ID to check

        Returns:
            True if collection exists, False otherwise
        """
        collection_path = self.collections_path / collection_id
        return collection_path.exists()

    def get_default_collection(self) -> str:
        """
        Get default collection ID (battlefield_reconnaissance or first available)

        Returns:
            Collection ID
        """
        # Check if battlefield_reconnaissance exists
        if self.validate_collection_exists("battlefield_reconnaissance"):
            return "battlefield_reconnaissance"

        # Get first available collection
        collections = self.list_collections()
        if collections:
            return collections[0]["id"]

        # No collections exist, return default
        return "battlefield_reconnaissance"

    def ensure_default_collection_exists(self):
        """Ensure at least one default collection exists"""
        if not self.validate_collection_exists("battlefield_reconnaissance"):
            self.create_collection(
                name="Battlefield Reconnaissance",
                description="Default collection for battlefield reconnaissance drone footage analysis"
            )


class CollectionState:
    """
    Manages the current state of active collections and upload targets
    Singleton pattern for UI-wide state management
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CollectionState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.active_collections = ["battlefield_reconnaissance"]  # Collections to scan for videos
        self.upload_collection = "battlefield_reconnaissance"  # Target collection for uploads
        self._initialized = True

    def set_active_collections(self, collection_ids: List[str]):
        """Set active collections for context scanning"""
        self.active_collections = collection_ids if collection_ids else []

    def get_active_collections(self) -> List[str]:
        """Get currently active collections"""
        return self.active_collections

    def set_upload_collection(self, collection_id: str):
        """Set target collection for uploads"""
        self.upload_collection = collection_id

    def get_upload_collection(self) -> str:
        """Get current upload target collection"""
        return self.upload_collection

    def add_to_active(self, collection_id: str):
        """Add a collection to active list"""
        if collection_id not in self.active_collections:
            self.active_collections.append(collection_id)

    def remove_from_active(self, collection_id: str):
        """Remove a collection from active list"""
        if collection_id in self.active_collections:
            self.active_collections.remove(collection_id)


# Global collection state instance
def get_collection_state() -> CollectionState:
    """Get global collection state instance"""
    return CollectionState()
