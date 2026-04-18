"""
Local Connection class for offline operation
Replaces VideoDB API connection
"""

from pathlib import Path
from typing import Optional
from .storage import LocalStorage
from .collection import LocalCollection
from .video import LocalVideo
from .constants import DEFAULT_STORAGE_PATH, DEFAULT_COLLECTION, MediaType


class LocalConnection:
    """
    Local connection for offline video database

    Provides same interface as VideoDB Connection but operates entirely offline
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize local connection

        Args:
            storage_path: Path to local storage directory
        """
        if storage_path is None:
            storage_path = DEFAULT_STORAGE_PATH

        self._storage = LocalStorage(storage_path)

        # Ensure default collection exists
        if not self._storage.collection_exists(DEFAULT_COLLECTION):
            self._storage.create_collection(
                collection_id=DEFAULT_COLLECTION,
                name="Default Collection",
                description="Default video collection"
            )

        print(f"✓ Local VideoDB initialized at: {storage_path}")

    @property
    def storage(self) -> LocalStorage:
        """Get storage instance"""
        return self._storage

    def get_collection(self, collection_id: str = DEFAULT_COLLECTION) -> LocalCollection:
        """
        Get or create collection

        Args:
            collection_id: Collection ID (default: "default")

        Returns:
            LocalCollection instance
        """
        return LocalCollection(
            storage=self._storage,
            collection_id=collection_id
        )

    def create_collection(
        self,
        name: str,
        description: str = "",
        is_public: bool = False
    ) -> LocalCollection:
        """
        Create new collection

        Args:
            name: Collection name
            description: Collection description
            is_public: Public/private (not used in offline mode)

        Returns:
            LocalCollection instance
        """
        # Use name as collection_id (sanitized)
        collection_id = name.lower().replace(" ", "_")

        if self._storage.collection_exists(collection_id):
            print(f"Collection '{collection_id}' already exists")
            return self.get_collection(collection_id)

        self._storage.create_collection(
            collection_id=collection_id,
            name=name,
            description=description
        )

        return LocalCollection(
            storage=self._storage,
            collection_id=collection_id
        )

    def upload(
        self,
        source: str,
        media_type: MediaType = MediaType.video,
        name: str = None,
        description: str = "",
        collection_id: str = DEFAULT_COLLECTION,
        **kwargs
    ) -> LocalVideo:
        """
        Upload video to default collection

        Args:
            source: Path to video file
            media_type: Type of media
            name: Video name
            description: Video description
            collection_id: Target collection
            **kwargs: Additional arguments (ignored)

        Returns:
            LocalVideo instance
        """
        collection = self.get_collection(collection_id)

        return collection.upload(
            source=source,
            media_type=media_type,
            name=name,
            description=description
        )

    def get_video(
        self,
        video_id: str,
        collection_id: str = DEFAULT_COLLECTION
    ) -> LocalVideo:
        """
        Get video by ID

        Args:
            video_id: Video ID
            collection_id: Collection ID

        Returns:
            LocalVideo instance
        """
        return LocalVideo(
            storage=self._storage,
            collection_id=collection_id,
            video_id=video_id
        )

    def get_storage_path(self) -> Path:
        """Get storage directory path"""
        return self._storage.storage_path

    def get_stats(self) -> dict:
        """
        Get storage statistics

        Returns:
            Dictionary with stats
        """
        stats = {
            "storage_path": str(self._storage.storage_path),
            "collections": []
        }

        # Iterate collections
        if self._storage.collections_path.exists():
            for coll_dir in self._storage.collections_path.iterdir():
                if coll_dir.is_dir():
                    metadata = self._storage.get_collection_metadata(coll_dir.name)
                    if metadata:
                        video_count = len(metadata.get("videos", []))
                        stats["collections"].append({
                            "id": coll_dir.name,
                            "name": metadata.get("name", ""),
                            "video_count": video_count
                        })

        return stats

    def cleanup(self):
        """Cleanup resources (for compatibility)"""
        pass

    def __repr__(self):
        return f"LocalConnection(storage_path='{self._storage.storage_path}')"
