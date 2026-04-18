"""
Local Collection class for offline operation
"""

from typing import List, Optional
from pathlib import Path
from .storage import LocalStorage
from .video import LocalVideo
from .constants import MediaType, DEFAULT_COLLECTION


class LocalCollection:
    """
    Local collection implementation
    """

    def __init__(
        self,
        storage: LocalStorage,
        collection_id: str = DEFAULT_COLLECTION
    ):
        """
        Initialize local collection

        Args:
            storage: Local storage instance
            collection_id: Collection ID
        """
        self._storage = storage
        self._collection_id = collection_id

        # Load or create collection
        if not self._storage.collection_exists(collection_id):
            self._storage.create_collection(
                collection_id=collection_id,
                name=collection_id,
                description=f"Collection: {collection_id}"
            )

        self._metadata = self._storage.get_collection_metadata(collection_id)

    @property
    def id(self) -> str:
        """Collection ID"""
        return self._metadata["id"]

    @property
    def name(self) -> str:
        """Collection name"""
        return self._metadata.get("name", "")

    @property
    def description(self) -> str:
        """Collection description"""
        return self._metadata.get("description", "")

    def upload(
        self,
        source: str,
        media_type: MediaType = MediaType.video,
        name: str = None,
        description: str = "",
        **kwargs
    ) -> LocalVideo:
        """
        Upload video to collection

        Args:
            source: Path to video file
            media_type: Type of media (only video supported currently)
            name: Video name (default: filename)
            description: Video description

        Returns:
            LocalVideo instance
        """
        if media_type != MediaType.video:
            raise NotImplementedError("Only video media type supported in offline mode")

        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Generate video ID
        video_id = self._storage.generate_id("m")

        # Use filename as default name
        if name is None:
            name = source_path.stem

        # Create video entry
        video_metadata = self._storage.create_video(
            collection_id=self._collection_id,
            video_id=video_id,
            source_path=str(source_path),
            name=name,
            description=description
        )

        # Update video length
        from .video_processor import VideoProcessor
        try:
            with VideoProcessor(str(source_path)) as processor:
                self._storage.update_video_metadata(
                    self._collection_id,
                    video_id,
                    {"length": processor.info.duration}
                )
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")

        # Return video object
        return LocalVideo(
            storage=self._storage,
            collection_id=self._collection_id,
            video_id=video_id
        )

    def get_videos(self) -> List[LocalVideo]:
        """
        Get all videos in collection

        Returns:
            List of LocalVideo instances
        """
        video_ids = self._storage.list_videos(self._collection_id)

        videos = []
        for video_id in video_ids:
            try:
                video = LocalVideo(
                    storage=self._storage,
                    collection_id=self._collection_id,
                    video_id=video_id
                )
                videos.append(video)
            except Exception as e:
                print(f"Warning: Could not load video {video_id}: {e}")

        return videos

    def get_video(self, video_id: str) -> LocalVideo:
        """
        Get specific video

        Args:
            video_id: Video ID

        Returns:
            LocalVideo instance
        """
        return LocalVideo(
            storage=self._storage,
            collection_id=self._collection_id,
            video_id=video_id
        )

    def delete_video(self, video_id: str):
        """
        Delete video from collection

        Args:
            video_id: Video ID
        """
        self._storage.delete_video(self._collection_id, video_id)

    def delete(self):
        """Delete entire collection"""
        self._storage.delete_collection(self._collection_id)
