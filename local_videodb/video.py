"""
Local Video class for offline operation
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from .storage import LocalStorage
from .video_processor import VideoProcessor
from .scene import LocalScene, LocalSceneCollection, LocalFrame
from .constants import SceneExtractionType


class LocalVideo:
    """
    Local video implementation with same interface as VideoDB Video class
    """

    def __init__(
        self,
        storage: LocalStorage,
        collection_id: str,
        video_id: str
    ):
        """
        Initialize local video

        Args:
            storage: Local storage instance
            collection_id: Collection ID
            video_id: Video ID
        """
        self._storage = storage
        self._collection_id = collection_id
        self._video_id = video_id

        # Load metadata
        self._metadata = self._storage.get_video_metadata(collection_id, video_id)

        if not self._metadata:
            raise ValueError(f"Video not found: {video_id}")

    @property
    def id(self) -> str:
        """Video ID"""
        return self._metadata["id"]

    @property
    def collection_id(self) -> str:
        """Collection ID"""
        return self._metadata["collection_id"]

    @property
    def name(self) -> str:
        """Video name"""
        return self._metadata.get("name", "")

    @property
    def description(self) -> str:
        """Video description"""
        return self._metadata.get("description", "")

    @property
    def length(self) -> float:
        """Video duration in seconds"""
        return self._metadata.get("length", 0.0)

    @property
    def stream_url(self) -> str:
        """Stream URL"""
        return self._metadata.get("stream_url", "")

    @property
    def player_url(self) -> str:
        """Player URL"""
        return self._metadata.get("player_url", "")

    @property
    def thumbnail_url(self) -> str:
        """Thumbnail URL"""
        return self._metadata.get("thumbnail_url", "")

    @property
    def transcript(self) -> List[Dict]:
        """Transcript with timestamps"""
        return self._metadata.get("transcript", [])

    @property
    def transcript_text(self) -> str:
        """Plain text transcript"""
        return self._metadata.get("transcript_text", "")

    @property
    def scenes(self) -> List[Dict]:
        """Scene data"""
        return self._metadata.get("scenes", [])

    def _get_video_processor(self) -> VideoProcessor:
        """Get video processor instance"""
        file_path = self._metadata.get("file_path")
        if not file_path:
            raise ValueError("Video file path not found in metadata")

        return VideoProcessor(file_path)

    def _update_metadata(self, updates: Dict):
        """Update video metadata"""
        self._metadata.update(updates)
        self._storage.save_video_metadata(
            self._collection_id,
            self._video_id,
            self._metadata
        )

    def extract_scenes(
        self,
        extraction_type: SceneExtractionType,
        extraction_config: Dict,
        force: bool = False,
        callback_url: str = None
    ) -> LocalSceneCollection:
        """
        Extract scenes from video

        Args:
            extraction_type: "time_based" or "shot_based"
            extraction_config: Configuration dict
                For time_based: {"time": int, "frame_count": int, "select_frames": ["first", "middle", "last"]}
                For shot_based: {"threshold": int, "frame_count": int}
            force: Force re-extraction
            callback_url: Not used in offline mode

        Returns:
            LocalSceneCollection
        """
        # Generate scene collection ID
        scene_collection_id = self._storage.generate_id("sc")

        with self._get_video_processor() as processor:
            # Update video length if not set
            if self._metadata.get("length", 0.0) == 0.0:
                self._update_metadata({"length": processor.info.duration})

            # Extract segments based on type
            if extraction_type == SceneExtractionType.time_based:
                segment_duration = extraction_config.get("time", 10)
                frame_positions = extraction_config.get("select_frames", ["first", "middle", "last"])

                segments = processor.segment_by_time(
                    segment_duration=segment_duration,
                    frame_positions=frame_positions
                )

            else:  # shot_based
                threshold = extraction_config.get("threshold", 20)
                frame_count = extraction_config.get("frame_count", 1)

                segments = processor.segment_by_shots(
                    threshold=threshold,
                    frame_count=frame_count
                )

            # Create scene collection
            scene_collection = LocalSceneCollection(
                id=scene_collection_id,
                video_id=self._video_id,
                extraction_type=str(extraction_type.value),
                config=extraction_config
            )

            # Process each segment
            for i, segment in enumerate(segments):
                scene_id = self._storage.generate_id("scene")

                # Process frames
                frames = []
                for frame_data in segment.get("frames", []):
                    frame_id = self._storage.generate_id("f")
                    frame_time = frame_data["frame_time"]
                    frame_array = frame_data["frame"]

                    # Save frame image
                    import cv2
                    _, buffer = cv2.imencode('.jpg', frame_array)
                    frame_path = self._storage.save_frame(
                        self._collection_id,
                        self._video_id,
                        frame_id,
                        buffer.tobytes()
                    )

                    # Create frame object
                    frame = LocalFrame(
                        id=frame_id,
                        video_id=self._video_id,
                        scene_id=scene_id,
                        url=frame_path,
                        frame_time=frame_time
                    )
                    frames.append(frame)

                # Create scene
                scene = LocalScene(
                    id=scene_id,
                    video_id=self._video_id,
                    start=segment["start"],
                    end=segment["end"],
                    frames=frames
                )

                # Save scene metadata
                self._storage.save_scene_metadata(
                    self._collection_id,
                    self._video_id,
                    scene_id,
                    scene.to_dict()
                )

                scene_collection.add_scene(scene)

        return scene_collection

    def generate_thumbnail(self, time: Optional[float] = None) -> str:
        """
        Generate thumbnail at specific time

        Args:
            time: Timestamp (default: middle)

        Returns:
            Path to thumbnail
        """
        with self._get_video_processor() as processor:
            frame = processor.generate_thumbnail(time)

            if frame is not None:
                # Generate thumbnail ID
                thumbnail_id = "thumbnail"

                # Save thumbnail
                import cv2
                _, buffer = cv2.imencode('.jpg', frame)
                thumbnail_path = self._storage.save_frame(
                    self._collection_id,
                    self._video_id,
                    thumbnail_id,
                    buffer.tobytes()
                )

                # Update metadata
                self._update_metadata({"thumbnail_url": thumbnail_path})

                return thumbnail_path

        return ""

    def get_thumbnails(self) -> List[str]:
        """
        Get all extracted thumbnails

        Returns:
            List of thumbnail paths
        """
        frames_dir = self._storage.get_video_path(
            self._collection_id,
            self._video_id
        ) / "frames"

        if frames_dir.exists():
            return [str(p) for p in frames_dir.glob("*.jpg")]

        return []

    def generate_stream(self, timeline: List[List[float]] = None) -> str:
        """
        Generate stream URL (for compatibility)

        Args:
            timeline: List of [start, end] segments

        Returns:
            Stream URL (local file path)
        """
        # In offline mode, return file path
        return self._metadata.get("file_path", "")

    def get_transcript(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        segmenter: str = "word",
        length: int = 1,
        force: bool = False
    ) -> List[Dict]:
        """
        Get video transcript (placeholder for offline)

        Note: Requires speech-to-text integration
        """
        # TODO: Integrate Whisper or similar for offline transcription
        return self._metadata.get("transcript", [])

    def get_transcript_text(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        segmenter: str = "word",
        length: int = 1,
        force: bool = False
    ) -> str:
        """Get plain text transcript"""
        return self._metadata.get("transcript_text", "")

    def delete(self):
        """Delete video"""
        self._storage.delete_video(self._collection_id, self._video_id)

    def play(self):
        """
        Play video (opens file with default application)
        """
        import subprocess
        import sys

        file_path = self._metadata.get("file_path")

        if file_path and Path(file_path).exists():
            if sys.platform == "darwin":  # macOS
                subprocess.call(["open", file_path])
            elif sys.platform == "win32":  # Windows
                subprocess.call(["start", file_path], shell=True)
            else:  # Linux
                subprocess.call(["xdg-open", file_path])
