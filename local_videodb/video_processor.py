"""
Video Processing Utilities for Offline Operation
Handles segmentation, frame extraction, and video analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video information"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass
class SceneSegment:
    """Scene segment information"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


class VideoProcessor:
    """
    Handles video processing operations
    """

    def __init__(self, video_path: str):
        """
        Initialize video processor

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video info
        self.info = self._get_video_info()

    def _get_video_info(self) -> VideoInfo:
        """Get video information"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration
        )

    def extract_frame(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract frame at specific timestamp

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array (BGR) or None
        """
        if timestamp < 0 or timestamp > self.info.duration:
            return None

        # Calculate frame number
        frame_num = int(timestamp * self.info.fps)

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read frame
        ret, frame = self.cap.read()

        if ret:
            return frame

        return None

    def extract_frames(self, timestamps: List[float]) -> List[Tuple[float, np.ndarray]]:
        """
        Extract multiple frames

        Args:
            timestamps: List of timestamps

        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []

        for timestamp in timestamps:
            frame = self.extract_frame(timestamp)
            if frame is not None:
                frames.append((timestamp, frame))

        return frames

    def segment_by_time(
        self,
        segment_duration: float = 10.0,
        frame_positions: List[str] = None
    ) -> List[Dict]:
        """
        Segment video by fixed time intervals

        Args:
            segment_duration: Duration of each segment in seconds
            frame_positions: Which frames to extract ("first", "middle", "last")

        Returns:
            List of segment dictionaries with frames
        """
        if frame_positions is None:
            frame_positions = ["first", "middle", "last"]

        segments = []
        current_time = 0.0

        while current_time < self.info.duration:
            # Calculate segment boundaries
            start_time = current_time
            end_time = min(current_time + segment_duration, self.info.duration)

            # Calculate frame timestamps to extract
            frame_times = []

            if "first" in frame_positions:
                frame_times.append(start_time)

            if "middle" in frame_positions:
                frame_times.append((start_time + end_time) / 2)

            if "last" in frame_positions:
                frame_times.append(end_time)

            # Extract frames
            frames_data = []
            for frame_time in frame_times:
                frame = self.extract_frame(frame_time)
                if frame is not None:
                    frames_data.append({
                        "frame_time": frame_time,
                        "frame": frame
                    })

            segment = {
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "frames": frames_data
            }

            segments.append(segment)
            current_time = end_time

        return segments

    def segment_by_shots(
        self,
        threshold: float = 20.0,
        frame_count: int = 1
    ) -> List[Dict]:
        """
        Segment video by detecting shot changes using histogram comparison

        Args:
            threshold: Sensitivity threshold (1-100, higher = less sensitive)
            frame_count: Number of frames to extract per scene

        Returns:
            List of segment dictionaries with frames
        """
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_hist = None
        scenes = []
        scene_start = 0
        frame_idx = 0

        # Normalized threshold (0-1 scale)
        norm_threshold = threshold / 100.0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Convert to grayscale for histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Compare with previous frame
            if prev_hist is not None:
                # Calculate histogram difference
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

                # Detect scene change
                if diff > norm_threshold:
                    # End previous scene
                    scene_end = frame_idx
                    start_time = scene_start / self.info.fps
                    end_time = scene_end / self.info.fps

                    # Extract representative frames
                    frame_times = self._get_representative_frame_times(
                        start_time,
                        end_time,
                        frame_count
                    )

                    frames_data = []
                    for frame_time in frame_times:
                        f = self.extract_frame(frame_time)
                        if f is not None:
                            frames_data.append({
                                "frame_time": frame_time,
                                "frame": f
                            })

                    scenes.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "frames": frames_data
                    })

                    # Start new scene
                    scene_start = frame_idx

            prev_hist = hist
            frame_idx += 1

        # Add final scene
        if scene_start < frame_idx:
            start_time = scene_start / self.info.fps
            end_time = frame_idx / self.info.fps

            frame_times = self._get_representative_frame_times(
                start_time,
                end_time,
                frame_count
            )

            frames_data = []
            for frame_time in frame_times:
                f = self.extract_frame(frame_time)
                if f is not None:
                    frames_data.append({
                        "frame_time": frame_time,
                        "frame": f
                    })

            scenes.append({
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "frames": frames_data
            })

        return scenes

    def _get_representative_frame_times(
        self,
        start: float,
        end: float,
        count: int
    ) -> List[float]:
        """
        Get representative frame timestamps within segment

        Args:
            start: Start time
            end: End time
            count: Number of frames

        Returns:
            List of timestamps
        """
        if count == 1:
            # Return middle frame
            return [(start + end) / 2]

        # Distribute evenly
        duration = end - start
        step = duration / (count + 1)

        return [start + step * (i + 1) for i in range(count)]

    def generate_thumbnail(self, timestamp: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Generate thumbnail

        Args:
            timestamp: Timestamp (default: middle of video)

        Returns:
            Thumbnail frame
        """
        if timestamp is None:
            timestamp = self.info.duration / 2

        return self.extract_frame(timestamp)

    def save_frame_as_image(
        self,
        frame: np.ndarray,
        output_path: str,
        quality: int = 95
    ):
        """
        Save frame as image file

        Args:
            frame: Frame array
            output_path: Output path
            quality: JPEG quality (0-100)
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(str(output), frame, encode_params)

    def cleanup(self):
        """Release video resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
