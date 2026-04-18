"""
Video Utilities: Helper functions for video processing
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata

    Args:
        video_path: Path to video file

    Returns:
        Dict with video info (fps, width, height, frame_count, duration)
    """
    try:
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": duration
        }

    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


def extract_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extract a single frame at timestamp

    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds

    Returns:
        Frame as numpy array (BGR)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Seek to timestamp
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


def extract_frames(video_path: str, start_time: float, end_time: float,
                   sample_rate: int = 1) -> List[np.ndarray]:
    """
    Extract frames from a time range

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        sample_rate: Extract every Nth frame

    Returns:
        List of frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        current_frame = start_frame
        frame_idx = 0

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frames.append(frame)

            frame_idx += 1
            current_frame += 1

        cap.release()

        return frames

    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int],
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize frame

    Args:
        frame: Input frame
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized frame
    """
    try:
        if keep_aspect_ratio:
            h, w = frame.shape[:2]
            target_w, target_h = target_size

            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad if needed
            if new_w != target_w or new_h != target_h:
                # Create black canvas
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                # Center the resized image
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return canvas
            else:
                return resized
        else:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    except Exception as e:
        print(f"Error resizing frame: {e}")
        return frame


def crop_region(frame: np.ndarray, bbox: Tuple[float, float, float, float],
               padding: int = 0) -> np.ndarray:
    """
    Crop region from frame

    Args:
        frame: Input frame
        bbox: Bounding box (x_center, y_center, width, height)
        padding: Padding around bbox

    Returns:
        Cropped region
    """
    try:
        h, w = frame.shape[:2]
        x_center, y_center, width, height = bbox

        # Convert to corner coordinates
        x1 = int(x_center - width / 2) - padding
        y1 = int(y_center - height / 2) - padding
        x2 = int(x_center + width / 2) + padding
        y2 = int(y_center + height / 2) + padding

        # Clip to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return frame[y1:y2, x1:x2]

    except Exception as e:
        print(f"Error cropping region: {e}")
        return frame


def save_frame(frame: np.ndarray, output_path: str):
    """
    Save frame to file

    Args:
        frame: Frame to save
        output_path: Output file path
    """
    try:
        cv2.imwrite(output_path, frame)
    except Exception as e:
        print(f"Error saving frame: {e}")


def draw_detections(frame: np.ndarray, detections: List[dict],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
    """
    Draw detection boxes on frame

    Args:
        frame: Input frame
        detections: List of detection dicts with x, y, width, height
        color: Box color (B, G, R)
        thickness: Box thickness

    Returns:
        Frame with drawn boxes
    """
    try:
        output = frame.copy()

        for det in detections:
            x_center = det["x"]
            y_center = det["y"]
            width = det["width"]
            height = det["height"]

            # Convert to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.get('class', 'unknown')} {det.get('confidence', 0):.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output

    except Exception as e:
        print(f"Error drawing detections: {e}")
        return frame


def create_video_from_frames(frames: List[np.ndarray], output_path: str,
                            fps: float = 30.0):
    """
    Create video from frames

    Args:
        frames: List of frames
        output_path: Output video path
        fps: Frames per second
    """
    try:
        if not frames:
            return

        h, w = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()

    except Exception as e:
        print(f"Error creating video: {e}")


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"
