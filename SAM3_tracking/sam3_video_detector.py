"""
SAM3 Video Object Detection and Tracking System
Uses transformers library to detect and track soldiers, armed vehicles, and tanks in videos.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from tqdm import tqdm

try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
except ImportError:
    raise ImportError(
        "transformers library with SAM3 support is required. "
        "Please install: pip install transformers>=4.47.0"
    )


@dataclass
class DetectionResult:
    """Store detection results for a single frame"""
    frame_idx: int
    objects: List[Dict] = field(default_factory=list)

    def add_object(self, obj_id: int, bbox: List[float], class_name: str, confidence: float):
        """Add a detected object to this frame"""
        self.objects.append({
            'object_id': obj_id,
            'bbox': bbox,  # [x1, y1, x2, y2]
            'class': class_name,
            'confidence': confidence
        })


@dataclass
class VideoAnalysisResults:
    """Store complete analysis results for a video"""
    video_path: str
    total_frames: int
    fps: float
    frame_results: Dict[int, DetectionResult] = field(default_factory=dict)
    object_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_frame_result(self, result: DetectionResult):
        """Add detection result for a frame"""
        self.frame_results[result.frame_idx] = result

    def update_object_counts(self):
        """Count unique objects of each class across all frames"""
        unique_objects = defaultdict(set)

        for frame_result in self.frame_results.values():
            for obj in frame_result.objects:
                unique_objects[obj['class']].add(obj['object_id'])

        self.object_counts = {
            class_name: len(obj_ids)
            for class_name, obj_ids in unique_objects.items()
        }

    def to_dict(self) -> dict:
        """Convert results to dictionary format"""
        return {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'object_counts': dict(self.object_counts),
            'total_detections': sum(self.object_counts.values()),
            'frames': {
                frame_idx: {
                    'objects': result.objects
                }
                for frame_idx, result in self.frame_results.items()
            }
        }

    def save_json(self, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to: {output_path}")


class SAM3VideoDetector:
    """
    SAM3-based video object detector and tracker using transformers library.
    Detects and tracks soldiers, armed vehicles, and tanks in battlefield footage.
    """

    # Object class definitions with multiple prompt variations for robust detection
    OBJECT_CLASSES = {
        'soldier': ['soldier', 'soldiers', 'military personnel', 'infantry'],
        'armed_vehicle': ['military vehicle', 'armed vehicle', 'armored vehicle', 'military truck'],
        'tank': ['tank', 'tanks', 'main battle tank', 'armored tank', 'military tank']
    }

    # Colors for visualization (BGR format for OpenCV)
    CLASS_COLORS = {
        'soldier': (0, 255, 0),      # Green
        'armed_vehicle': (255, 165, 0),  # Orange
        'tank': (0, 0, 255)          # Red
    }

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize SAM3 video detector

        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Model dtype (torch.bfloat16 or torch.float32)
            confidence_threshold: Minimum confidence for detections
        """
        print(f"Initializing SAM3 detector on {device}...")
        self.device = device
        self.dtype = dtype
        self.confidence_threshold = confidence_threshold

        # Load model and processor from transformers
        print(f"Loading model: {model_name}")
        self.model = Sam3VideoModel.from_pretrained(model_name).to(device, dtype=dtype)
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.model.eval()

        print("SAM3 detector initialized successfully")

    def load_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], float, int]:
        """
        Load video frames using OpenCV

        Args:
            video_path: Path to input video file

        Returns:
            Tuple of (frames list, fps, total_frames)
        """
        print(f"Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        with tqdm(total=total_frames, desc="Loading frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB for model
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)

        cap.release()
        print(f"Loaded {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps, total_frames

    def detect_and_track(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> VideoAnalysisResults:
        """
        Detect and track objects in video

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None for all)

        Returns:
            VideoAnalysisResults object with detection data
        """
        # Load video frames
        frames, fps, total_frames = self.load_video_frames(video_path)

        if max_frames is not None:
            frames = frames[:max_frames]
            total_frames = len(frames)

        # Initialize results
        results = VideoAnalysisResults(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps
        )

        # Process each object class separately
        for class_name, prompts in self.OBJECT_CLASSES.items():
            print(f"\nDetecting {class_name} objects...")

            # Use the first prompt as primary
            primary_prompt = prompts[0]

            # Initialize inference session
            inference_session = self.processor.init_video_session(
                video=frames,
                inference_device=self.device,
                dtype=self.dtype
            )

            # Add text prompt for this class
            inference_session = self.processor.add_text_prompt(
                inference_session=inference_session,
                text=primary_prompt
            )

            # Track objects across frames
            with tqdm(total=total_frames, desc=f"Tracking {class_name}") as pbar:
                for model_outputs in self.model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=total_frames
                ):
                    # Post-process outputs
                    processed = self.processor.postprocess_outputs(
                        inference_session, model_outputs
                    )

                    frame_idx = model_outputs.frame_idx

                    # Get or create frame result
                    if frame_idx not in results.frame_results:
                        results.add_frame_result(DetectionResult(frame_idx=frame_idx))

                    frame_result = results.frame_results[frame_idx]

                    # Extract detections
                    if 'object_ids' in processed and len(processed['object_ids']) > 0:
                        object_ids = processed['object_ids'].cpu().numpy()
                        boxes = processed['boxes'].cpu().numpy()  # [N, 4] in XYXY format

                        # Get scores if available
                        if 'scores' in processed:
                            scores = processed['scores'].cpu().numpy()
                        else:
                            scores = np.ones(len(object_ids))

                        # Add each detection
                        for obj_id, bbox, score in zip(object_ids, boxes, scores):
                            if score >= self.confidence_threshold:
                                frame_result.add_object(
                                    obj_id=int(obj_id),
                                    bbox=bbox.tolist(),
                                    class_name=class_name,
                                    confidence=float(score)
                                )

                    pbar.update(1)

        # Update object counts
        results.update_object_counts()

        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        for class_name, count in results.object_counts.items():
            print(f"{class_name.upper()}: {count} unique objects detected")
        print(f"TOTAL: {sum(results.object_counts.values())} objects")
        print("="*60)

        return results

    def visualize_results(
        self,
        video_path: str,
        results: VideoAnalysisResults,
        output_path: str,
        show_ids: bool = True,
        show_confidence: bool = True
    ):
        """
        Create annotated video with bounding boxes

        Args:
            video_path: Path to original video
            results: Detection results
            output_path: Path for output video
            show_ids: Whether to show object IDs
            show_confidence: Whether to show confidence scores
        """
        print(f"\nGenerating annotated video: {output_path}")

        # Load original video
        cap = cv2.VideoCapture(video_path)
        fps = results.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        with tqdm(total=results.total_frames, desc="Rendering video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw detections if available for this frame
                if frame_idx in results.frame_results:
                    frame_result = results.frame_results[frame_idx]

                    for obj in frame_result.objects:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, obj['bbox'])
                        class_name = obj['class']
                        obj_id = obj['object_id']
                        confidence = obj['confidence']

                        # Get color for this class
                        color = self.CLASS_COLORS.get(class_name, (255, 255, 255))

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Prepare label
                        label_parts = [class_name]
                        if show_ids:
                            label_parts.append(f"ID:{obj_id}")
                        if show_confidence:
                            label_parts.append(f"{confidence:.2f}")

                        label = " ".join(label_parts)

                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_h - 10),
                            (x1 + label_w, y1),
                            color,
                            -1
                        )

                        # Draw label text
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

                # Add summary overlay
                y_offset = 30
                for class_name, count in results.object_counts.items():
                    color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
                    text = f"{class_name}: {count}"
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    y_offset += 30

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()
        print(f"Annotated video saved to: {output_path}")

    def process_video(
        self,
        video_path: str,
        output_dir: str = "./output",
        max_frames: Optional[int] = None,
        save_json: bool = True,
        generate_video: bool = True
    ) -> VideoAnalysisResults:
        """
        Complete pipeline: detect, track, and visualize

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            max_frames: Maximum frames to process
            save_json: Whether to save JSON results
            generate_video: Whether to generate annotated video

        Returns:
            VideoAnalysisResults object
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get video basename
        video_name = Path(video_path).stem

        # Run detection and tracking
        results = self.detect_and_track(video_path, max_frames=max_frames)

        # Save JSON results
        if save_json:
            json_path = output_path / f"{video_name}_detections.json"
            results.save_json(str(json_path))

        # Generate annotated video
        if generate_video:
            output_video = output_path / f"{video_name}_annotated.mp4"
            self.visualize_results(video_path, results, str(output_video))

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM3 Video Object Detection and Tracking"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip generating annotated video"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON results"
    )

    args = parser.parse_args()

    # Initialize detector
    detector = SAM3VideoDetector(
        device=args.device,
        confidence_threshold=args.confidence
    )

    # Process video
    results = detector.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        save_json=not args.no_json,
        generate_video=not args.no_video
    )

    print("\nProcessing complete!")
    return results


if __name__ == "__main__":
    main()
