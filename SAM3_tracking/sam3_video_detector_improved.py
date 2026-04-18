"""
SAM3 Video Object Detection and Tracking System (Improved V2)
Uses transformers library to detect and track soldiers and armed vehicles in videos.

V2 Improvements:
1. Merged tank class into armed_vehicle (tanks are armored vehicles)
2. Enhanced false positive filtering:
   - Edge density validation (boxes must contain actual features)
   - Aspect ratio validation (realistic proportions)
   - Position-based filtering (avoid sky/background regions)
   - Texture variance checking (sufficient color variation)
3. Cross-class NMS and supervision library integration

References:
- Edge Detection: https://resources.unitxlabs.com/key-concepts-edge-detection-machine-vision/
- False Positive Reduction: https://arxiv.org/html/2403.02639v1
- Aspect Ratio Validation: https://ncbi.nlm.nih.gov/pmc/articles/PMC8271464
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
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

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
    # Check supervision version for compatibility
    try:
        sv_version = sv.__version__
        print(f"Supervision version: {sv_version}")
    except AttributeError:
        print("WARNING: Could not determine supervision version")
except ImportError:
    print("WARNING: supervision library not available. Install with: pip install supervision>=0.26.0")
    SUPERVISION_AVAILABLE = False


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

    # Statistics
    filtered_stats: Dict[str, int] = field(default_factory=lambda: {
        'duplicate_detections': 0,
        'small_area_filtered': 0,
        'large_area_filtered': 0,
        'low_confidence_filtered': 0,
        'low_edge_density': 0,
        'invalid_aspect_ratio': 0,
        'sky_region_filtered': 0,
        'low_texture_variance': 0
    })

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
            'filtered_stats': self.filtered_stats,
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
    Detects and tracks soldiers and armed vehicles (including tanks) in battlefield footage.

    V2 Features:
    - Merged tank into armed_vehicle class
    - Enhanced false positive filtering with edge/texture analysis
    - Cross-class NMS to eliminate duplicate detections
    - Supervision library integration
    """

    # Object class definitions - TANKS MERGED INTO ARMED_VEHICLE
    OBJECT_CLASSES = {
        'soldier': ['soldier', 'soldiers', 'military personnel', 'infantry', 'person in uniform'],
        'armed_vehicle': [
            # General vehicles
            'military vehicle', 'armed vehicle', 'armored vehicle', 'military truck',
            # Tanks (now included in armed_vehicle)
            'tank', 'tanks', 'main battle tank', 'armored tank', 'military tank',
            'armoured vehicle', 'combat vehicle'
        ]
    }

    # Class hierarchy: higher value = higher priority in case of overlap
    CLASS_PRIORITY = {
        'armed_vehicle': 2,  # Higher priority
        'soldier': 1         # Lower priority
    }

    # Colors for visualization (BGR format for OpenCV)
    CLASS_COLORS = {
        'soldier': (0, 255, 0),       # Green
        'armed_vehicle': (0, 0, 255)  # Red (previously tank color)
    }

    # Aspect ratio constraints (width/height)
    ASPECT_RATIO_LIMITS = {
        'soldier': (0.2, 1.5),      # People are typically taller than wide
        'armed_vehicle': (0.5, 4.0)  # Vehicles can be wider, including long tanks
    }

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        min_box_area: int = 100,
        max_box_area: Optional[int] = None,
        use_supervision: bool = True,
        # New filtering parameters
        min_edge_density: float = 0.05,
        min_texture_variance: float = 10.0,
        sky_region_threshold: float = 0.25
    ):
        """
        Initialize SAM3 video detector with enhanced filtering

        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Model dtype (torch.bfloat16 or torch.float32)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS (lower = more aggressive)
            min_box_area: Minimum bounding box area in pixels
            max_box_area: Maximum bounding box area in pixels (None = no limit)
            use_supervision: Use supervision library for visualization
            min_edge_density: Minimum ratio of edge pixels (0.0-1.0)
            min_texture_variance: Minimum color variance in box region
            sky_region_threshold: Top % of frame considered sky (0.0-1.0)
        """
        print(f"Initializing SAM3 detector on {device}...")
        self.device = device
        self.dtype = dtype
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.use_supervision = use_supervision and SUPERVISION_AVAILABLE

        # Enhanced filtering parameters
        self.min_edge_density = min_edge_density
        self.min_texture_variance = min_texture_variance
        self.sky_region_threshold = sky_region_threshold

        # Load model and processor from transformers
        print(f"Loading model: {model_name}")
        self.model = Sam3VideoModel.from_pretrained(model_name).to(device, dtype=dtype)
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.model.eval()

        if self.use_supervision:
            print("Using supervision library for visualization")
        else:
            print("Using OpenCV for visualization")

        print("SAM3 detector initialized successfully")
        print(f"Enhanced filtering enabled:")
        print(f"  - Edge density threshold: {min_edge_density}")
        print(f"  - Texture variance threshold: {min_texture_variance}")
        print(f"  - Sky region: top {sky_region_threshold*100:.0f}% of frame")

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def filter_by_area(self, bbox: List[float]) -> bool:
        """
        Filter bounding box by area

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            True if bbox should be kept, False if filtered
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        if area < self.min_box_area:
            return False

        if self.max_box_area is not None and area > self.max_box_area:
            return False

        return True

    def filter_by_aspect_ratio(self, bbox: List[float], class_name: str) -> bool:
        """
        Filter by aspect ratio (width/height) based on class

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: Object class

        Returns:
            True if aspect ratio is valid for the class
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if height == 0:
            return False

        aspect_ratio = width / height
        min_ratio, max_ratio = self.ASPECT_RATIO_LIMITS.get(class_name, (0.1, 10.0))

        return min_ratio <= aspect_ratio <= max_ratio

    def filter_by_position(self, bbox: List[float], frame_height: int) -> bool:
        """
        Filter boxes in likely sky/background regions

        Args:
            bbox: [x1, y1, x2, y2]
            frame_height: Height of frame in pixels

        Returns:
            True if position is valid (not in sky region)
        """
        # Calculate center of bounding box
        center_y = (bbox[1] + bbox[3]) / 2

        # Sky region is typically in top portion of frame
        sky_threshold = frame_height * self.sky_region_threshold

        # Reject if box center is in sky region
        return center_y > sky_threshold

    def calculate_edge_density(self, frame: np.ndarray, bbox: List[float]) -> float:
        """
        Calculate edge density within bounding box using Canny edge detection

        Args:
            frame: RGB frame
            bbox: [x1, y1, x2, y2]

        Returns:
            Ratio of edge pixels to total pixels (0.0-1.0)
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # Extract region
            region = frame[y1:y2, x1:x2]

            if region.size == 0:
                return 0.0

            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            else:
                gray = region

            # Apply Canny edge detection
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)

            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size

            return edge_pixels / total_pixels if total_pixels > 0 else 0.0

        except Exception as e:
            # If any error, assume valid (don't reject)
            return 1.0

    def calculate_texture_variance(self, frame: np.ndarray, bbox: List[float]) -> float:
        """
        Calculate color/texture variance within bounding box

        Args:
            frame: RGB frame
            bbox: [x1, y1, x2, y2]

        Returns:
            Standard deviation of pixel intensities
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # Extract region
            region = frame[y1:y2, x1:x2]

            if region.size == 0:
                return 0.0

            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            else:
                gray = region

            # Calculate standard deviation
            return float(np.std(gray))

        except Exception as e:
            # If any error, assume valid
            return 100.0

    def apply_enhanced_filtering(
        self,
        bbox: List[float],
        class_name: str,
        confidence: float,
        frame: np.ndarray,
        results_stats: Dict
    ) -> Tuple[bool, str]:
        """
        Apply enhanced filtering to detect false positives

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: Object class
            confidence: Detection confidence
            frame: Current frame (RGB)
            results_stats: Statistics dictionary to update

        Returns:
            (should_keep, reason) tuple
        """
        # Stage 1: Confidence threshold
        if confidence < self.confidence_threshold:
            results_stats['low_confidence_filtered'] += 1
            return False, "low_confidence"

        # Stage 2: Area filtering
        if not self.filter_by_area(bbox):
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area < self.min_box_area:
                results_stats['small_area_filtered'] += 1
                return False, "small_area"
            else:
                results_stats['large_area_filtered'] += 1
                return False, "large_area"

        # Stage 3: Aspect ratio validation
        if not self.filter_by_aspect_ratio(bbox, class_name):
            results_stats['invalid_aspect_ratio'] += 1
            return False, "invalid_aspect_ratio"

        # Stage 4: Position filtering (sky region)
        frame_height = frame.shape[0]
        if not self.filter_by_position(bbox, frame_height):
            results_stats['sky_region_filtered'] += 1
            return False, "sky_region"

        # Stage 5: Edge density validation
        edge_density = self.calculate_edge_density(frame, bbox)
        if edge_density < self.min_edge_density:
            results_stats['low_edge_density'] += 1
            return False, "low_edge_density"

        # Stage 6: Texture variance validation
        texture_variance = self.calculate_texture_variance(frame, bbox)
        if texture_variance < self.min_texture_variance:
            results_stats['low_texture_variance'] += 1
            return False, "low_texture_variance"

        # Passed all filters
        return True, "valid"

    def apply_cross_class_nms(
        self,
        detections: List[Dict],
        results_stats: Dict
    ) -> List[Dict]:
        """
        Apply cross-class Non-Maximum Suppression to eliminate duplicates
        Uses class hierarchy to keep more specific detections

        Args:
            detections: List of detection dictionaries
            results_stats: Statistics dictionary to update

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return detections

        # Sort by confidence score (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        suppress = []

        for i, det in enumerate(detections):
            if i in suppress:
                continue

            should_keep = True
            for j, kept_det in enumerate(keep):
                bbox1 = np.array(det['bbox'])
                bbox2 = np.array(kept_det['bbox'])
                iou = self.calculate_iou(bbox1, bbox2)

                if iou > self.nms_threshold:
                    # High overlap detected
                    priority1 = self.CLASS_PRIORITY.get(det['class'], 0)
                    priority2 = self.CLASS_PRIORITY.get(kept_det['class'], 0)

                    if priority1 > priority2:
                        # Current detection has higher priority, replace
                        keep[j] = det
                        should_keep = False
                        results_stats['duplicate_detections'] += 1
                        break
                    elif priority1 < priority2:
                        # Kept detection has higher priority, skip current
                        should_keep = False
                        results_stats['duplicate_detections'] += 1
                        break
                    else:
                        # Same priority, keep the one with higher confidence
                        if det['confidence'] <= kept_det['confidence']:
                            should_keep = False
                            results_stats['duplicate_detections'] += 1
                            break

            if should_keep:
                keep.append(det)

        return keep

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
        Detect and track objects in video with enhanced filtering

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

        # Store all detections per frame (before NMS)
        frame_detections_buffer = defaultdict(list)

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

                    # Extract detections
                    if 'object_ids' in processed and len(processed['object_ids']) > 0:
                        object_ids = processed['object_ids'].cpu().numpy()
                        boxes = processed['boxes'].cpu().numpy()  # [N, 4] in XYXY format

                        # Get scores if available
                        if 'scores' in processed:
                            scores = processed['scores'].cpu().numpy()
                        else:
                            scores = np.ones(len(object_ids))

                        # Get current frame for filtering
                        current_frame = frames[frame_idx]

                        # Apply enhanced filtering
                        for obj_id, bbox, score in zip(object_ids, boxes, scores):
                            should_keep, reason = self.apply_enhanced_filtering(
                                bbox=bbox.tolist(),
                                class_name=class_name,
                                confidence=float(score),
                                frame=current_frame,
                                results_stats=results.filtered_stats
                            )

                            if should_keep:
                                # Add to buffer for later NMS
                                frame_detections_buffer[frame_idx].append({
                                    'object_id': int(obj_id),
                                    'bbox': bbox.tolist(),
                                    'class': class_name,
                                    'confidence': float(score)
                                })

                    pbar.update(1)

        # Apply cross-class NMS to each frame
        print("\nApplying cross-class NMS...")
        for frame_idx in tqdm(sorted(frame_detections_buffer.keys()), desc="NMS processing"):
            detections = frame_detections_buffer[frame_idx]

            # Apply NMS
            filtered_detections = self.apply_cross_class_nms(detections, results.filtered_stats)

            # Create frame result
            if frame_idx not in results.frame_results:
                results.add_frame_result(DetectionResult(frame_idx=frame_idx))

            frame_result = results.frame_results[frame_idx]

            # Add filtered detections
            for det in filtered_detections:
                frame_result.add_object(
                    obj_id=det['object_id'],
                    bbox=det['bbox'],
                    class_name=det['class'],
                    confidence=det['confidence']
                )

        # Update object counts
        results.update_object_counts()

        print("\n" + "="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        for class_name, count in results.object_counts.items():
            print(f"{class_name.upper()}: {count} unique objects detected")
        print(f"TOTAL: {sum(results.object_counts.values())} objects")
        print("\nENHANCED FILTERING STATISTICS")
        print("="*70)
        print(f"Duplicate detections removed: {results.filtered_stats['duplicate_detections']}")
        print(f"Low confidence filtered: {results.filtered_stats['low_confidence_filtered']}")
        print(f"Small area filtered: {results.filtered_stats['small_area_filtered']}")
        print(f"Large area filtered: {results.filtered_stats['large_area_filtered']}")
        print(f"Invalid aspect ratio: {results.filtered_stats['invalid_aspect_ratio']}")
        print(f"Sky region filtered: {results.filtered_stats['sky_region_filtered']}")
        print(f"Low edge density: {results.filtered_stats['low_edge_density']}")
        print(f"Low texture variance: {results.filtered_stats['low_texture_variance']}")
        total_filtered = sum(results.filtered_stats.values())
        print(f"TOTAL FILTERED: {total_filtered}")
        print("="*70)

        return results

    def visualize_results_supervision(
        self,
        video_path: str,
        results: VideoAnalysisResults,
        output_path: str
    ):
        """
        Create annotated video using supervision library

        Args:
            video_path: Path to original video
            results: Detection results
            output_path: Path for output video
        """
        print(f"\nGenerating annotated video with supervision: {output_path}")

        # Load original video
        cap = cv2.VideoCapture(video_path)
        fps = results.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Setup supervision annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.5,
            text_padding=10
        )

        frame_idx = 0
        with tqdm(total=results.total_frames, desc="Rendering video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw detections if available for this frame
                if frame_idx in results.frame_results:
                    frame_result = results.frame_results[frame_idx]

                    if len(frame_result.objects) > 0:
                        # Prepare supervision Detections object
                        xyxy = np.array([obj['bbox'] for obj in frame_result.objects])
                        class_ids = np.array([
                            list(self.OBJECT_CLASSES.keys()).index(obj['class'])
                            for obj in frame_result.objects
                        ])
                        confidence = np.array([obj['confidence'] for obj in frame_result.objects])
                        tracker_ids = np.array([obj['object_id'] for obj in frame_result.objects])

                        detections = sv.Detections(
                            xyxy=xyxy,
                            class_id=class_ids,
                            confidence=confidence,
                            tracker_id=tracker_ids
                        )

                        # Create labels
                        labels = [
                            f"{obj['class']} ID:{obj['object_id']} {obj['confidence']:.2f}"
                            for obj in frame_result.objects
                        ]

                        # Annotate frame
                        frame = box_annotator.annotate(scene=frame, detections=detections)
                        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

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

    def visualize_results_opencv(
        self,
        video_path: str,
        results: VideoAnalysisResults,
        output_path: str,
        show_ids: bool = True,
        show_confidence: bool = True
    ):
        """
        Create annotated video with bounding boxes using OpenCV

        Args:
            video_path: Path to original video
            results: Detection results
            output_path: Path for output video
            show_ids: Whether to show object IDs
            show_confidence: Whether to show confidence scores
        """
        print(f"\nGenerating annotated video with OpenCV: {output_path}")

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

    def visualize_results(
        self,
        video_path: str,
        results: VideoAnalysisResults,
        output_path: str,
        show_ids: bool = True,
        show_confidence: bool = True
    ):
        """
        Create annotated video - dispatches to supervision or OpenCV based on availability
        """
        if self.use_supervision:
            self.visualize_results_supervision(video_path, results, output_path)
        else:
            self.visualize_results_opencv(
                video_path, results, output_path, show_ids, show_confidence
            )

    def process_video(
        self,
        video_path: str,
        output_dir: str = "./output",
        max_frames: Optional[int] = None,
        save_json: bool = True,
        generate_video: bool = True
    ) -> VideoAnalysisResults:
        """
        Complete pipeline: detect, track, filter, and visualize

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
        description="SAM3 Video Object Detection and Tracking (Improved V2)"
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
        "--nms-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for cross-class NMS"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum bounding box area in pixels"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Maximum bounding box area in pixels"
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=0.05,
        help="Minimum edge density (0.0-1.0)"
    )
    parser.add_argument(
        "--min-texture-variance",
        type=float,
        default=10.0,
        help="Minimum texture variance"
    )
    parser.add_argument(
        "--sky-region",
        type=float,
        default=0.25,
        help="Top portion of frame considered sky (0.0-1.0)"
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
    parser.add_argument(
        "--no-supervision",
        action="store_true",
        help="Use OpenCV instead of supervision for visualization"
    )

    args = parser.parse_args()

    # Initialize detector
    detector = SAM3VideoDetector(
        device=args.device,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        min_box_area=args.min_area,
        max_box_area=args.max_area,
        min_edge_density=args.min_edge_density,
        min_texture_variance=args.min_texture_variance,
        sky_region_threshold=args.sky_region,
        use_supervision=not args.no_supervision
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
