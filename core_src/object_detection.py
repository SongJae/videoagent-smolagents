"""
ObjectDetectionProcessor: SAM3-based detector and tracker for battlefield objects.

Uses the official SAM3 video predictor API from facebookresearch/sam3.
Reference: https://github.com/facebookresearch/sam3

IMPORTANT: SAM3 text prompts use contrastive learning alignment and can only
process ONE text prompt at a time per session. To track multiple object classes:
1. For each class: start session → add text prompt → propagate → collect results
2. Merge results from all classes with unique object IDs

See: https://github.com/facebookresearch/sam3/issues/206

DTYPE NOTE: SAM3 internally uses @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
on key methods (add_prompt, _run_single_frame_inference). When processing multiple
object classes sequentially, the model's internal state accumulates in bfloat16 while
some tensors may remain float32, causing dtype mismatch errors.

Solution: Wrap all predictor calls with torch.autocast("cuda", dtype=torch.bfloat16)
to ensure consistent dtype handling across the entire inference chain.

See: https://github.com/facebookresearch/sam2/issues/577
"""

import os
import subprocess
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
import tempfile
from tqdm import tqdm
from collections import defaultdict
import gc

# Try to import SAM3 from local installation
SAM3_AVAILABLE = False
try:
    from sam3.model_builder import build_sam3_video_predictor
    SAM3_AVAILABLE = True
    print("SAM3 video predictor is available")
except ImportError as e:
    print(f"SAM3 not available: {e}")
    print("   Install with: cd SAM3_tracking/sam3 && pip install -e .")


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in XYWH format (normalized 0-1).

    Args:
        box1, box2: [x, y, w, h] in normalized coordinates

    Returns:
        float: IoU score
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to XYXY
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class ObjectDetectionProcessor:
    """
    SAM3-based processor for detecting and tracking battlefield objects.

    CRITICAL: SAM3's text-based detection uses contrastive learning alignment
    that can only process ONE text embedding at a time. When tracking multiple
    object classes (e.g., soldier, tank, truck), you MUST:

    1. Process each class in a SEPARATE session
    2. For each class: start_session → add_prompt → propagate_in_video → close_session
    3. Merge results from all classes, assigning unique object IDs

    Reference: https://github.com/facebookresearch/sam3/issues/206
    """

    # Default object class definitions with text prompts for SAM3
    # These can be overridden via config/models_config.yaml
    DEFAULT_OBJECT_CLASSES = {
        'soldier': 'soldier',
        'tank': 'tank',
        'truck': 'military truck'
    }

    # Default colors for visualization (BGR format for OpenCV)
    # Auto-generated colors will be used for classes not in this dict
    DEFAULT_CLASS_COLORS = {
        'soldier': (0, 255, 0),      # Green
        'tank': (0, 0, 255),         # Red
        'truck': (255, 165, 0),      # Orange
    }

    # Color palette for dynamically generating colors for new classes
    COLOR_PALETTE = [
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 165, 0),    # Orange
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (255, 192, 203),  # Pink
        (165, 42, 42),    # Brown
        (0, 255, 255),    # Yellow
    ]

    def __init__(self,
                 device: str = 'cuda',
                 target_classes: List[str] = None,
                 sam3_config: Optional[Dict[str, Any]] = None,
                 gpus_to_use: Optional[List[int]] = None,
                 confidence_threshold: float = 0.3,
                 class_text_prompts: Optional[Dict[str, str]] = None):
        """
        Initialize SAM3-based object detector using official video predictor API.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            target_classes: List of target classes to detect (e.g., ["soldier", "tank", "truck"])
            sam3_config: Optional SAM3 configuration dict
            gpus_to_use: List of GPU indices to use for multi-GPU inference
            confidence_threshold: Minimum confidence for detections (default: 0.3)
            class_text_prompts: Optional dict mapping class names to custom text prompts
                               e.g., {"truck": "military truck", "vehicle": "armored vehicle"}
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold

        # Store custom text prompts if provided
        self.custom_text_prompts = class_text_prompts or {}

        # Map target classes to text prompts
        self.target_classes = target_classes or list(self.DEFAULT_OBJECT_CLASSES.keys())
        self.text_prompts = self._build_text_prompts(self.target_classes)

        # Build class colors dynamically
        self.class_colors = self._build_class_colors(self.target_classes)

        # SAM3 configuration
        self.sam3_config = sam3_config or {}
        self.iou_threshold = self.sam3_config.get("iou_threshold", 0.5)

        # GPU configuration
        if gpus_to_use is None:
            self.gpus_to_use = [0] if torch.cuda.is_available() else []
        else:
            self.gpus_to_use = gpus_to_use

        # Model state
        self.predictor = None
        self.tracking_enabled = False

        # Initialize SAM3 if available
        if SAM3_AVAILABLE:
            self._init_sam3()
        else:
            print("SAM3 not available. Install with: cd SAM3_tracking/sam3 && pip install -e .")

        print(f"Object detector initialized")
        print(f"  Device: {self.device}")
        print(f"  GPUs: {self.gpus_to_use}")
        print(f"  Target classes: {self.target_classes}")
        print(f"  Text prompts: {self.text_prompts}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Tracking enabled: {self.tracking_enabled}")

    def _build_text_prompts(self, target_classes: List[str]) -> Dict[str, str]:
        """
        Build text prompts from target classes for SAM3.

        Priority for text prompt selection:
        1. Custom text prompts passed via class_text_prompts parameter
        2. Default prompts from DEFAULT_OBJECT_CLASSES
        3. Use class name as prompt if no custom prompt defined
        """
        prompts = {}
        for cls in target_classes:
            cls_lower = cls.lower()
            # Priority: custom prompts > default prompts > class name
            if cls_lower in self.custom_text_prompts:
                prompts[cls_lower] = self.custom_text_prompts[cls_lower]
            elif cls_lower in self.DEFAULT_OBJECT_CLASSES:
                prompts[cls_lower] = self.DEFAULT_OBJECT_CLASSES[cls_lower]
            else:
                # For new classes not in defaults, use class name as prompt
                prompts[cls_lower] = cls_lower
        return prompts

    def _build_class_colors(self, target_classes: List[str]) -> Dict[str, Tuple[int, int, int]]:
        """
        Build class colors dynamically for visualization.

        Uses default colors for known classes, generates colors from palette for new classes.
        """
        colors = {}
        palette_idx = 0
        for cls in target_classes:
            cls_lower = cls.lower()
            if cls_lower in self.DEFAULT_CLASS_COLORS:
                colors[cls_lower] = self.DEFAULT_CLASS_COLORS[cls_lower]
            else:
                # Assign color from palette for new classes
                colors[cls_lower] = self.COLOR_PALETTE[palette_idx % len(self.COLOR_PALETTE)]
                palette_idx += 1
        return colors

    def _init_sam3(self):
        """Initialize SAM3 video predictor."""
        try:
            print("Loading SAM3 video predictor...")
            print(f"  GPUs to use: {self.gpus_to_use}")

            # Build the SAM3 video predictor
            self.predictor = build_sam3_video_predictor(
                gpus_to_use=self.gpus_to_use if self.gpus_to_use else None,
                apply_temporal_disambiguation=True,
            )

            self.tracking_enabled = True
            print("SAM3 video predictor initialized successfully")

        except Exception as e:
            print(f"Failed to initialize SAM3 video predictor: {e}")
            import traceback
            traceback.print_exc()
            self.predictor = None
            self.tracking_enabled = False

    def _clear_cuda_cache(self):
        """
        Clear CUDA cache to help prevent dtype mismatches between sessions.

        Note: This alone doesn't fix the bfloat16/float32 mismatch - we also
        need to use torch.autocast("cuda", dtype=torch.bfloat16) around
        all predictor calls.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _get_autocast_context(self):
        """
        Get the appropriate autocast context for SAM3 inference.

        SAM3 internally uses @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        on key methods. When processing multiple object classes sequentially,
        inconsistent dtype handling can cause "Input type (c10::BFloat16) and
        bias type (float) should be the same" errors.

        By wrapping all predictor calls with autocast, we ensure the entire
        inference chain uses consistent bfloat16 dtype.

        Reference: https://github.com/facebookresearch/sam2/issues/577
        """
        if torch.cuda.is_available() and self.device == 'cuda':
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            # For CPU, use float32 (bfloat16 may not be well supported)
            return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)

    def _detect_single_class_in_frame(self, frame_path: str, class_name: str,
                                      text_prompt: str, width: int, height: int,
                                      obj_id_offset: int = 0) -> List[Dict]:
        """
        Detect objects of a single class in a frame using a dedicated session.

        SAM3 can only process one text prompt per session due to contrastive
        learning alignment. This method handles a single class properly.

        Args:
            frame_path: Path to the frame image file
            class_name: Class name for labeling detections
            text_prompt: Text prompt for SAM3
            width: Frame width for coordinate conversion
            height: Frame height for coordinate conversion
            obj_id_offset: Offset to add to object IDs for uniqueness across classes

        Returns:
            List of detection dictionaries
        """
        predictions = []

        # Clear CUDA cache before starting a new session
        self._clear_cuda_cache()

        try:
            # Use autocast with bfloat16 to match SAM3's internal dtype handling
            # This prevents "Input type (c10::BFloat16) and bias type (float)" errors
            with self._get_autocast_context():
                # Start a NEW session for this class
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=frame_path,
                    )
                )
                session_id = response["session_id"]

                try:
                    # Add text prompt for this class
                    response = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=text_prompt,
                        )
                    )

                    # Extract detections from response
                    outputs = response.get("outputs", {})
                    if outputs is not None:
                        obj_ids = outputs.get("out_obj_ids", np.array([]))
                        boxes_xywh = outputs.get("out_boxes_xywh", np.array([]))
                        probs = outputs.get("out_probs", np.array([]))

                        if len(obj_ids) > 0:
                            for obj_id, bbox, score in zip(obj_ids, boxes_xywh, probs):
                                if score >= self.confidence_threshold:
                                    # Convert normalized XYWH to pixel coordinates
                                    x, y, bw, bh = bbox
                                    x_pixel = x * width
                                    y_pixel = y * height
                                    w_pixel = bw * width
                                    h_pixel = bh * height

                                    # Convert to center format (Roboflow-compatible)
                                    x_center = x_pixel + w_pixel / 2
                                    y_center = y_pixel + h_pixel / 2

                                    predictions.append({
                                        "class": class_name,
                                        "confidence": float(score),
                                        "x": float(x_center),
                                        "y": float(y_center),
                                        "width": float(w_pixel),
                                        "height": float(h_pixel),
                                        "object_id": int(obj_id) + obj_id_offset
                                    })

                finally:
                    # ALWAYS close session to free resources
                    self.predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                        )
                    )

        except Exception as e:
            print(f"  Warning: Error detecting {class_name}: {e}")

        return predictions

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in a single frame using SAM3.

        IMPORTANT: Each class is processed in a SEPARATE session because SAM3
        can only handle one text prompt at a time.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Detection results in Roboflow-compatible format
        """
        if not self.tracking_enabled or self.predictor is None:
            return {"predictions": [], "image": {"width": frame.shape[1], "height": frame.shape[0]}}

        try:
            h, w = frame.shape[:2]

            # Save frame to temporary file (SAM3 expects file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, frame)

            try:
                all_predictions = []
                obj_id_offset = 0

                # Process EACH class in a SEPARATE session
                # SAM3 limitation: can only process one text embedding at a time
                for class_name, text_prompt in self.text_prompts.items():
                    class_predictions = self._detect_single_class_in_frame(
                        frame_path=tmp_path,
                        class_name=class_name,
                        text_prompt=text_prompt,
                        width=w,
                        height=h,
                        obj_id_offset=obj_id_offset
                    )
                    all_predictions.extend(class_predictions)

                    # Update offset for next class to ensure unique object IDs
                    if class_predictions:
                        max_id = max(p["object_id"] for p in class_predictions)
                        obj_id_offset = max_id + 1

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            # Deduplicate overlapping detections across classes
            all_predictions = self._deduplicate_predictions(all_predictions, w, h)

            return {
                "predictions": all_predictions,
                "image": {"width": w, "height": h}
            }

        except Exception as e:
            print(f"Error detecting objects in frame: {e}")
            import traceback
            traceback.print_exc()
            return {"predictions": [], "image": {"width": frame.shape[1], "height": frame.shape[0]}}

    def _deduplicate_predictions(self, predictions: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Deduplicate overlapping predictions using IoU-based NMS."""
        if len(predictions) <= 1:
            return predictions

        # Sort by confidence
        sorted_preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        keep = []
        suppressed = set()

        for i, pred1 in enumerate(sorted_preds):
            if i in suppressed:
                continue

            keep.append(pred1)

            # Convert to normalized XYWH for IoU (from center format)
            box1 = [
                (pred1["x"] - pred1["width"] / 2) / img_w,
                (pred1["y"] - pred1["height"] / 2) / img_h,
                pred1["width"] / img_w,
                pred1["height"] / img_h
            ]

            for j in range(i + 1, len(sorted_preds)):
                if j in suppressed:
                    continue

                pred2 = sorted_preds[j]
                box2 = [
                    (pred2["x"] - pred2["width"] / 2) / img_w,
                    (pred2["y"] - pred2["height"] / 2) / img_h,
                    pred2["width"] / img_w,
                    pred2["height"] / img_h
                ]

                iou = compute_iou(box1, box2)
                if iou > self.iou_threshold:
                    suppressed.add(j)

        return keep

    def _deduplicate_tracking_detections(self,
                                         all_frame_results: Dict[int, List[Dict]],
                                         unique_objects: Dict[str, set],
                                         width: int,
                                         height: int) -> Tuple[Dict[int, List[Dict]], Dict[str, set]]:
        """
        Deduplicate overlapping detections across different classes in tracking results.

        When the same physical object is detected by multiple class prompts (e.g., a tank
        detected as both "tank" and "truck"), this method keeps only the first detected class
        and removes duplicates based on IoU overlap.

        Args:
            all_frame_results: Dict mapping frame_idx to list of detections
            unique_objects: Dict mapping class_name to set of object_ids
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Tuple of (deduplicated_frame_results, updated_unique_objects)
        """
        if not all_frame_results:
            return all_frame_results, unique_objects

        # Track which object_ids to suppress (mapped to the class they belong to)
        suppressed_object_ids = set()  # Set of (class_name, object_id) tuples to remove

        # Build a mapping of object_id -> (class_name, representative_bbox)
        # Use the first frame where each object appears to get its representative bbox
        object_info = {}  # object_id -> {"class": class_name, "bbox": [x1, y1, x2, y2], "first_frame": frame_idx}

        # First pass: collect information about all objects
        for frame_idx in sorted(all_frame_results.keys()):
            for det in all_frame_results[frame_idx]:
                obj_id = det.get("object_id")
                class_name = det.get("class")
                if obj_id is not None and obj_id not in object_info:
                    # Store first occurrence of this object
                    # Detection format from tracking: 'bbox': [x1, y1, x2, y2] (XYXY pixel coords)
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    object_info[obj_id] = {
                        "class": class_name,
                        "bbox": bbox,  # XYXY format: [x1, y1, x2, y2]
                        "first_frame": frame_idx
                    }

        # Second pass: find overlapping objects across different classes
        object_ids = list(object_info.keys())
        for i, obj_id1 in enumerate(object_ids):
            info1 = object_info[obj_id1]

            # bbox is in XYXY format: [x1, y1, x2, y2] (pixel coordinates)
            # Convert to normalized XYWH format for IoU calculation
            x1_1, y1_1, x2_1, y2_1 = info1["bbox"]
            box1 = [
                x1_1 / width,           # x (normalized)
                y1_1 / height,          # y (normalized)
                (x2_1 - x1_1) / width,  # width (normalized)
                (y2_1 - y1_1) / height  # height (normalized)
            ]

            for j in range(i + 1, len(object_ids)):
                obj_id2 = object_ids[j]
                info2 = object_info[obj_id2]

                # Only check for duplicates between DIFFERENT classes
                if info1["class"] == info2["class"]:
                    continue

                x1_2, y1_2, x2_2, y2_2 = info2["bbox"]
                box2 = [
                    x1_2 / width,           # x (normalized)
                    y1_2 / height,          # y (normalized)
                    (x2_2 - x1_2) / width,  # width (normalized)
                    (y2_2 - y1_2) / height  # height (normalized)
                ]

                iou = compute_iou(box1, box2)
                if iou > self.iou_threshold:
                    # Keep the one detected first (lower first_frame), suppress the other
                    if info1["first_frame"] <= info2["first_frame"]:
                        # Suppress obj_id2
                        suppressed_object_ids.add(obj_id2)
                        print(f"    Dedup: Suppressing {info2['class']} (id={obj_id2}) - "
                              f"duplicate of {info1['class']} (id={obj_id1}), IoU={iou:.2f}")
                    else:
                        # Suppress obj_id1
                        suppressed_object_ids.add(obj_id1)
                        print(f"    Dedup: Suppressing {info1['class']} (id={obj_id1}) - "
                              f"duplicate of {info2['class']} (id={obj_id2}), IoU={iou:.2f}")

        # Third pass: filter out suppressed detections from all frames
        if suppressed_object_ids:
            print(f"    Dedup: Removing {len(suppressed_object_ids)} duplicate object(s)")

            filtered_frame_results = defaultdict(list)
            for frame_idx, detections in all_frame_results.items():
                for det in detections:
                    if det.get("object_id") not in suppressed_object_ids:
                        filtered_frame_results[frame_idx].append(det)

            # Update unique_objects counts
            filtered_unique_objects = defaultdict(set)
            for obj_id, info in object_info.items():
                if obj_id not in suppressed_object_ids:
                    filtered_unique_objects[info["class"]].add(obj_id)

            return dict(filtered_frame_results), dict(filtered_unique_objects)

        return all_frame_results, unique_objects

    def process_segment(self, video_path: str,
                       start_time: float,
                       end_time: float,
                       sample_rate: int = 1) -> Dict[str, Any]:
        """
        Process a video segment and detect objects (no tracking).

        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            sample_rate: Process every Nth frame (1 = every frame)

        Returns:
            Dict with detections and statistics
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            all_detections = []
            frame_idx = 0
            current_frame_num = start_frame

            while current_frame_num < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    results = self.detect_frame(frame)
                    frame_detections = {
                        "frame_number": current_frame_num,
                        "timestamp": current_frame_num / fps,
                        "predictions": results["predictions"]
                    }
                    all_detections.append(frame_detections)

                frame_idx += 1
                current_frame_num += 1

            cap.release()

            statistics = self._compute_statistics(all_detections)

            return {
                "segment_start": start_time,
                "segment_end": end_time,
                "num_frames_processed": len(all_detections),
                "frame_detections": all_detections,
                "statistics": statistics
            }

        except Exception as e:
            print(f"Error processing segment: {e}")
            return {
                "segment_start": start_time,
                "segment_end": end_time,
                "num_frames_processed": 0,
                "frame_detections": [],
                "statistics": {}
            }

    def process_segment_with_tracking(self,
                                     video_path: str,
                                     start_time: float,
                                     end_time: float,
                                     output_video_path: Optional[str] = None,
                                     video_id: Optional[str] = None,
                                     segment_id: Optional[int] = None,
                                     save_tracking_video: bool = True,
                                     collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a video segment with SAM3 tracking.

        IMPORTANT: SAM3 can only track one text prompt per session. This method
        processes each class separately and merges the results.

        Workflow for each class:
        1. start_session → add_prompt → propagate_in_video → close_session
        2. Merge results from all classes with unique object IDs

        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            output_video_path: Optional explicit path to save annotated video
            video_id: Video ID in local_videodb
            segment_id: Segment ID for naming the output file
            save_tracking_video: Whether to save the annotated tracking video
            collection_name: Collection name to save tracking video to

        Returns:
            Dict with tracking results and statistics
        """
        if not self.tracking_enabled:
            print("SAM3 tracking not available. Falling back to detection-only mode.")
            return self.process_segment(video_path, start_time, end_time)

        try:
            print(f"Processing segment with SAM3 tracking...")
            print(f"   Video: {video_path}")
            print(f"   Time: {start_time:.1f}s - {end_time:.1f}s")

            # Extract segment to temporary video file
            temp_segment_path = self._extract_segment_to_temp(video_path, start_time, end_time)

            if not temp_segment_path:
                return {"error": "Failed to extract video segment"}

            # Determine output video path
            final_output_path = None
            if save_tracking_video:
                if output_video_path:
                    final_output_path = output_video_path
                elif video_id:
                    target_collection = collection_name or "battlefield_reconnaissance"
                    final_output_path = self._construct_tracking_output_path(
                        video_id, segment_id, start_time, end_time, target_collection
                    )

            # Run SAM3 tracking (processes each class separately)
            tracking_result = self._run_sam3_tracking(
                temp_segment_path,
                output_video_path=final_output_path
            )

            # Clean up temporary file
            try:
                os.unlink(temp_segment_path)
            except:
                pass

            # Add segment timing
            tracking_result["segment_start"] = start_time
            tracking_result["segment_end"] = end_time

            if final_output_path and os.path.exists(final_output_path):
                tracking_result["tracking_video_path"] = final_output_path

            return tracking_result

        except Exception as e:
            print(f"Error in tracking segment: {e}")
            import traceback
            traceback.print_exc()
            return {
                "segment_start": start_time,
                "segment_end": end_time,
                "error": str(e)
            }

    def _extract_segment_to_temp(self, video_path: str, start_time: float, end_time: float) -> Optional[str]:
        """Extract video segment to temporary file using ffmpeg."""
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)

            duration = end_time - start_time
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'copy',
                '-y',
                temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"ffmpeg error: {result.stderr}")
                os.unlink(temp_path)
                return None

            return temp_path

        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None

    def _track_single_class_in_video(self, video_path: str, class_name: str,
                                     text_prompt: str, width: int, height: int,
                                     obj_id_offset: int = 0) -> Tuple[Dict[int, List[Dict]], set]:
        """
        Track objects of a single class through a video using a dedicated session.

        SAM3 can only process one text prompt per session. This method handles
        a single class with proper session lifecycle:
        start_session → add_prompt → propagate_in_video → close_session

        Args:
            video_path: Path to video file
            class_name: Class name for labeling detections
            text_prompt: Text prompt for SAM3
            width: Video width for coordinate conversion
            height: Video height for coordinate conversion
            obj_id_offset: Offset to add to object IDs for uniqueness across classes

        Returns:
            Tuple of (frame_results dict, set of unique object IDs)
        """
        frame_results = defaultdict(list)
        unique_obj_ids = set()

        # Clear CUDA cache before starting a new session
        self._clear_cuda_cache()

        try:
            # Use autocast with bfloat16 to match SAM3's internal dtype handling
            # This prevents "Input type (c10::BFloat16) and bias type (float)" errors
            with self._get_autocast_context():
                # Start a NEW session for this class
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=video_path,
                    )
                )
                session_id = response["session_id"]

                try:
                    # Add text prompt for this class on frame 0
                    response = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=text_prompt,
                        )
                    )

                    # Process initial frame detections if any
                    outputs = response.get("outputs", {})
                    if outputs is not None and len(outputs.get("out_obj_ids", [])) > 0:
                        # Frame 0 results from add_prompt
                        self._extract_frame_detections(
                            frame_idx=0,
                            outputs=outputs,
                            class_name=class_name,
                            width=width,
                            height=height,
                            obj_id_offset=obj_id_offset,
                            frame_results=frame_results,
                            unique_obj_ids=unique_obj_ids
                        )

                    # Propagate through the entire video
                    for response in self.predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            propagation_direction="forward",
                        )
                    ):
                        frame_idx = response["frame_index"]
                        outputs = response.get("outputs", {})

                        if outputs is not None:
                            self._extract_frame_detections(
                                frame_idx=frame_idx,
                                outputs=outputs,
                                class_name=class_name,
                                width=width,
                                height=height,
                                obj_id_offset=obj_id_offset,
                                frame_results=frame_results,
                                unique_obj_ids=unique_obj_ids
                            )

                finally:
                    # ALWAYS close session to free resources
                    self.predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                        )
                    )

        except Exception as e:
            print(f"  Warning: Error tracking {class_name}: {e}")
            import traceback
            traceback.print_exc()

        return frame_results, unique_obj_ids

    def _extract_frame_detections(self, frame_idx: int, outputs: Dict,
                                  class_name: str, width: int, height: int,
                                  obj_id_offset: int,
                                  frame_results: Dict[int, List[Dict]],
                                  unique_obj_ids: set):
        """Extract detections from SAM3 output and add to frame_results."""
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        boxes_xywh = outputs.get("out_boxes_xywh", np.array([]))
        probs = outputs.get("out_probs", np.array([]))
        masks = outputs.get("out_binary_masks", np.array([]))

        if len(obj_ids) == 0:
            return

        for i, (obj_id, bbox, score) in enumerate(zip(obj_ids, boxes_xywh, probs)):
            if score >= self.confidence_threshold:
                # Convert normalized XYWH to pixel XYXY
                x, y, w_box, h_box = bbox
                x1 = int(x * width)
                y1 = int(y * height)
                x2 = int((x + w_box) * width)
                y2 = int((y + h_box) * height)

                adjusted_obj_id = int(obj_id) + obj_id_offset

                det = {
                    'object_id': adjusted_obj_id,
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score)
                }

                # Add mask if available
                if len(masks) > i:
                    det['mask'] = masks[i]

                frame_results[frame_idx].append(det)
                unique_obj_ids.add(adjusted_obj_id)

    def _run_sam3_tracking(self, video_path: str, output_video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run SAM3 detection and tracking on a video file.

        CRITICAL: SAM3 can only process ONE text prompt per session. This method
        processes each class in a SEPARATE session and merges results.

        For each class:
        1. start_session: Initialize with video
        2. add_prompt: Add text prompt for this class
        3. propagate_in_video: Track objects through all frames
        4. close_session: Clean up resources

        Args:
            video_path: Path to video file
            output_video_path: Optional path to save tracking video

        Returns:
            Dict with tracking results
        """
        try:
            print(f"  Loading video...")

            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Failed to open video", "num_frames_processed": 0}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Read all frames for visualization
            frames_bgr = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
            cap.release()

            print(f"  Loaded {len(frames_bgr)} frames ({fps:.1f} FPS, {width}x{height})")

            if len(frames_bgr) == 0:
                return {"error": "No frames loaded", "num_frames_processed": 0}

            # Collect all detections with tracking
            # Key insight: Process each class in a SEPARATE session
            all_frame_results = defaultdict(list)  # frame_idx -> list of detections
            unique_objects = defaultdict(set)  # class_name -> set of object_ids
            obj_id_offset = 0

            # Process EACH class in a SEPARATE session
            # This is required because SAM3 can only process one text embedding at a time
            for class_name, text_prompt in self.text_prompts.items():
                print(f"  Tracking {class_name}...")

                class_frame_results, class_obj_ids = self._track_single_class_in_video(
                    video_path=video_path,
                    class_name=class_name,
                    text_prompt=text_prompt,
                    width=width,
                    height=height,
                    obj_id_offset=obj_id_offset
                )

                # Merge results
                for frame_idx, detections in class_frame_results.items():
                    all_frame_results[frame_idx].extend(detections)

                unique_objects[class_name] = class_obj_ids
                print(f"     Found {len(class_obj_ids)} unique {class_name}(s)")

                # Update offset for next class to ensure unique object IDs
                if class_obj_ids:
                    obj_id_offset = max(class_obj_ids) + 1

            # Deduplicate overlapping detections across different classes
            # This prevents the same object from being detected as multiple classes
            print(f"  Deduplicating cross-class detections...")
            all_frame_results, unique_objects = self._deduplicate_tracking_detections(
                all_frame_results, unique_objects, width, height
            )

            # Count unique objects per class (after deduplication)
            object_counts = {cls: len(ids) for cls, ids in unique_objects.items()}
            total_objects = sum(object_counts.values())

            print(f"  Total unique objects tracked (after dedup): {total_objects}")
            for cls, count in object_counts.items():
                print(f"     - {cls}: {count}")

            # Save tracking video if requested
            if output_video_path and total_objects > 0:
                print(f"  Saving annotated video...")
                self._save_annotated_video(
                    frames_bgr, all_frame_results, output_video_path, fps, object_counts
                )

            # Convert to tracking data format
            tracking_data = []
            for frame_idx in sorted(all_frame_results.keys()):
                frame_dets = all_frame_results[frame_idx]
                if frame_dets:
                    tracking_data.append({
                        "frame_number": frame_idx,
                        "timestamp": frame_idx / fps,
                        "num_objects": len(frame_dets),
                        "detections": frame_dets
                    })

            return {
                "num_objects_tracked": total_objects,
                "num_frames_processed": len(frames_bgr),
                "object_counts": object_counts,
                "tracking_data": tracking_data,
                "fps": fps
            }

        except Exception as e:
            print(f"Error in SAM3 tracking: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "num_frames_processed": 0}

    def _save_annotated_video(self, frames_bgr: List[np.ndarray],
                             frame_results: Dict[int, List[Dict]],
                             output_path: str,
                             fps: float,
                             object_counts: Dict[str, int]):
        """Save annotated tracking video."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if len(frames_bgr) == 0:
                print("  No frames to save")
                return

            height, width = frames_bgr[0].shape[:2]
            temp_path = output_path.parent / f"_temp_{output_path.name}"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

            for frame_idx, frame in enumerate(tqdm(frames_bgr, desc="Rendering")):
                frame = frame.copy()

                # Draw detections for this frame
                if frame_idx in frame_results:
                    for det in frame_results[frame_idx]:
                        x1, y1, x2, y2 = det['bbox']
                        class_name = det['class']
                        obj_id = det['object_id']
                        confidence = det['confidence']

                        color = self.class_colors.get(class_name, (255, 255, 255))

                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = f"{class_name} ID:{obj_id} {confidence:.2f}"
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw summary overlay
                y_offset = 30
                for class_name, count in object_counts.items():
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    text = f"{class_name}: {count}"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset += 30

                # Frame number
                cv2.putText(frame, f"Frame {frame_idx}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                writer.write(frame)

            writer.release()

            # Re-encode with ffmpeg for better compatibility
            try:
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', str(temp_path), '-c:v', 'libx264', '-preset', 'fast', str(output_path)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    os.unlink(str(temp_path))
                    print(f"  Tracking video saved: {output_path}")
                else:
                    os.rename(str(temp_path), str(output_path))
                    print(f"  Tracking video saved: {output_path}")
            except FileNotFoundError:
                os.rename(str(temp_path), str(output_path))
                print(f"  Tracking video saved: {output_path}")

        except Exception as e:
            print(f"  Failed to save video: {e}")
            import traceback
            traceback.print_exc()

    def _construct_tracking_output_path(self, video_id: str, segment_id: Optional[int],
                                       start_time: float, end_time: float,
                                       collection_name: str = "battlefield_reconnaissance") -> str:
        """Construct output path for tracking video."""
        storage_base = Path("./data/local_videodb")
        video_dir = storage_base / "collections" / collection_name / "videos" / video_id
        tracking_dir = video_dir / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)

        if segment_id is not None:
            filename = f"segment_{segment_id:03d}_tracked.mp4"
        else:
            filename = f"tracked_{int(start_time)}s_to_{int(end_time)}s.mp4"

        output_path = tracking_dir / filename
        print(f"  Tracking video path: {output_path}")
        return str(output_path)

    def _compute_statistics(self, all_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics across all frame detections."""
        try:
            object_counts = {cls: 0 for cls in self.target_classes}
            total_detections = 0

            for frame_det in all_detections:
                for pred in frame_det["predictions"]:
                    obj_class = pred["class"]
                    if obj_class in object_counts:
                        object_counts[obj_class] += 1
                        total_detections += 1

            num_frames = len(all_detections)
            avg_objects_per_frame = total_detections / num_frames if num_frames > 0 else 0

            representative_objects = self._get_representative_objects(all_detections)

            return {
                "object_counts": object_counts,
                "total_detections": total_detections,
                "avg_objects_per_frame": avg_objects_per_frame,
                "representative_objects": representative_objects
            }

        except Exception as e:
            print(f"Error computing statistics: {e}")
            return {}

    def _get_representative_objects(self, all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get the most confident detection for each class."""
        try:
            best_detections = {}

            for frame_det in all_detections:
                for pred in frame_det["predictions"]:
                    obj_class = pred["class"]
                    confidence = pred["confidence"]

                    if obj_class not in best_detections or confidence > best_detections[obj_class]["confidence"]:
                        best_detections[obj_class] = {
                            "class": obj_class,
                            "confidence": confidence,
                            "x": pred["x"],
                            "y": pred["y"],
                            "width": pred["width"],
                            "height": pred["height"],
                            "timestamp": frame_det["timestamp"]
                        }

            return list(best_detections.values())

        except Exception as e:
            print(f"Error getting representative objects: {e}")
            return []

    def extract_object_crops(self, frame: np.ndarray,
                            detections: List[Dict[str, Any]],
                            padding: int = 10) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract cropped regions for detected objects."""
        try:
            crops = []
            h, w = frame.shape[:2]

            for det in detections:
                x_center = det["x"]
                y_center = det["y"]
                width = det["width"]
                height = det["height"]

                x1 = int(x_center - width / 2) - padding
                y1 = int(y_center - height / 2) - padding
                x2 = int(x_center + width / 2) + padding
                y2 = int(y_center + height / 2) + padding

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crops.append((crop, det))

            return crops

        except Exception as e:
            print(f"Error extracting object crops: {e}")
            return []

    def get_middle_frame(self, video_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Get the middle frame of a segment."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            middle_time = (start_time + end_time) / 2
            middle_frame_num = int(middle_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
            ret, frame = cap.read()

            cap.release()

            if ret:
                return frame
            else:
                return None

        except Exception as e:
            print(f"Error getting middle frame: {e}")
            return None

    def process_segment_simple(self, video_path: str,
                              start_time: float,
                              end_time: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Simplified segment processing: detect on middle frame only.

        Args:
            video_path: Path to video file
            start_time: Segment start time
            end_time: Segment end time

        Returns:
            Tuple of (detections, middle_frame)
        """
        try:
            frame = self.get_middle_frame(video_path, start_time, end_time)

            if frame is None:
                return [], None

            results = self.detect_frame(frame)

            return results["predictions"], frame

        except Exception as e:
            print(f"Error in simple segment processing: {e}")
            return [], None

    def get_target_classes(self) -> List[str]:
        """
        Get the list of target classes configured for detection.

        Returns:
            List of class names (e.g., ["soldier", "tank", "truck"])
        """
        return self.target_classes.copy()

    def get_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get the color mapping for all target classes.

        Returns:
            Dict mapping class name to BGR color tuple
        """
        return self.class_colors.copy()

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.predictor is not None:
                self.predictor.shutdown()
                self.predictor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Object detection processor cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
