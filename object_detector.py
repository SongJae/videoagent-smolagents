import cv2
import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from scipy.optimize import linear_sum_assignment
from typing import List

# ==============================================================================
# Detection Module: LLMDet Zero-Shot Detector
# ==============================================================================
class LLMDetDetector:
    def __init__(self, model_name="iSEE-Laboratory/llmdet_large", device='cuda'):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        # Target classes
        self.text_queries = ["truck", "tank", "soldier"]
        self.text_batch = [self.text_queries]


    def apply_nms(self, boxes, scores, labels, iou_threshold=0.5):
        """Apply NMS using torchvision (faster)"""
        if len(boxes) == 0:
            return [], [], []

        # Ensure tensors
        if not torch.is_tensor(boxes):
            boxes = torch.tensor(boxes).to(self.device)
        if not torch.is_tensor(scores):
            scores = torch.tensor(scores).to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).to(self.device)

        keep_indices = []
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]

            if len(class_boxes) > 0:
                keep = nms(class_boxes, class_scores, iou_threshold)
                keep_indices.extend(class_indices[keep].tolist())

        keep_indices = sorted(keep_indices)
        return (boxes[keep_indices],
                scores[keep_indices],
                labels[keep_indices])


    def detect(self, frame: np.ndarray, conf_threshold=0.25, nms_threshold=0.5):
        # Convert BGR frame to PIL RGB image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        img_height, img_width = frame.shape[:2]

        # Prepare inputs
        inputs = self.processor(text=self.text_batch, images=frame_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Post-process detections
        target_size = torch.Tensor([frame_pil.size[::-1]])  # (height, width)
        results = self.processor.post_process_grounded_object_detection(outputs=outputs,
                                                               threshold=conf_threshold,
                                                               target_sizes=target_size)
        predictions = []
        if len(results) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]

            labels_array = np.array(labels)
            mask = np.array([label in self.text_queries for label in labels])

            labels = np.array(labels)[mask].tolist()
            labels = [self.text_queries.index(label) for label in labels]

            boxes = boxes[mask]
            scores = scores[mask]

            boxes, scores, labels = self.apply_nms(
                boxes, scores, labels, iou_threshold=nms_threshold
            )

            for box, score, label_idx in zip(boxes, scores, labels):
                score = score.item()
                label_idx = int(label_idx.item())
                if score < conf_threshold:
                    continue

                x1, y1, x2, y2 = box.tolist()

                # Calculate center coordinates and dimensions
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                predictions.append({
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height,
                    "confidence": score,
                    "class": self.text_queries[label_idx],
                    "class_id": label_idx
                })

        # Return in Roboflow Inference format
        return {
            "predictions": predictions,
            "image": {
                "width": img_width,
                "height": img_height
            }
        }