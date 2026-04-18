from object_detector import LLMDetDetector
import supervision as sv
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_camera_predictor
from IPython.display import Video


def filter_segments_by_distance(mask: np.ndarray, distance_threshold: float = 300) -> np.ndarray:
    """
    Keeps the main segment and removes segments farther than distance_threshold.

    Args:
        mask (np.ndarray): Boolean mask.
        distance_threshold (float): Maximum allowed distance from the main segment.

    Returns:
        np.ndarray: Boolean mask after filtering.
    """
    assert mask.dtype == bool, "Input mask must be boolean."
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask.copy()
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_centroid = centroids[main_label]
    filtered_mask = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        centroid = centroids[label]
        dist = np.linalg.norm(centroid - main_centroid)
        if label == main_label or dist <= distance_threshold:
            filtered_mask[labels == label] = True
    return filtered_mask


def main(SOURCE_VIDEO_PATH,
         SAM2_CHECKPOINT,
         SAM2_CONFIG,
         TARGET_VIDEO_PATH, 
         TARGET_VIDEO_COMPRESSED_PATH):
    detection_model = LLMDetDetector()

    COLOR = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    frame_generator = sv.get_video_frames_generator("/content/source/arma4.mp4")
    frame = next(frame_generator)

    predictor = build_sam2_camera_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)

    mask_annotator = sv.MaskAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
        opacity=0.5)
    box_annotator = sv.BoxAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
        thickness=2
    )


    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)

    result = detection_model.detect(frame, conf_threshold=0.15, nms_threshold=0.35)
    detections = sv.Detections.from_inference(result)

    TRACKE_ID = list(range(1, len(detections.class_id) + 1))
    # print(TRACKE_ID)
    detections.tracker_id = TRACKE_ID

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    # sv.plot_image(annotated_frame)

    # we prompt SAM2.1 using RF-DETR model detections

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.load_first_frame(frame)

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            xyxy = np.array([xyxy])

            _, object_ids, mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=tracker_id,
                bbox=xyxy
            )

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = predictor.track(frame)
            tracker_ids = np.array(tracker_ids)
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)

            masks = np.array([
                filter_segments_by_distance(mask, distance_threshold=300)
                for mask
                in masks
            ])

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=tracker_ids
            )

            annotated_frame = frame.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            return annotated_frame

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback,
        show_progress=True
    )

    # !ffmpeg -y -loglevel error -i {TARGET_VIDEO_PATH} -vcodec libx264 -crf 28 {TARGET_VIDEO_COMPRESSED_PATH}
    # Video(TARGET_VIDEO_COMPRESSED_PATH, embed=True, width=1080)