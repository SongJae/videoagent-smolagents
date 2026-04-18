"""
Compare original vs improved SAM3 video detector
Helps visualize the benefits of cross-class NMS and false positive filtering
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import argparse


def load_results(json_path: str) -> Dict[str, Any]:
    """Load detection results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_duplicates(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze potential duplicate detections
    (Objects with high IoU across different classes)
    """
    duplicate_candidates = []
    total_detections = 0

    frames = results.get('frames', {})

    for frame_idx, frame_data in frames.items():
        objects = frame_data.get('objects', [])
        total_detections += len(objects)

        # Check for potential duplicates (same position, different class)
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if obj1['class'] != obj2['class']:
                    # Calculate IoU
                    box1 = obj1['bbox']
                    box2 = obj2['bbox']
                    iou = calculate_iou(box1, box2)

                    if iou > 0.3:  # Potential duplicate
                        duplicate_candidates.append({
                            'frame': frame_idx,
                            'object1': f"{obj1['class']} (ID: {obj1['object_id']})",
                            'object2': f"{obj2['class']} (ID: {obj2['object_id']})",
                            'iou': iou,
                            'confidence1': obj1['confidence'],
                            'confidence2': obj2['confidence']
                        })

    return {
        'total_detections': total_detections,
        'duplicate_candidates': duplicate_candidates,
        'num_duplicates': len(duplicate_candidates)
    }


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def analyze_false_positives(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze potential false positives
    (Low confidence, small area, etc.)
    """
    suspicious_detections = {
        'low_confidence': [],
        'very_small': [],
        'very_large': []
    }

    frames = results.get('frames', {})

    for frame_idx, frame_data in frames.items():
        for obj in frame_data.get('objects', []):
            # Check confidence
            if obj['confidence'] < 0.4:
                suspicious_detections['low_confidence'].append({
                    'frame': frame_idx,
                    'class': obj['class'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox']
                })

            # Check area
            bbox = obj['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if area < 100:
                suspicious_detections['very_small'].append({
                    'frame': frame_idx,
                    'class': obj['class'],
                    'area': area,
                    'confidence': obj['confidence']
                })
            elif area > 500000:
                suspicious_detections['very_large'].append({
                    'frame': frame_idx,
                    'class': obj['class'],
                    'area': area,
                    'confidence': obj['confidence']
                })

    return suspicious_detections


def print_comparison(original_results: Dict[str, Any], improved_results: Dict[str, Any]):
    """Print detailed comparison between original and improved results"""

    print("="*80)
    print("SAM3 DETECTOR COMPARISON: Original vs Improved")
    print("="*80)

    # Basic counts
    print("\n1. OBJECT COUNTS")
    print("-"*80)
    print(f"{'Class':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
    print("-"*80)

    all_classes = set(original_results.get('object_counts', {}).keys()) | \
                  set(improved_results.get('object_counts', {}).keys())

    for class_name in sorted(all_classes):
        orig_count = original_results.get('object_counts', {}).get(class_name, 0)
        imp_count = improved_results.get('object_counts', {}).get(class_name, 0)
        change = imp_count - orig_count
        change_str = f"{change:+d}" if change != 0 else "0"

        print(f"{class_name:<20} {orig_count:<15} {imp_count:<15} {change_str:<15}")

    orig_total = original_results.get('total_detections', 0)
    imp_total = improved_results.get('total_detections', 0)
    total_change = imp_total - orig_total

    print("-"*80)
    print(f"{'TOTAL':<20} {orig_total:<15} {imp_total:<15} {total_change:+d}")

    # Duplicate analysis
    print("\n2. DUPLICATE DETECTION ANALYSIS")
    print("-"*80)

    orig_dup = analyze_duplicates(original_results)
    imp_dup = analyze_duplicates(improved_results)

    print(f"Potential duplicates (Original): {orig_dup['num_duplicates']}")
    print(f"Potential duplicates (Improved): {imp_dup['num_duplicates']}")
    print(f"Duplicates eliminated: {orig_dup['num_duplicates'] - imp_dup['num_duplicates']}")

    if orig_dup['num_duplicates'] > 0:
        print("\nSample duplicate detections from original:")
        for i, dup in enumerate(orig_dup['duplicate_candidates'][:3], 1):
            print(f"  {i}. Frame {dup['frame']}: {dup['object1']} overlaps with "
                  f"{dup['object2']} (IoU: {dup['iou']:.2f})")

    # False positive analysis
    print("\n3. FALSE POSITIVE INDICATORS")
    print("-"*80)

    orig_fp = analyze_false_positives(original_results)
    imp_fp = analyze_false_positives(improved_results)

    print(f"Low confidence detections (<0.4):")
    print(f"  Original: {len(orig_fp['low_confidence'])}")
    print(f"  Improved: {len(imp_fp['low_confidence'])}")

    print(f"Very small detections (<100 px²):")
    print(f"  Original: {len(orig_fp['very_small'])}")
    print(f"  Improved: {len(imp_fp['very_small'])}")

    print(f"Very large detections (>500k px²):")
    print(f"  Original: {len(orig_fp['very_large'])}")
    print(f"  Improved: {len(imp_fp['very_large'])}")

    # Filtering statistics (if available in improved)
    if 'filtered_stats' in improved_results:
        print("\n4. FILTERING STATISTICS (Improved Version)")
        print("-"*80)
        stats = improved_results['filtered_stats']
        print(f"Duplicate detections removed: {stats.get('duplicate_detections', 0)}")
        print(f"Small area filtered: {stats.get('small_area_filtered', 0)}")
        print(f"Large area filtered: {stats.get('large_area_filtered', 0)}")
        print(f"Low confidence filtered: {stats.get('low_confidence_filtered', 0)}")
        total_filtered = sum(stats.values())
        print(f"Total filtered: {total_filtered}")

    # Per-frame analysis
    print("\n5. PER-FRAME DETECTION ANALYSIS")
    print("-"*80)

    orig_frames = len(original_results.get('frames', {}))
    imp_frames = len(improved_results.get('frames', {}))

    print(f"Frames with detections (Original): {orig_frames}")
    print(f"Frames with detections (Improved): {imp_frames}")

    # Calculate average detections per frame
    orig_avg = orig_dup['total_detections'] / orig_frames if orig_frames > 0 else 0
    imp_avg = imp_dup['total_detections'] / imp_frames if imp_frames > 0 else 0

    print(f"Average detections per frame (Original): {orig_avg:.2f}")
    print(f"Average detections per frame (Improved): {imp_avg:.2f}")

    # Summary
    print("\n6. IMPROVEMENT SUMMARY")
    print("="*80)

    improvements = []
    if orig_dup['num_duplicates'] > imp_dup['num_duplicates']:
        improvements.append(f"✓ Reduced duplicates by {orig_dup['num_duplicates'] - imp_dup['num_duplicates']}")

    if len(orig_fp['low_confidence']) > len(imp_fp['low_confidence']):
        improvements.append(f"✓ Filtered {len(orig_fp['low_confidence']) - len(imp_fp['low_confidence'])} low-confidence detections")

    if len(orig_fp['very_small']) > len(imp_fp['very_small']):
        improvements.append(f"✓ Filtered {len(orig_fp['very_small']) - len(imp_fp['very_small'])} very small detections")

    if improvements:
        for improvement in improvements:
            print(improvement)
    else:
        print("No significant differences detected")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and improved SAM3 detector results"
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original detector JSON results"
    )
    parser.add_argument(
        "--improved",
        type=str,
        required=True,
        help="Path to improved detector JSON results"
    )

    args = parser.parse_args()

    # Check files exist
    if not Path(args.original).exists():
        print(f"Error: Original results file not found: {args.original}")
        sys.exit(1)

    if not Path(args.improved).exists():
        print(f"Error: Improved results file not found: {args.improved}")
        sys.exit(1)

    # Load results
    print("Loading results...")
    original_results = load_results(args.original)
    improved_results = load_results(args.improved)

    # Print comparison
    print_comparison(original_results, improved_results)


if __name__ == "__main__":
    main()
