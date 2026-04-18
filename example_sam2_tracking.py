#!/usr/bin/env python3
"""
Example script demonstrating SAM2 tracking integration
Shows how to use ObjectDetectionProcessor with tracking capabilities
"""

import sys
from pathlib import Path
import yaml

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.object_detection import ObjectDetectionProcessor


def example_basic_tracking(video_path: str, output_path: str = None):
    """
    Example 1: Basic tracking on a video segment
    """
    print("\n" + "="*60)
    print("Example 1: Basic SAM2 Tracking")
    print("="*60 + "\n")

    # Load configuration
    config_path = Path(__file__).parent / "config" / "models_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize processor with SAM2 tracking
    print("Initializing ObjectDetectionProcessor with SAM2 tracking...")
    processor = ObjectDetectionProcessor(
        model_name=config["object_detection"]["model_name"],
        conf_threshold=config["object_detection"]["conf_threshold"],
        nms_threshold=config["object_detection"]["nms_threshold"],
        target_classes=config["object_detection"]["target_classes"],
        sam2_config=config.get("sam2_tracking")  # Optional: will work without it
    )

    # Process first 30 seconds with tracking
    print(f"\nProcessing video: {video_path}")
    print("Tracking objects through first 30 seconds...\n")

    results = processor.process_segment_with_tracking(
        video_path=video_path,
        start_time=0.0,
        end_time=30.0,
        output_video_path=output_path
    )

    # Print results
    print("\n" + "="*60)
    print("Tracking Results")
    print("="*60)
    print(f"Segment: {results['segment_start']:.1f}s - {results['segment_end']:.1f}s")
    print(f"Objects Tracked: {results.get('num_objects_tracked', 0)}")
    print(f"Frames Processed: {results.get('num_frames_processed', 0)}")

    if 'initial_detections' in results and results['initial_detections']:
        print(f"\nDetected Classes: {results['initial_detections']}")

    if output_path:
        print(f"\n✅ Annotated video saved to: {output_path}")

    # Cleanup
    processor.cleanup()
    print("\n" + "="*60 + "\n")

    return results


def example_detection_only(video_path: str):
    """
    Example 2: Fallback to detection-only mode (no SAM2)
    """
    print("\n" + "="*60)
    print("Example 2: Detection-Only Mode (No SAM2)")
    print("="*60 + "\n")

    # Initialize without SAM2 config
    print("Initializing ObjectDetectionProcessor without SAM2...")
    processor = ObjectDetectionProcessor(
        model_name="iSEE-Laboratory/llmdet_large",
        target_classes=["truck", "tank", "soldier"]
    )

    # Process segment - will automatically use detection-only mode
    print(f"\nProcessing video: {video_path}")
    print("Using detection-only mode...\n")

    results = processor.process_segment_simple(
        video_path=video_path,
        start_time=0.0,
        end_time=30.0
    )

    detections, frame = results

    # Print results
    print("\n" + "="*60)
    print("Detection Results")
    print("="*60)
    print(f"Detections in middle frame: {len(detections)}")

    if detections:
        print("\nDetected Objects:")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2f}")

    # Cleanup
    processor.cleanup()
    print("\n" + "="*60 + "\n")

    return detections, frame


def example_compare_tracking_methods(video_path: str):
    """
    Example 3: Compare tracking vs detection-only
    """
    print("\n" + "="*60)
    print("Example 3: Compare Tracking vs Detection-Only")
    print("="*60 + "\n")

    # Load configuration
    config_path = Path(__file__).parent / "config" / "models_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test with tracking
    print("Testing WITH SAM2 tracking...")
    processor_with_tracking = ObjectDetectionProcessor(
        model_name=config["object_detection"]["model_name"],
        target_classes=config["object_detection"]["target_classes"],
        sam2_config=config.get("sam2_tracking")
    )

    tracking_results = processor_with_tracking.process_segment_with_tracking(
        video_path=video_path,
        start_time=0.0,
        end_time=10.0  # Shorter for comparison
    )

    processor_with_tracking.cleanup()

    # Test without tracking
    print("\nTesting WITHOUT SAM2 tracking...")
    processor_no_tracking = ObjectDetectionProcessor(
        model_name=config["object_detection"]["model_name"],
        target_classes=config["object_detection"]["target_classes"]
    )

    detection_results = processor_no_tracking.process_segment(
        video_path=video_path,
        start_time=0.0,
        end_time=10.0,
        sample_rate=1
    )

    processor_no_tracking.cleanup()

    # Compare results
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    print(f"\nWith Tracking:")
    print(f"  Objects Tracked: {tracking_results.get('num_objects_tracked', 0)}")
    print(f"  Frames Processed: {tracking_results.get('num_frames_processed', 0)}")
    print(f"  Has Tracking Data: {'tracking_data' in tracking_results}")

    print(f"\nWithout Tracking:")
    print(f"  Frames Processed: {detection_results.get('num_frames_processed', 0)}")
    print(f"  Total Detections: {detection_results.get('statistics', {}).get('total_detections', 0)}")

    print("\n" + "="*60 + "\n")

    return tracking_results, detection_results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM2 Tracking Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking with output video
  python example_sam2_tracking.py basic --video video.mp4 --output tracked.mp4

  # Detection-only mode
  python example_sam2_tracking.py detection-only --video video.mp4

  # Compare tracking methods
  python example_sam2_tracking.py compare --video video.mp4
        """
    )

    subparsers = parser.add_subparsers(dest="example", help="Example to run")

    # Basic tracking
    basic_parser = subparsers.add_parser("basic", help="Basic tracking example")
    basic_parser.add_argument("--video", required=True, help="Path to video file")
    basic_parser.add_argument("--output", help="Path to save annotated video")

    # Detection only
    detection_parser = subparsers.add_parser("detection-only", help="Detection-only example")
    detection_parser.add_argument("--video", required=True, help="Path to video file")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare tracking methods")
    compare_parser.add_argument("--video", required=True, help="Path to video file")

    args = parser.parse_args()

    # Run example
    if args.example == "basic":
        example_basic_tracking(args.video, args.output)

    elif args.example == "detection-only":
        example_detection_only(args.video)

    elif args.example == "compare":
        example_compare_tracking_methods(args.video)

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
