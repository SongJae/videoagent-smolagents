"""
Example usage of SAM3 Video Detector
Demonstrates how to use the detector for battlefield reconnaissance videos
"""

from sam3_video_detector import SAM3VideoDetector
import torch


def example_basic_usage():
    """Basic usage example"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Initialize detector
    detector = SAM3VideoDetector(
        device="cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold=0.5
    )

    # Process a video
    results = detector.process_video(
        video_path="path/to/your/battlefield_video.mp4",
        output_dir="./output",
        max_frames=None,  # Process all frames
        save_json=True,
        generate_video=True
    )

    # Access results
    print("\nDetection Results:")
    print(f"Total frames processed: {results.total_frames}")
    print(f"Video FPS: {results.fps:.2f}")
    print("\nObject counts:")
    for class_name, count in results.object_counts.items():
        print(f"  {class_name}: {count}")


def example_custom_detection():
    """Custom detection with specific parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Detection Parameters")
    print("=" * 80)

    # Initialize with custom settings
    detector = SAM3VideoDetector(
        model_name="facebook/sam3",
        device="cuda",
        dtype=torch.bfloat16,
        confidence_threshold=0.7  # Higher threshold for more confident detections
    )

    # Detect and track
    results = detector.detect_and_track(
        video_path="path/to/video.mp4",
        max_frames=100  # Process only first 100 frames
    )

    # Save results separately
    results.save_json("custom_results.json")

    # Generate video with custom settings
    detector.visualize_results(
        video_path="path/to/video.mp4",
        results=results,
        output_path="custom_annotated.mp4",
        show_ids=True,
        show_confidence=True
    )


def example_detailed_analysis():
    """Detailed frame-by-frame analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Detailed Frame-by-Frame Analysis")
    print("=" * 80)

    detector = SAM3VideoDetector()

    # Process video
    results = detector.detect_and_track(
        video_path="path/to/video.mp4",
        max_frames=50
    )

    # Analyze specific frames
    print("\nFrame-by-frame analysis:")
    for frame_idx in range(min(5, results.total_frames)):
        if frame_idx in results.frame_results:
            frame_result = results.frame_results[frame_idx]
            print(f"\nFrame {frame_idx}:")
            print(f"  Total objects: {len(frame_result.objects)}")

            for obj in frame_result.objects:
                print(f"    - {obj['class']} (ID: {obj['object_id']})")
                print(f"      BBox: {obj['bbox']}")
                print(f"      Confidence: {obj['confidence']:.3f}")


def example_batch_processing():
    """Process multiple videos"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Video Processing")
    print("=" * 80)

    detector = SAM3VideoDetector()

    video_files = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        "path/to/video3.mp4"
    ]

    all_results = []
    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        try:
            results = detector.process_video(
                video_path=video_path,
                output_dir=f"./output/{Path(video_path).stem}",
                max_frames=None
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    # Summary across all videos
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    total_counts = {'soldier': 0, 'armed_vehicle': 0, 'tank': 0}
    for results in all_results:
        for class_name, count in results.object_counts.items():
            total_counts[class_name] += count

    print("Total objects detected across all videos:")
    for class_name, count in total_counts.items():
        print(f"  {class_name}: {count}")


def example_programmatic_access():
    """Access detection data programmatically"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Programmatic Data Access")
    print("=" * 80)

    detector = SAM3VideoDetector()
    results = detector.detect_and_track("path/to/video.mp4", max_frames=30)

    # Convert to dictionary
    results_dict = results.to_dict()

    # Access structured data
    print(f"Video: {results_dict['video_path']}")
    print(f"FPS: {results_dict['fps']}")
    print(f"Total detections: {results_dict['total_detections']}")

    # Extract all soldier detections
    soldier_detections = []
    for frame_idx, frame_data in results_dict['frames'].items():
        for obj in frame_data['objects']:
            if obj['class'] == 'soldier':
                soldier_detections.append({
                    'frame': frame_idx,
                    'id': obj['object_id'],
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence']
                })

    print(f"\nFound {len(soldier_detections)} soldier detections")

    # Track specific object across frames
    if soldier_detections:
        first_soldier_id = soldier_detections[0]['id']
        print(f"\nTracking soldier ID {first_soldier_id}:")

        for detection in soldier_detections:
            if detection['id'] == first_soldier_id:
                print(f"  Frame {detection['frame']}: "
                      f"BBox {detection['bbox']}, "
                      f"Conf {detection['confidence']:.3f}")


def example_quick_count():
    """Quick object counting without video generation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Quick Object Counting")
    print("=" * 80)

    detector = SAM3VideoDetector()

    # Only run detection, skip video generation
    results = detector.process_video(
        video_path="path/to/video.mp4",
        output_dir="./output",
        save_json=True,
        generate_video=False  # Skip video rendering for speed
    )

    # Print quick summary
    print("\nQuick Count Summary:")
    print(f"Soldiers: {results.object_counts.get('soldier', 0)}")
    print(f"Armed Vehicles: {results.object_counts.get('armed_vehicle', 0)}")
    print(f"Tanks: {results.object_counts.get('tank', 0)}")
    print(f"Total: {sum(results.object_counts.values())}")


if __name__ == "__main__":
    from pathlib import Path

    print("SAM3 Video Detector - Example Usage")
    print("=" * 80)
    print("\nThese examples demonstrate various ways to use the detector.")
    print("Replace 'path/to/your/video.mp4' with actual video paths.\n")

    # Uncomment the example you want to run:

    # example_basic_usage()
    # example_custom_detection()
    # example_detailed_analysis()
    # example_batch_processing()
    # example_programmatic_access()
    # example_quick_count()

    print("\nTo use these examples:")
    print("1. Uncomment the example function you want to run")
    print("2. Replace video paths with your actual video files")
    print("3. Run: python example_usage.py")
