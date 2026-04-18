"""
Test script to verify supervision library integration is working correctly
"""

import sys
import numpy as np


def test_supervision_import():
    """Test if supervision library can be imported"""
    print("="*80)
    print("TEST 1: Supervision Import")
    print("="*80)

    try:
        import supervision as sv
        print("✓ supervision library imported successfully")

        try:
            version = sv.__version__
            print(f"✓ Supervision version: {version}")
        except AttributeError:
            print("⚠ Could not determine supervision version")

        return True
    except ImportError as e:
        print(f"✗ Failed to import supervision: {e}")
        print("\nInstall with: pip install supervision>=0.26.0")
        return False


def test_annotators():
    """Test if BoxAnnotator and LabelAnnotator can be instantiated correctly"""
    print("\n" + "="*80)
    print("TEST 2: Annotator Initialization")
    print("="*80)

    try:
        import supervision as sv

        # Test BoxAnnotator (should only accept thickness, color, color_lookup)
        print("\nTesting BoxAnnotator...")
        try:
            box_annotator = sv.BoxAnnotator(thickness=2)
            print("✓ BoxAnnotator initialized successfully with thickness=2")
        except TypeError as e:
            print(f"✗ BoxAnnotator initialization failed: {e}")
            return False

        # Test that invalid parameters are rejected
        print("\nTesting BoxAnnotator with invalid parameters (should fail)...")
        try:
            invalid_box_annotator = sv.BoxAnnotator(
                thickness=2,
                text_thickness=2  # This should fail
            )
            print("⚠ WARNING: BoxAnnotator accepted text_thickness (unexpected)")
        except TypeError:
            print("✓ BoxAnnotator correctly rejects text_thickness parameter")

        # Test LabelAnnotator (should accept text parameters)
        print("\nTesting LabelAnnotator...")
        try:
            label_annotator = sv.LabelAnnotator(
                text_thickness=2,
                text_scale=0.5,
                text_padding=10
            )
            print("✓ LabelAnnotator initialized successfully")
        except TypeError as e:
            print(f"✗ LabelAnnotator initialization failed: {e}")
            return False

        return True

    except ImportError:
        print("✗ supervision library not available")
        return False


def test_annotation():
    """Test actual annotation with dummy data"""
    print("\n" + "="*80)
    print("TEST 3: Annotation Functionality")
    print("="*80)

    try:
        import supervision as sv
        import cv2

        print("\nCreating dummy image and detections...")

        # Create a dummy image (640x480, black background)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create dummy detections
        xyxy = np.array([
            [100, 100, 200, 200],  # Box 1
            [300, 150, 450, 300],  # Box 2
        ])
        class_ids = np.array([0, 1])
        confidence = np.array([0.9, 0.85])
        tracker_ids = np.array([1, 2])

        detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=confidence,
            tracker_id=tracker_ids
        )

        print("✓ Created dummy detections")

        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.5,
            text_padding=10
        )

        print("✓ Created annotators")

        # Annotate
        labels = ["Object 1", "Object 2"]

        try:
            annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
            print("✓ BoxAnnotator.annotate() successful")
        except Exception as e:
            print(f"✗ BoxAnnotator.annotate() failed: {e}")
            return False

        try:
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels
            )
            print("✓ LabelAnnotator.annotate() successful")
        except Exception as e:
            print(f"✗ LabelAnnotator.annotate() failed: {e}")
            return False

        # Verify the image was modified
        if not np.array_equal(image, annotated):
            print("✓ Image was successfully annotated (image modified)")
        else:
            print("⚠ Warning: Image appears unchanged after annotation")

        return True

    except ImportError:
        print("✗ supervision library not available")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detections_class():
    """Test Detections class creation"""
    print("\n" + "="*80)
    print("TEST 4: Detections Class")
    print("="*80)

    try:
        import supervision as sv

        print("\nCreating Detections object...")

        xyxy = np.array([[10, 10, 50, 50]])
        detections = sv.Detections(
            xyxy=xyxy,
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            tracker_id=np.array([1])
        )

        print("✓ Detections object created successfully")
        print(f"  - Number of detections: {len(detections)}")
        print(f"  - Bounding boxes shape: {detections.xyxy.shape}")

        return True

    except ImportError:
        print("✗ supervision library not available")
        return False
    except Exception as e:
        print(f"✗ Failed to create Detections: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SUPERVISION LIBRARY - COMPATIBILITY TEST SUITE")
    print("="*80)

    results = {
        'import': test_supervision_import(),
        'annotators': False,
        'annotation': False,
        'detections': False
    }

    # Only run subsequent tests if import succeeded
    if results['import']:
        results['annotators'] = test_annotators()
        results['annotation'] = test_annotation()
        results['detections'] = test_detections_class()

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✓ All tests passed! Supervision integration is working correctly.")
        print("\nYou can now use sam3_video_detector_improved.py with supervision.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install/update supervision: pip install --upgrade supervision>=0.26.0")
        print("2. Check Python version: Python 3.8+ required")
        print("3. Reinstall if needed: pip uninstall supervision && pip install supervision>=0.26.0")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
