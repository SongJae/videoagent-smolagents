"""
Test script for SAM3 Video Detector
Verifies installation and basic functionality
"""

import sys
import torch


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import cv2
        print("  ✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False

    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        print("  ✓ Transformers SAM3 modules imported successfully")
    except ImportError as e:
        print(f"  ✗ Transformers SAM3 import failed: {e}")
        print("    Please install: pip install transformers>=4.47.0")
        return False

    try:
        from sam3_video_detector import SAM3VideoDetector, DetectionResult, VideoAnalysisResults
        print("  ✓ SAM3VideoDetector modules imported successfully")
    except ImportError as e:
        print(f"  ✗ SAM3VideoDetector import failed: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    - GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        return True
    else:
        print("  ⚠ CUDA not available, will use CPU (slower)")
        return False


def test_model_loading():
    """Test if SAM3 model can be loaded"""
    print("\nTesting model loading...")
    try:
        from sam3_video_detector import SAM3VideoDetector

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading model on {device}...")

        detector = SAM3VideoDetector(
            device=device,
            confidence_threshold=0.5
        )

        print("  ✓ SAM3 model loaded successfully")
        print(f"  ✓ Model device: {detector.device}")
        print(f"  ✓ Model dtype: {detector.dtype}")

        return True

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        print("\nThis may take a few minutes on first run as the model downloads.")
        return False


def test_data_structures():
    """Test data structure classes"""
    print("\nTesting data structures...")
    try:
        from sam3_video_detector import DetectionResult, VideoAnalysisResults

        # Test DetectionResult
        result = DetectionResult(frame_idx=0)
        result.add_object(
            obj_id=1,
            bbox=[100, 200, 150, 250],
            class_name="soldier",
            confidence=0.95
        )
        assert len(result.objects) == 1
        print("  ✓ DetectionResult working correctly")

        # Test VideoAnalysisResults
        video_results = VideoAnalysisResults(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0
        )
        video_results.add_frame_result(result)
        video_results.update_object_counts()
        assert video_results.object_counts['soldier'] == 1
        print("  ✓ VideoAnalysisResults working correctly")

        # Test to_dict
        data = video_results.to_dict()
        assert 'video_path' in data
        assert 'object_counts' in data
        print("  ✓ Data serialization working correctly")

        return True

    except Exception as e:
        print(f"  ✗ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_object_classes():
    """Test object class definitions"""
    print("\nTesting object class definitions...")
    try:
        from sam3_video_detector import SAM3VideoDetector

        detector = SAM3VideoDetector()

        # Check object classes
        assert 'soldier' in detector.OBJECT_CLASSES
        assert 'armed_vehicle' in detector.OBJECT_CLASSES
        assert 'tank' in detector.OBJECT_CLASSES
        print("  ✓ Object classes defined correctly")

        # Check class colors
        assert 'soldier' in detector.CLASS_COLORS
        assert 'armed_vehicle' in detector.CLASS_COLORS
        assert 'tank' in detector.CLASS_COLORS
        print("  ✓ Class colors defined correctly")

        # Check prompts
        for class_name, prompts in detector.OBJECT_CLASSES.items():
            assert len(prompts) > 0, f"No prompts for {class_name}"
        print("  ✓ Text prompts defined correctly")

        return True

    except Exception as e:
        print(f"  ✗ Object class test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("SAM3 Video Detector - System Test")
    print("=" * 80)

    results = {
        'imports': test_imports(),
        'cuda': test_cuda(),
        'data_structures': test_data_structures(),
        'object_classes': test_object_classes(),
    }

    # Only test model loading if imports succeeded
    if results['imports']:
        results['model_loading'] = test_model_loading()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare a video file with soldiers, armed vehicles, or tanks")
        print("2. Run: python sam3_video_detector.py --video path/to/your/video.mp4")
        print("3. Check the ./output directory for results")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements_sam3_detector.txt")
        print("- For CUDA issues: Reinstall PyTorch with CUDA support")
        print("- For transformers issues: pip install --upgrade transformers>=4.47.0")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
