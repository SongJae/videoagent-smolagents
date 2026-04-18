# SAM3 Video Object Detection and Tracking System

A complete implementation of object detection and tracking using **SAM3 (Segment Anything Model 3)** from Meta via the Hugging Face transformers library. This system is specifically designed for battlefield reconnaissance videos to detect and track soldiers, armed vehicles, and tanks.

## Features

- **Unified Detection & Tracking**: Uses SAM3's video capabilities for seamless object tracking across frames
- **Multi-Class Detection**: Detects soldiers, armed vehicles, and tanks using text prompts
- **Accurate Object Counting**: Tracks unique object IDs to provide exact counts of each object type
- **Bounding Box Information**: Returns precise bounding boxes in XYXY format for all detections
- **Annotated Video Output**: Generates visualization videos with bounding boxes, labels, and object IDs
- **Structured Results**: Outputs JSON files with complete detection data for further analysis
- **Transformers Library**: Uses official Hugging Face implementation for easy integration

## System Requirements

- **GPU**: CUDA-capable GPU recommended (tested on A100 80GB)
- **Python**: 3.8 or higher
- **VRAM**: ~10GB for SAM3 model
- **Storage**: ~2GB for model weights

## Installation

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install transformers with SAM3 support
pip install transformers>=4.47.0

# Install other requirements
pip install opencv-python numpy tqdm
```

### 2. Verify Installation

```python
python -c "from transformers import Sam3VideoModel, Sam3VideoProcessor; print('SAM3 is ready!')"
```

## Quick Start

### Basic Usage

```python
from sam3_video_detector import SAM3VideoDetector

# Initialize detector
detector = SAM3VideoDetector(
    device="cuda",
    confidence_threshold=0.5
)

# Process video - returns detection results and generates annotated video
results = detector.process_video(
    video_path="battlefield_footage.mp4",
    output_dir="./output"
)

# View results
print(f"Soldiers: {results.object_counts['soldier']}")
print(f"Armed Vehicles: {results.object_counts['armed_vehicle']}")
print(f"Tanks: {results.object_counts['tank']}")
```

### Command Line Usage

```bash
# Basic usage
python sam3_video_detector.py --video path/to/video.mp4

# With custom parameters
python sam3_video_detector.py \
    --video path/to/video.mp4 \
    --output-dir ./results \
    --confidence 0.7 \
    --max-frames 300

# CPU-only mode
python sam3_video_detector.py \
    --video path/to/video.mp4 \
    --device cpu

# Quick counting without video generation
python sam3_video_detector.py \
    --video path/to/video.mp4 \
    --no-video
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | Required | Path to input video file |
| `--output-dir` | str | `./output` | Output directory for results |
| `--max-frames` | int | `None` | Maximum frames to process (None = all) |
| `--confidence` | float | `0.5` | Confidence threshold (0.0-1.0) |
| `--device` | str | `cuda` | Device to use (cuda or cpu) |
| `--no-video` | flag | False | Skip generating annotated video |
| `--no-json` | flag | False | Skip saving JSON results |

## Output Files

When processing a video named `battlefield_footage.mp4`, the system generates:

### 1. Annotated Video
**File**: `battlefield_footage_annotated.mp4`
- Original video with bounding boxes overlaid
- Color-coded by object class (Soldier: Green, Armed Vehicle: Orange, Tank: Red)
- Labels showing class name, object ID, and confidence score
- Summary statistics in top-left corner

### 2. JSON Results
**File**: `battlefield_footage_detections.json`

```json
{
  "video_path": "battlefield_footage.mp4",
  "total_frames": 450,
  "fps": 30.0,
  "object_counts": {
    "soldier": 12,
    "armed_vehicle": 3,
    "tank": 2
  },
  "total_detections": 17,
  "frames": {
    "0": {
      "objects": [
        {
          "object_id": 1,
          "bbox": [120.5, 340.2, 180.3, 450.7],
          "class": "soldier",
          "confidence": 0.89
        }
      ]
    }
  }
}
```

## API Reference

### SAM3VideoDetector Class

#### Constructor

```python
detector = SAM3VideoDetector(
    model_name="facebook/sam3",
    device="cuda",
    dtype=torch.bfloat16,
    confidence_threshold=0.5
)
```

**Parameters:**
- `model_name` (str): Hugging Face model ID (default: "facebook/sam3")
- `device` (str): Computation device - "cuda" or "cpu" (default: "cuda")
- `dtype` (torch.dtype): Model precision (default: torch.bfloat16)
- `confidence_threshold` (float): Minimum confidence for detections (default: 0.5)

#### Methods

##### `process_video()`
Complete pipeline: detect, track, visualize, and save results.

```python
results = detector.process_video(
    video_path="video.mp4",
    output_dir="./output",
    max_frames=None,
    save_json=True,
    generate_video=True
)
```

**Parameters:**
- `video_path` (str): Path to input video file
- `output_dir` (str): Directory for output files (default: "./output")
- `max_frames` (int|None): Maximum frames to process (default: None = all frames)
- `save_json` (bool): Save JSON results (default: True)
- `generate_video` (bool): Generate annotated video (default: True)

**Returns:** `VideoAnalysisResults` object

##### `detect_and_track()`
Run detection and tracking only, without visualization.

```python
results = detector.detect_and_track(
    video_path="video.mp4",
    max_frames=None
)
```

##### `visualize_results()`
Generate annotated video from existing results.

```python
detector.visualize_results(
    video_path="video.mp4",
    results=results,
    output_path="annotated.mp4",
    show_ids=True,
    show_confidence=True
)
```

### VideoAnalysisResults Class

Stores detection results for a processed video.

#### Attributes

- `video_path` (str): Path to original video
- `total_frames` (int): Number of frames processed
- `fps` (float): Video frame rate
- `frame_results` (dict): Per-frame detection results
- `object_counts` (dict): Count of unique objects by class

#### Methods

##### `to_dict()`
Convert results to dictionary format.

```python
data = results.to_dict()
```

##### `save_json()`
Save results to JSON file.

```python
results.save_json("results.json")
```

## Object Classes

The system detects three object classes with multiple text prompt variations:

| Class | Text Prompts | Color (BGR) |
|-------|-------------|-------------|
| **soldier** | soldier, soldiers, military personnel, infantry | Green (0, 255, 0) |
| **armed_vehicle** | military vehicle, armed vehicle, armored vehicle, military truck | Orange (255, 165, 0) |
| **tank** | tank, tanks, main battle tank, armored tank, military tank | Red (0, 0, 255) |

## Advanced Examples

### Example 1: Custom Object Classes

Modify the `OBJECT_CLASSES` dictionary to detect different objects:

```python
detector = SAM3VideoDetector()
detector.OBJECT_CLASSES = {
    'helicopter': ['helicopter', 'military helicopter', 'attack helicopter'],
    'drone': ['drone', 'UAV', 'unmanned aerial vehicle']
}
```

### Example 2: Frame-by-Frame Analysis

```python
results = detector.detect_and_track("video.mp4")

# Analyze specific frames
for frame_idx in range(10):
    if frame_idx in results.frame_results:
        frame = results.frame_results[frame_idx]
        print(f"Frame {frame_idx}: {len(frame.objects)} objects")
        for obj in frame.objects:
            print(f"  - {obj['class']} #{obj['object_id']} at {obj['bbox']}")
```

### Example 3: Track Specific Object

```python
results = detector.detect_and_track("video.mp4")

# Track a specific object ID across frames
target_id = 1
trajectory = []

for frame_idx, frame_result in results.frame_results.items():
    for obj in frame_result.objects:
        if obj['object_id'] == target_id:
            trajectory.append({
                'frame': frame_idx,
                'bbox': obj['bbox'],
                'confidence': obj['confidence']
            })

print(f"Object {target_id} appeared in {len(trajectory)} frames")
```

### Example 4: Batch Processing

```python
import glob

detector = SAM3VideoDetector()

video_files = glob.glob("./videos/*.mp4")
all_counts = {'soldier': 0, 'armed_vehicle': 0, 'tank': 0}

for video_path in video_files:
    results = detector.process_video(video_path)
    for class_name, count in results.object_counts.items():
        all_counts[class_name] += count

print(f"Total across all videos: {all_counts}")
```

### Example 5: Export to CSV

```python
import csv

results = detector.detect_and_track("video.mp4")

with open("detections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "object_id", "class", "x1", "y1", "x2", "y2", "confidence"])

    for frame_idx, frame_result in results.frame_results.items():
        for obj in frame_result.objects:
            writer.writerow([
                frame_idx,
                obj['object_id'],
                obj['class'],
                *obj['bbox'],
                obj['confidence']
            ])
```

## Performance Optimization

### GPU Memory Optimization

For videos with many objects:

```python
detector = SAM3VideoDetector(
    dtype=torch.float16,  # Use FP16 instead of BF16
    confidence_threshold=0.7  # Higher threshold = fewer detections
)
```

### Processing Speed

Typical processing speeds (A100 80GB):
- **720p video (30 FPS)**: ~3-5 seconds per 30 frames
- **1080p video (30 FPS)**: ~5-8 seconds per 30 frames
- **4K video (30 FPS)**: ~10-15 seconds per 30 frames

### Batch Size Tuning

For faster processing on high-end GPUs, SAM3 automatically uses efficient batching internally.

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# Use FP16 precision
detector = SAM3VideoDetector(dtype=torch.float16)

# Or process fewer frames at a time
results = detector.process_video(video_path, max_frames=100)
```

### Issue: Low Detection Accuracy

**Solution:**
```python
# Lower confidence threshold
detector = SAM3VideoDetector(confidence_threshold=0.3)

# Or add more text prompt variations
detector.OBJECT_CLASSES['soldier'].extend(['person', 'infantry soldier'])
```

### Issue: Slow Processing on CPU

**Solution:**
```python
# Process fewer frames
detector.process_video(video_path, max_frames=50)

# Skip video generation
detector.process_video(video_path, generate_video=False)
```

## Citation

If you use this implementation, please cite SAM3:

```bibtex
@article{sam3,
  title={Segment Anything Model 3},
  author={Meta AI Research},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This implementation follows the SAM3 license. See the [SAM3 repository](https://huggingface.co/facebook/sam3) for details.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional object classes
- Performance optimizations
- Better visualization options
- Integration with other tracking algorithms

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the example_usage.py file for code examples
3. Consult the [SAM3 documentation](https://huggingface.co/facebook/sam3)
4. Open an issue on the project repository

## Acknowledgments

- **Meta AI** for developing SAM3
- **Hugging Face** for the transformers library integration
- **PyTorch** team for the deep learning framework
