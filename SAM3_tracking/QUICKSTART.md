# SAM3 Video Detector - Quick Start Guide

Get started with object detection and tracking in under 5 minutes!

## Installation (2 minutes)

### Step 1: Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower)
pip install torch torchvision
```

### Step 2: Install Dependencies

```bash
pip install -r requirements_sam3_detector.txt
```

### Step 3: Verify Installation

```bash
python test_sam3_detector.py
```

Expected output:
```
✓ All tests passed! The system is ready to use.
```

## Your First Detection (2 minutes)

### Method 1: Command Line (Easiest)

```bash
python sam3_video_detector.py --video path/to/your/video.mp4
```

This will:
- ✓ Detect and track soldiers, armed vehicles, and tanks
- ✓ Generate an annotated video with bounding boxes
- ✓ Save detection data to JSON
- ✓ Output results to `./output/` directory

### Method 2: Python Script

Create a file `my_detection.py`:

```python
from sam3_video_detector import SAM3VideoDetector

# Initialize
detector = SAM3VideoDetector()

# Process video
results = detector.process_video("your_video.mp4")

# View counts
print(f"Soldiers: {results.object_counts.get('soldier', 0)}")
print(f"Armed Vehicles: {results.object_counts.get('armed_vehicle', 0)}")
print(f"Tanks: {results.object_counts.get('tank', 0)}")
```

Run it:
```bash
python my_detection.py
```

## Understanding the Output

After processing, you'll find in `./output/`:

### 1. Annotated Video: `your_video_annotated.mp4`
- Shows bounding boxes around detected objects
- Color-coded: Green (soldiers), Orange (armed vehicles), Red (tanks)
- Displays object IDs and confidence scores
- Shows real-time object counts

### 2. Detection Data: `your_video_detections.json`
```json
{
  "object_counts": {
    "soldier": 12,
    "armed_vehicle": 3,
    "tank": 2
  },
  "frames": {
    "0": {
      "objects": [...]
    }
  }
}
```

## Common Use Cases

### Use Case 1: Quick Object Count

Just want to count objects without generating video?

```bash
python sam3_video_detector.py --video video.mp4 --no-video
```

### Use Case 2: Process First 100 Frames Only

```bash
python sam3_video_detector.py --video video.mp4 --max-frames 100
```

### Use Case 3: Higher Confidence Threshold

Reduce false positives by requiring higher confidence:

```bash
python sam3_video_detector.py --video video.mp4 --confidence 0.7
```

### Use Case 4: Batch Process Multiple Videos

```python
from sam3_video_detector import SAM3VideoDetector
import glob

detector = SAM3VideoDetector()

for video in glob.glob("videos/*.mp4"):
    print(f"Processing {video}...")
    results = detector.process_video(video)
    print(f"  Found {sum(results.object_counts.values())} objects")
```

## Tips for Best Results

### For Accurate Counts
- Process the entire video (`--max-frames` not set)
- Use default confidence threshold (0.5)
- Ensure video quality is good (720p or higher recommended)

### For Faster Processing
- Use GPU (much faster than CPU)
- Process fewer frames (`--max-frames 100`)
- Skip video generation (`--no-video`)
- Use lower confidence threshold (0.3)

### For Fewer False Positives
- Increase confidence threshold (`--confidence 0.7`)
- Ensure good lighting in video
- Use higher resolution video

## Troubleshooting

### Error: "CUDA out of memory"

**Solution 1**: Use FP16 precision
```python
detector = SAM3VideoDetector(dtype=torch.float16)
```

**Solution 2**: Process fewer frames
```bash
python sam3_video_detector.py --video video.mp4 --max-frames 50
```

### Error: "Cannot import Sam3VideoModel"

**Solution**: Update transformers
```bash
pip install --upgrade transformers>=4.47.0
```

### Issue: Very slow on CPU

This is expected. SAM3 requires GPU for reasonable speed.

**Solution 1**: Use a GPU

**Solution 2**: Process fewer frames
```bash
python sam3_video_detector.py --video video.mp4 --max-frames 30 --no-video
```

### Issue: Low detection accuracy

**Solution 1**: Lower confidence threshold
```bash
python sam3_video_detector.py --video video.mp4 --confidence 0.3
```

**Solution 2**: Ensure video quality
- Check resolution (720p+ recommended)
- Ensure good lighting
- Reduce motion blur

## Next Steps

### Learn More
- Read `README_SAM3_DETECTOR.md` for complete documentation
- Check `example_usage.py` for advanced usage patterns
- See API reference in README for programmatic access

### Customize
- Modify `OBJECT_CLASSES` in `sam3_video_detector.py` to detect different objects
- Adjust `CLASS_COLORS` to change visualization colors
- Add more text prompts for better detection

### Integrate
- Import results into your analysis pipeline
- Export to CSV for spreadsheet analysis
- Build dashboards using the JSON output

## Command Reference

```bash
# Basic usage
python sam3_video_detector.py --video VIDEO_PATH

# Common options
--output-dir DIR          # Output directory (default: ./output)
--max-frames N           # Process only N frames
--confidence FLOAT       # Confidence threshold (0.0-1.0)
--device cuda/cpu        # Device to use
--no-video              # Skip video generation
--no-json               # Skip JSON output

# Examples
python sam3_video_detector.py --video drone_footage.mp4
python sam3_video_detector.py --video video.mp4 --confidence 0.7
python sam3_video_detector.py --video video.mp4 --max-frames 100 --no-video
```

## Performance Expectations

| Video Resolution | GPU | Processing Speed |
|-----------------|-----|-----------------|
| 720p (30 FPS) | A100 80GB | ~3-5s per 30 frames |
| 1080p (30 FPS) | A100 80GB | ~5-8s per 30 frames |
| 720p (30 FPS) | RTX 3090 | ~8-12s per 30 frames |
| 720p (30 FPS) | CPU | ~60-120s per 30 frames |

## Support

Need help?
1. Check this guide
2. Read `README_SAM3_DETECTOR.md`
3. Review `test_sam3_detector.py` output
4. Check SAM3 documentation: https://huggingface.co/facebook/sam3

---

**Ready to detect! 🎯** Run your first detection now:

```bash
python sam3_video_detector.py --video your_video.mp4
```
