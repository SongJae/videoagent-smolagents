# Battlefield Reconnaissance Video Agent System

A comprehensive AI-powered system for analyzing battlefield reconnaissance drone footage using state-of-the-art object detection, embedding models, and agentic workflows.

## 🎯 Overview

This system combines multiple AI models to automatically analyze drone footage, detect military objects (trucks, tanks, soldiers), generate semantic embeddings, describe events, and enable natural language querying through an intelligent agent interface.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web UI                            │
│  Video Upload | Agent Chat | Process Visualization          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│        CodeAgent (EXAONE-4.0-32B-AWQ + smolagents)          │
│  Tools: PDF RAG | VideoDB Query | Python Execution          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Video Analysis Pipeline                          │
│  1. Upload & Segmentation (VideoDB)                        │
│  2. Object Detection & Tracking (SAM3)                     │
│  3. Embedding Generation (PE-Core-L14-336)                 │
│  4. Event Description (SmolVLM2 2.2B)                      │
│  5. Storage & Indexing                                     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Key Components

### Models
- **Object Detection & Tracking**: SAM3 (unified detection and tracking via text prompts)
- **Embeddings**: PE-Core-L14-336 (1024-dim vectors)
- **Event Description**: SmolVLM2 2.2B Instruct
- **Agent**: EXAONE-4.0-32B-AWQ with smolagents CodeAgent

### Core Systems
- **VideoDBManager**: Video storage, segmentation, and metadata management
- **ObjectDetectionProcessor**: Object detection and tracking wrapper
- **EmbeddingGenerator**: Visual and text embedding generation
- **EventDescriptionGenerator**: VLM-based event description
- **VideoAnalysisSystem**: Complete pipeline orchestration

### Agent Tools
- **PDF RAG Tool**: Query reference documents using retrieval-augmented generation
- **VideoDB Query Tools**: Semantic search, object-based queries, event queries

## 🚀 Installation

### 1. Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- NVIDIA GPU with 80GB VRAM (A100 recommended)
- **No internet connection required** - Fully offline operation (after initial model downloads)

### 2. Install Python Dependencies

```bash
# Navigate to project directory
cd videoagent_smolagents_v0.2_multi_context

# Step 1: Install core dependencies from requirements.txt
pip install -r requirements.txt

# Step 2: Install local smolagents package (modified version)
pip install -e smolagents/

# Step 3: Install perception_models for embeddings
cd perception_models && pip install -e . && cd ..

# Step 4: Install SAM3 for object detection and tracking
cd SAM3_tracking/sam3 && pip install -e . && cd ../..

# Step 5 (Optional): Install map_mcp for tactical map features
cd map_mcp && pip install -e . && cd ..
```

### Key Dependencies Installed

| Category | Packages |
|----------|----------|
| **Deep Learning** | torch>=2.0.0, torchvision>=0.15.0, transformers>=4.40.0 |
| **Object Detection** | SAM3 (local install), supervision>=0.20.0 |
| **Embeddings** | perception_models (local install), huggingface-hub>=0.20.0 |
| **Agent** | smolagents (local install), vllm>=0.3.0 |
| **RAG** | PyPDF2>=3.0.0, faiss-cpu>=1.7.4 |
| **UI** | gradio>=4.0.0, folium>=0.14.0 |
| **Map** | map_mcp (local install, optional) |

### 3. Verify Installation

```bash
# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify key imports
python -c "from core_src.video_analysis_system import VideoAnalysisSystem; print('Core imports OK')"

# Run full environment check
python main.py check
```

**Note**: No API keys required! All data stored locally in `./data/`

### 4. Download Model Weights

The system will automatically download model weights on first use:
- **SAM3**: Unified detection + tracking model
- **PE-Core-L14-336**: Embedding model from HuggingFace Hub
- **SmolVLM2**: VLM for event descriptions from HuggingFace Hub
- **EXAONE-4.0-32B-AWQ**: Agent LLM from HuggingFace Hub

Optional: Pre-download SAM3 checkpoint for offline use:
```bash
# Create checkpoints directory
mkdir -p SAM3_tracking/checkpoints

# Download SAM3 checkpoint (if required for offline)
# Update config/models_config.yaml with checkpoint path:
# object_detection:
#   sam3_config:
#     checkpoint_path: "./SAM3_tracking/checkpoints/sam3_checkpoint.pt"
```

### 5. Download Map Tiles (Optional)

For offline tactical map functionality:
```bash
# Download tiles for a specific area (requires internet)
cd map_mcp
map-mcp-download --lat 37.5665 --lon 126.9780 --radius 100 --output ./tiles --fast
cd ..
```

## 💻 Usage

### Option 1: Web UI (Recommended)

```bash
# Launch Gradio interface
python main.py ui

# Then open browser at http://localhost:7860
```

**Web UI Features:**
1. **Upload & Analyze Video**: Upload drone footage for automatic analysis
2. **Add PDFs**: Add reference documents for agent context
3. **Query Agent**: Natural language interaction with analyzed footage

### Option 2: Command Line

#### Analyze Video
```bash
python main.py analyze \
  --video path/to/drone_footage.mp4 \
  --segment-duration 30 \
  --output results/
```

#### Query Agent
```bash
python main.py query \
  --query "How many tanks were detected?" \
  --video-id m-xxxxx
```

#### Add PDF
```bash
python main.py add-pdf \
  --pdf path/to/manual.pdf
```

#### Check Environment
```bash
python main.py check
```

### Option 3: Python API

```python
from core.video_analysis_system import VideoAnalysisSystem
from agent.battlefield_agent import create_battlefield_agent

# Analyze video
system = VideoAnalysisSystem()
results = system.analyze_video("video.mp4", segment_duration=30)
video_id = results["video_id"]

# Query with agent
agent = create_battlefield_agent()
agent.set_video_context(video_id)
response = agent.run("Describe the tank movements")
print(response)
```

## 💾 Local Storage (Closed Network Mode)

This system operates **completely offline** without any external API dependencies:

- **Storage**: File-based JSON storage via `local_videodb` module + FAISS vector index
- **Videos**: Copied to local `./local_videodb/collections/battlefield_reconnaissance/videos/` directory
- **Metadata**: All segments, objects, descriptions stored in JSON files
- **Embeddings**: 1024-dim vectors in FAISS for fast similarity search
- **Architecture**: Uses existing `local_videodb` module with FAISS integration
- **Portable**: Copy the `local_videodb/` folder to transfer everything

See [LOCAL_VIDEODB_INTEGRATION.md](LOCAL_VIDEODB_INTEGRATION.md) for detailed documentation.

## 📊 Video Analysis Pipeline

### Step 1: Upload & Segmentation
- Video copied to local storage
- Assigned unique ID (e.g., `m-abc123def45`)
- Split into time-based segments (default: 30s)
- Metadata stored in JSON files via local_videodb

### Step 2: Object Detection & Tracking
- Frames extracted from each segment
- SAM3 detects and tracks objects via text prompts: soldiers, tanks, trucks
- Unified detection + tracking in a single model
- Bounding boxes, confidence scores, and tracking IDs recorded

### Step 3: Embedding Generation
- **Segment embeddings**: Full frame → 1024-dim vector
- **Object embeddings**: Cropped objects → 1024-dim vectors
- Uses PE-Core-L14-336 model

### Step 4: Event Description
- SmolVLM2 generates natural language description
- Includes detected objects as context
- Description embedded for semantic search

### Step 5: Storage
All metadata stored in local_videodb + FAISS:
- Segment info (timestamps, IDs) → JSON files
- Object detections (class, count, bounding boxes) → JSON files
- Embeddings (segment, objects, descriptions) → FAISS index
- Event descriptions → JSON files

## 🔍 Querying System

### Semantic Search
```python
# Via tool
query_video_semantic(
    query="tanks moving through forest",
    top_k=5
)
```

### Object-Based Search
```python
# Via tool
query_video_by_object(
    object_type="tank",
    min_count=2
)
```

### Event Search
```python
# Via tool
query_video_by_event(
    event_query="movement"
)
```

### Get Summary
```python
# Via tool
get_video_summary()
# Returns: total counts, segment info, URLs
```

## 🛠️ Configuration

Edit configuration files in `config/`:

### models_config.yaml
- Object detection settings
- Embedding model configuration
- VLM parameters
- Agent model settings

### videodb_config.yaml
- VideoDB connection
- Segment duration settings
- Storage options

### agent_config.yaml
- CodeAgent parameters
- RAG settings
- UI configuration

## 📁 Project Structure

```
videoagent_smolagents_v0.10/
├── config/                    # Configuration files
│   ├── models_config.yaml
│   ├── videodb_config.yaml
│   └── agent_config.yaml
├── core/                      # Core video analysis
│   ├── videodb_manager.py
│   ├── object_detection.py
│   ├── embedding_generator.py
│   ├── event_description.py
│   └── video_analysis_system.py
├── local_videodb/             # Local video database module
│   ├── __init__.py
│   ├── connection.py
│   ├── storage.py
│   ├── collection.py
│   ├── video.py
│   ├── video_processor.py
│   ├── scene.py
│   └── constants.py
├── tools/                     # Smolagents tools
│   ├── pdf_rag_tool.py
│   └── videodb_query_tool.py
├── agent/                     # Agent system
│   ├── model_loader.py
│   └── battlefield_agent.py
├── ui/                        # Gradio interface
│   └── gradio_app.py
├── utils/                     # Utilities
│   ├── video_utils.py
│   └── embedding_utils.py
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## 🎓 Example Queries

Once a video is analyzed, you can ask:

- "How many tanks appear in the video?"
- "Describe the soldier movements in the first 2 minutes"
- "Find all segments where trucks are detected"
- "What events happened around timestamp 01:30?"
- "Show me segments with the most military activity"
- "Count the total number of soldiers detected"
- "Describe the terrain in segment 5"

## 🔧 Advanced Usage

### SAM3 Object Detection & Tracking

SAM3 provides unified detection and tracking using text prompts:

```python
# Use SAM3 in Python
from core_src.object_detection import ObjectDetectionProcessor

# SAM3 is configured via models_config.yaml
processor = ObjectDetectionProcessor(config_path="./config")

# Detect and track objects in a segment
results = processor.process_segment_with_tracking(
    video_path="video.mp4",
    start_time=0.0,
    end_time=30.0,
    output_video_path="tracked.mp4"
)
```

SAM3 uses text prompts for detection. Configure target classes in `config/models_config.yaml`:
```yaml
object_detection:
  target_classes:
    - "soldier"
    - "tank"
    - "truck"
  sam3_config:
    checkpoint_path: null  # Auto-download, or specify local path
    iou_threshold: 0.5
```

See [SAM3_MIGRATION_GUIDE.md](SAM3_MIGRATION_GUIDE.md) for detailed documentation.

### Custom Object Classes
Edit `config/models_config.yaml`:
```yaml
object_detection:
  target_classes:
    - "truck"
    - "tank"
    - "soldier"
    - "helicopter"  # Add custom class
```

### Adjust Segment Duration
```python
system.analyze_video(
    video_path="video.mp4",
    segment_duration=60  # 60 seconds per segment
)
```

### Export Analysis Results
```python
system.export_results(
    video_id="m-xxxxx",
    output_path="results/analysis.json"
)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch sizes in `config/models_config.yaml`
- Process fewer segments at once
- Use smaller models

### Model Download Issues
- Check internet connection (for initial model downloads)
- Verify HuggingFace Hub access
- Manually download and specify local paths in config

### Local Storage Issues
- Ensure write permissions for `./local_videodb/` directory
- Check disk space availability
- Verify FAISS index is being saved properly

### Agent Timeouts
- Increase `max_steps` in agent config
- Simplify queries
- Check model loading status

### SAM3 Detection Issues
- **Module not found**: Install with `cd SAM3_tracking/sam3 && pip install -e .`
- **Checkpoint missing**: SAM3 auto-downloads checkpoints; for offline use, set `checkpoint_path` in config
- **Out of memory**: Reduce batch size or use `gpus_to_use` config for multi-GPU
- **Detection misses**: Adjust `iou_threshold` or add custom text prompts in `object_detection.py`

## 📝 Performance Notes

### Processing Times (A100 80GB)
- Video upload: ~1-5 seconds
- Object detection per segment: ~2-3 seconds
- Embedding generation: ~0.5 seconds per segment
- Event description: ~3-5 seconds per segment
- Total for 10-minute video (30s segments): ~3-5 minutes

### Resource Requirements
- GPU Memory: 40-60 GB (all models loaded)
- CPU RAM: 32 GB recommended
- Storage: ~1-2 GB per hour of analyzed footage

## 🤝 Contributing

This is a research project for battlefield reconnaissance analysis. Contributions welcome for:
- Additional object detection models
- Improved tracking algorithms
- Enhanced VLM descriptions
- Performance optimizations

## 📄 License

See individual model licenses:
- SAM3: Check model license
- PE-Core: Meta license
- SmolVLM2: HuggingFace license
- EXAONE: LG AI Research license

## 🙏 Acknowledgments

Built using:
- [smolagents](https://github.com/huggingface/smolagents) - Agent framework
- [Perception Encoders](https://github.com/facebookresearch/perception-encoders) - Embeddings
- SAM3, SmolVLM2, EXAONE - AI models
- [milsymbol](https://github.com/spatialillusions/milsymbol) - NATO military symbols

---

**Note**: This system is designed for closed network environments with local model hosting. Ensure all models are downloaded and configured for offline use if required.
