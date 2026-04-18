# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Battlefield Reconnaissance Video Agent System** - An AI-powered system for analyzing battlefield drone footage using SAM3 (unified object detection and tracking), PE-Core-L14-336 embeddings, SmolVLM2 event descriptions, and an agentic interface (EXAONE-4.0-32B-AWQ with smolagents).

This system operates **completely offline** - all models run locally, all data is stored in local file-based databases.

**Prerequisites:** Python 3.10+, CUDA 11.8+, NVIDIA GPU with 80GB VRAM (A100 recommended)

## Common Commands

```bash
# Launch Gradio web UI (recommended)
python main.py ui

# Analyze a video via CLI
python main.py analyze --video path/to/video.mp4 --output results/

# Query the agent
python main.py query --query "How many tanks?" --video-id m-xxxxx

# Add PDF to RAG system
python main.py add-pdf --pdf path/to/manual.pdf

# Check environment
python main.py check
```

### Installation

```bash
pip install -r requirements.txt
pip install -e smolagents/
cd perception_models && pip install -e . && cd ..
cd SAM3_tracking/sam3 && pip install -e . && cd ../..
cd map_mcp && pip install -e . && cd ..  # Optional: for tactical map
```

### Testing

```bash
python test_tool_signatures.py
python test_tool_returns.py
python test_agent_tool_view.py
python test_skill_memory_cache.py  # Test skill memory cache
```

### Syntax Check

```bash
python -m py_compile core_src/object_detection.py core_src/videodb_manager.py tools/videodb_query_tool.py
python -m py_compile skills/skill_manager.py skills/skill_tool.py skills/skill_memory_cache.py
```

## High-Level Architecture

### Processing Pipeline

```
Video Upload → VideoDBManager (local_videodb)
    → Stores in ./data/local_videodb/collections/{collection}/videos/
    → Assigns unique ID (e.g., m-abc123def45)

Segmentation → VideoProcessor
    → Splits into 30s segments, extracts frames

Object Detection → ObjectDetectionProcessor
    → SAM3 detects/tracks soldiers, tanks, trucks via text prompts
    → Stores bounding boxes, confidence scores, tracking IDs

Embeddings → EmbeddingGenerator
    → PE-Core-L14-336 generates 1024-dim vectors
    → Indexed in FAISS (IndexFlatIP)

Event Description → EventDescriptionGenerator
    → SmolVLM2 2.2B generates natural language descriptions

Storage
    → Metadata: JSON files via local_videodb
    → Embeddings: FAISS index per video ({video_id}_faiss.index)
    → Videos: Local filesystem copies
```

### Agent System

Uses **smolagents** CodeAgent with EXAONE-4.0-32B-AWQ:

- **Model Loading**: Centralized via `ModelManager` singleton in `core_src/model_manager.py`
- **Skills System**: Dynamic skill architecture in `skills/` - auto-discovers skills without code changes
- **Context Management**: Multi-video/PDF selection via global state (`_selected_video_ids`, `_selected_pdf_files`)
- **Tactical Map Context**: Unit positions/waypoints injected via `_prepare_message_with_context()`
- **CRITICAL**: Agent must call `videodb_query_skill("get_contexts")` FIRST before other skills
- **Custom Instructions**: `config/agent_custom_instructions.txt`

### Skills Architecture

Skills are dynamically discovered from `skills/` directory. Each skill has:
- `SKILL.md` - YAML frontmatter (name, description) + documentation
- `executor.py` - Defines `ACTIONS` dict and `execute(action, params)` function

**Available Skills:**
- `videodb_query` - Video database queries (get_contexts, object_search, semantic_search, etc.)
- `pdf_rag` - PDF document search
- `wargame_query` - Tactical map queries
- `final_response` - Report generation

**Skill Tools (in `skills/skill_tool.py`):**
- `get_available_skills()` - List all skills and their actions
- `get_skill_actions(skill_name)` - Get actions for specific skill
- `invoke_skill(skill_name, action, ...)` - Main entry point for dynamic invocation
- Convenience wrappers: `videodb_query_skill`, `pdf_rag_skill`, `wargame_query_skill`, `final_response_skill`

### Skill Memory Cache (Automatic)

The skills system includes an **automatic, transparent** memory cache layer (`skills/skill_memory_cache.py`):
- Caching happens automatically when using `invoke_skill()` - no extra tool calls needed
- Identical skill+action+params combinations return cached results instantly
- Reduces tool calls and improves inference efficiency

**How It Works:**
1. When `invoke_skill()` is called, the `SkillManager` checks cache first
2. If a matching result exists (same skill, action, params), it returns immediately
3. If not cached, the skill executes and the result is stored for future use
4. All cache operations are transparent - the agent just uses `invoke_skill()` normally

**Cache Components:**
- `SkillMemoryCache` - Singleton cache manager (TTL: 1 hour, max: 1000 entries)
- `CacheEntry` - Stores skill_name, action, params, result, timestamp, duration, hit_count
- Parameter normalization ensures equivalent params produce cache hits

**Non-Cacheable Skills (all actions always execute fresh):**
- `wargame_query` - Queries tactical map state that changes via UI in real-time

**Non-Cacheable Actions (always execute fresh):**
- `get_contexts` - Context state may change between calls
- `generate_report` - Always generates fresh reports

**Cache Metadata in Results:**
All skill results include `_cache` field for visibility:
```python
# Cache hit (instant return)
{"hit": True, "cache_id": "cache_xxx", "hit_count": 3}

# Cache miss (fresh execution, stored for next time)
{"hit": False, "cache_id": "cache_xxx", "stored": True, "duration_ms": 150.5}

# Non-cacheable skill (always fresh, never stored)
{"hit": False, "stored": False, "reason": "dynamic_skill:wargame_query"}
```

### Tool Conventions

1. **Return Types**: VideoDB tools return `dict`, PDF tools return `str` (JSON), Wargame tools return `dict`
2. **Error Handling**: Never raise exceptions - return `{"status": "error", "message": "..."}`
3. **Context Enforcement**: Tools validate `_selected_video_ids` before execution
4. **Wargame Tools**: 6 tools for tactical map queries (see Wargame Query Tools section)

### VideoDB Query Tools (8 tools in `tools/videodb_query_tool.py`)
- `get_selected_contexts()` - **MUST BE CALLED FIRST** - returns selected videos/PDFs
- `get_video_summary()` - Overview of video (duration, segments, object counts)
- `query_video_semantic()` - Semantic similarity search using embeddings
- `query_video_by_object()` - Find segments by detected object type (tank, truck, soldier)
- `query_video_by_event()` - Keyword search in event descriptions
- `get_segment_details()` - Detailed info for specific segment
- `set_active_video()` - Set default video for queries (deprecated, use context selection)
- `set_active_videos()` - Set multiple active videos

### PDF RAG Tools (2 tools in `tools/pdf_rag_tool.py`)
- `pdf_rag_search()` - Semantic search across indexed PDFs
- `get_indexed_pdfs()` - List all PDFs in knowledge base

### War Game Map (Query Agent Tab)

The Query Agent tab includes an integrated tactical map from `map_mcp`:

- **Left Column**: Interactive war game map with NATO military symbols (milsymbol.js)
- **Right Column**: Agent chat interface
- **Controls**: Add friendly/hostile units, select unit type/echelon, clear units
- **Architecture**: Uses `WarGameUI` from `map_mcp/src/map_mcp/wargame_ui.py`
- **State Sync**: Hidden Textbox bridge pattern for iframe-to-Gradio communication
- **State File**: `wargame_state.json` in temp directory (via `_get_temp_dir()`)

### Wargame Query Tools

6 tools in `tools/wargame_query_tool.py` for querying tactical map state:

- `get_tactical_situation()` - Overview of all units, positions, map center
- `get_friendly_units()` - List all friendly (FRIEND) units
- `get_hostile_units()` - List all hostile (HOSTILE) units
- `get_unit_details(unit_id)` - Detailed info for specific unit
- `get_unit_waypoints(unit_id)` - Waypoints/movement plan for unit
- `get_units_by_type(unit_type)` - Filter units by type (INFANTRY, ARMOR, etc.)

**Coordinate Format Handling**:
- `wargame_state.json` stores coordinates as lists: `[lat, lon]`
- Tools use `_parse_location()` and `_parse_waypoints()` helpers to handle both list and dict formats
- Output always normalized to `{"lat": float, "lon": float}` format

**Path Resolution**:
- Uses `_get_temp_dir()` from `map_mcp.wargame_ui` for dynamic temp path
- Shared singleton pattern ensures Gradio UI and query tools use same path

### Gradio UI Patterns

State sync pattern: Hidden Textbox (elem_id) → JavaScript updates value → `dispatchEvent('input')` → Python `.change()` handler

**Hidden Components (Gradio 5.x Compatibility):**
- Use `visible="hidden"` instead of `visible=False` for components that need JavaScript access
- `visible=False` doesn't render in DOM (Gradio 5.x optimization)
- `visible="hidden"` keeps element in DOM but hides it visually
- Reference: https://github.com/gradio-app/gradio/issues/11974

**JavaScript Injection:**
- Use `demo.load(fn=None, js=...)` for page-load scripts
- Event handlers support `js` parameter for client-side preprocessing

**Event Triggering from JavaScript:**
- Dispatch both `input` and `change` events for reliability
- Use `{ bubbles: true }` in Event constructor

## Key Directories

- `core_src/`: Video processing pipeline (model_manager.py, video_analysis_system.py, object_detection.py)
- `agent/`: Agent system (battlefield_agent.py, model_loader.py)
- `skills/`: Dynamic skill system (skill_manager.py, skill_tool.py, skill_memory_cache.py, {skill_name}/executor.py)
- `tools/`: Underlying tool implementations (videodb_query_tool.py, pdf_rag_tool.py, wargame_query_tool.py)
- `local_videodb/`: Custom offline video database module
- `ui/`: Gradio interface (gradio_app.py)
- `config/`: YAML configs and agent instructions
- `map_mcp/`: Offline tactical map with NATO symbols (integrated into Query Agent tab)

## Critical Implementation Notes

### Model Loading
- **Never create model instances directly** - use `ModelManager.get_global_model_manager()` singleton
- Pre-load with `model_manager.load_all_models(load_agent=True/False)`
- Get instances: `model_manager.get_video_system()`, `model_manager.get_agent()`

### VLLM Parameters
- `temperature` and `max_tokens` are per-request parameters, NOT init parameters
- Pass them via `agent.run(temperature=X, max_tokens=Y)`
- Only pass engine config to `VLLMModel()` init

### SAM3 Detection
- Unified detection + tracking (replaces LLMDet+SAM2)
- **CRITICAL**: SAM3 can only process ONE text prompt at a time per session
- For multi-class tracking: Process each class in a SEPARATE session, then merge results
- Uses `torch.autocast("cuda", dtype=torch.bfloat16)` to prevent dtype mismatch errors
- Target classes configured in `config/models_config.yaml` under `object_detection.target_classes`
- Text prompt mapping in `core_src/object_detection.py` `_build_text_prompts()`
- Reference: https://github.com/facebookresearch/sam3/issues/206

### Config-Driven Target Classes
Object detection target classes are configured centrally:
```yaml
# config/models_config.yaml
object_detection:
  target_classes:
    - "soldier"
    - "tank"
    - "truck"
```
- `ObjectDetectionProcessor` loads from config, supports custom text prompts
- `VideoDBManager.get_video_summary()` dynamically counts all detected classes
- `videodb_query_tool.get_target_classes()` loads from config for validation
- Adding new classes: Simply edit the config file - all components auto-update

### FAISS Indexes
- Per-video indexes: `{video_id}_faiss.index`
- Uses `IndexFlatIP` (inner product, not L2)
- Embeddings normalized before indexing
- Location: `./data/local_videodb/collections/{collection}/faiss/`

### War Game Map Integration
- `WarGameUI` instance created in `BattlefieldVideoAgentUI.__init__()`
- Static paths set up in `launch()` via `_setup_static_paths()`
- Assets served from `map_mcp/src/map_mcp/assets/` (leaflet.js, milsymbol.js)
- Temp HTML files served via Gradio file serving

### Local Storage Paths
- Videos: `./data/local_videodb/collections/{collection}/videos/`
- Metadata: `./data/local_videodb/collections/{collection}/metadata/`
- FAISS: `./data/local_videodb/collections/{collection}/faiss/`
- PDFs: `./data/pdfs/`
- RAG index: `./data/rag_index/`
- Map tiles: `./map_mcp/tiles/` (optional, uses placeholders if missing)

## Code Modification Guidelines

### Adding New Skills (Preferred)
New skills are auto-discovered without code changes:
1. Create directory: `skills/{skill_name}/`
2. Create `SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: skill_name
   description: What this skill does
   ---
   # Documentation here
   ```
3. Create `executor.py`:
   ```python
   ACTIONS = {
       "action_name": {"description": "...", "params": ["param1"]}
   }

   def execute(action: str, params: dict) -> dict:
       if action == "action_name":
           # Implementation
           return {"status": "success", "action": action, "result": ..., "message": ...}
   ```
4. Skill is automatically available via `invoke_skill("skill_name", "action_name", ...)`

### Adding New Tools (Legacy)
1. Use `@tool` decorator from smolagents with detailed docstring
2. Add type hints, return dict/str consistently
3. Handle errors gracefully - return error status, don't raise
4. Add to BOTH `battlefield_agent.py` AND `core_src/model_manager.py` tools lists (keep in sync)

### Modifying Video Processing
1. Check model pre-loading via ModelManager
2. Update pipeline in `core_src/video_analysis_system.py`
3. Consider FAISS index compatibility
4. Update UI display in `ui/gradio_app.py`

### Changing Gradio UI
1. Read `00_READ_ME_FIRST.txt` first
2. Assign unique `elem_id` to components accessed from JavaScript
3. Use hidden Textbox pattern for complex state

### Modifying War Game Map
1. Map logic in `map_mcp/src/map_mcp/wargame_map.py`
2. UI handlers in `ui/gradio_app.py` Query Agent tab section
3. Ensure static paths registered in `launch()` method
4. Query tools in `tools/wargame_query_tool.py` - use `_get_temp_dir()` for path resolution
5. Context injection in `ui/gradio_app.py` `_prepare_message_with_context()`

## Performance

- **GPU Memory**: Designed for 80GB VRAM (A100)
  - SAM3: ~10GB, Embeddings: ~5GB, VLM: ~10GB, Agent: ~25GB
- **Processing**: 10-minute video (30s segments) takes ~3-5 minutes on A100
- **Storage**: ~1-2 GB per hour of analyzed footage

## Troubleshooting

### CUDA Out of Memory
- Reduce batch sizes in `config/models_config.yaml`
- Process fewer segments at once

### SAM3 Detection Issues
- **Module not found**: Install with `cd SAM3_tracking/sam3 && pip install -e .`
- **Checkpoint missing**: SAM3 auto-downloads; for offline use, set `checkpoint_path` in config
- **Detection misses**: Adjust `iou_threshold` or add custom text prompts in `object_detection.py`

### Agent Timeouts
- Increase `max_steps` in agent config
- Check model loading status via `model_manager.check_models_ready()`
