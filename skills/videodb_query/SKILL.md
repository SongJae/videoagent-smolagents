---
name: videodb_query
description: Video database query skill for battlefield reconnaissance analysis. Use when analyzing drone footage, detecting military objects (tanks, trucks, soldiers), searching for specific events, or understanding video content. Always call get_contexts action first before other queries.
---

# Video Database Query Skill

## Overview

This skill provides comprehensive video analysis capabilities for battlefield reconnaissance. Use it to search for objects, events, and patterns in analyzed drone footage.

## Critical Rule: Always Get Contexts First

**Before any video query, you MUST call the `get_contexts` action first:**

```python
contexts = videodb_query_skill("get_contexts")
if contexts["status"] == "error":
    # No videos selected - inform user
    final_answer("Please select videos from the Context Catalog tab first.")
```

## Available Actions

### 1. get_contexts
Get currently selected video and PDF contexts. **Always call this first.**

```python
result = videodb_query_skill("get_contexts")
# Returns: selected_videos, selected_pdfs, num_videos, num_pdfs
```

### 2. get_summary
Get overview statistics for selected videos.

```python
result = videodb_query_skill("get_summary")
# Returns: video names, durations, object counts per video
```

### 3. semantic_search
Search video segments using natural language descriptions.

```python
result = videodb_query_skill("semantic_search", query="convoy moving through forest")
# Returns: matching segments with similarity scores
```

### 4. object_search
Find segments containing specific object types.

**Valid object types:** tank, truck, soldier

```python
result = videodb_query_skill("object_search", object_type="tank")
# Returns: timeline_text (pre-formatted), summary_text, max_objects_at_any_point
```

**Important:** Use `timeline_text` and `summary_text` directly in your response. Never sum object counts across segments - the same object may appear in multiple consecutive segments.

### 5. event_search
Search by event keywords in AI-generated descriptions.

```python
result = videodb_query_skill("event_search", query="moving")
# Returns: segments with matching descriptions
```

### 6. segment_details
Get detailed information about a specific segment.

```python
result = videodb_query_skill("segment_details", segment_id=0, video_ids=["m-abc123"])
# Returns: event_description, object_counts, individual object positions
```

## Temporal Continuity Awareness

**Critical Understanding:** The same object appearing in consecutive segments is NOT multiple objects. When reporting object counts:

- **WRONG:** "Found 45 tanks" (summing all detections)
- **CORRECT:** "Maximum 15 tanks observed at any point, appearing across 3 segments"

The `object_search` action returns pre-formatted `timeline_text` that handles this correctly. Use it directly.

## Multi-Video Analysis

When multiple videos are selected:

1. **Assess relationships:** Are videos sequential (same mission) or independent?
2. **Attribution:** Always specify which video findings come from
3. **Counting methodology:** Related videos may show same objects

### Example Multi-Video Workflow

```python
# Get contexts
contexts = videodb_query_skill("get_contexts")

# Query each video
for video_id in contexts["result"]["selected_videos"]:
    result = videodb_query_skill("object_search", object_type="tank", video_ids=[video_id])
    # Process results per video

# Provide context-aware response
```

## Result Structure

All actions return:
- `status`: "success", "no_results", or "error"
- `action`: The action performed
- `result`: Action-specific data
- `message`: Human-readable summary

## Best Practices

1. Always call `get_contexts` first
2. Check `status` before using results
3. Use `timeline_text` directly for object counts
4. Explain your counting methodology
5. Attribute findings to specific videos
6. Handle "no_results" gracefully (it's not an error)
