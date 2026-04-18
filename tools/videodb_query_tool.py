"""
VideoDBQueryTool: Smolagents tool for querying video database
ALL TOOLS NOW RETURN PYTHON DICTS (not JSON strings) for direct field access by agent
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import smolagents
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

from smolagents import tool

# Import core components
from core_src.videodb_manager import VideoDBManager
from core_src.embedding_generator import EmbeddingGenerator


# Global instances
_videodb_managers = {}  # Dictionary of collection_name -> VideoDBManager
_embedding_generator = None
_selected_video_ids = []  # Changed to list to support multiple videos
_active_collections = ["battlefield_reconnaissance"]  # Active collections for querying
_video_to_collection_map = {}  # Cache mapping video_id -> collection_name
_target_classes = None  # Will be loaded from config


def _load_target_classes_from_config() -> List[str]:
    """
    Load target classes from config/models_config.yaml.

    Returns:
        List of target class names (e.g., ["soldier", "tank", "truck"])
    """
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "models_config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        target_classes = config.get("object_detection", {}).get("target_classes", ["soldier", "tank", "truck"])
        return [cls.lower() for cls in target_classes]
    except Exception as e:
        print(f"Warning: Could not load target classes from config: {e}")
        return ["soldier", "tank", "truck"]  # Fallback defaults


def get_target_classes() -> List[str]:
    """
    Get the list of valid target classes for object queries.

    Returns:
        List of valid object class names
    """
    global _target_classes
    if _target_classes is None:
        _target_classes = _load_target_classes_from_config()
    return _target_classes


def set_target_classes(classes: List[str]):
    """
    Set the target classes programmatically.

    Args:
        classes: List of class names
    """
    global _target_classes
    _target_classes = [cls.lower() for cls in classes]


def get_videodb_manager(collection_name: str = "battlefield_reconnaissance") -> VideoDBManager:
    """
    Get or create VideoDBManager for a specific collection

    Args:
        collection_name: Collection to get manager for

    Returns:
        VideoDBManager instance for the collection
    """
    global _videodb_managers
    if collection_name not in _videodb_managers:
        _videodb_managers[collection_name] = VideoDBManager(
            storage_path="./data/local_videodb",
            collection_name=collection_name
        )
    return _videodb_managers[collection_name]


def get_collection_for_video(video_id: str) -> Optional[str]:
    """
    Find which collection a video belongs to

    Args:
        video_id: Video ID to find

    Returns:
        Collection name or None if not found
    """
    global _video_to_collection_map

    # Check cache first
    if video_id in _video_to_collection_map:
        return _video_to_collection_map[video_id]

    # Search through active collections
    from core_src.context_scanner import ContextScanner
    scanner = ContextScanner()

    for collection_id in get_active_collections():
        videos = scanner.scan_available_videos(collection_ids=[collection_id])
        for video in videos:
            if video['video_id'] == video_id:
                # Cache the mapping
                _video_to_collection_map[video_id] = collection_id
                return collection_id

    return None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global EmbeddingGenerator instance"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(
            model_name="PE-Core-L14-336",
            device="cuda",
            normalize=True
        )
    return _embedding_generator


def set_selected_video_ids(video_ids: list):
    """
    Set the selected video IDs for queries

    Args:
        video_ids: List of video IDs to query across
    """
    global _selected_video_ids
    _selected_video_ids = video_ids if video_ids else []


def get_selected_video_ids() -> list:
    """
    Get the selected video IDs

    Returns:
        List of selected video IDs
    """
    return _selected_video_ids


def set_active_collections(collection_ids: list):
    """
    Set the active collections for queries

    Args:
        collection_ids: List of collection IDs to query from
    """
    global _active_collections
    _active_collections = collection_ids if collection_ids else ["battlefield_reconnaissance"]
    print(f"[VideoDBTool] Active collections updated: {_active_collections}")


def get_active_collections() -> list:
    """
    Get the active collections

    Returns:
        List of active collection IDs
    """
    return _active_collections


def get_available_video_ids() -> list:
    """
    Get all available video IDs from active collections

    Returns:
        List of all available video IDs from active collections
    """
    try:
        # Get active collections
        active_collections = get_active_collections()

        # Scan videos from active collections
        from core_src.context_scanner import ContextScanner
        scanner = ContextScanner()
        videos = scanner.scan_available_videos(collection_ids=active_collections)

        video_ids = [v['video_id'] for v in videos]
        print(f"[VideoDBTool] Found {len(video_ids)} videos in active collections: {active_collections}")
        return video_ids
    except Exception as e:
        print(f"Error getting available video IDs: {e}")
        return []


def validate_video_ids(video_ids: list) -> Dict[str, Any]:
    """
    Validate that video IDs exist and are accessible

    Args:
        video_ids: List of video IDs to validate

    Returns:
        Dict with 'valid' (list of valid IDs), 'invalid' (list of invalid IDs), 'status'
    """
    available_ids = get_available_video_ids()

    valid = [vid for vid in video_ids if vid in available_ids]
    invalid = [vid for vid in video_ids if vid not in available_ids]

    return {
        "valid": valid,
        "invalid": invalid,
        "status": "success" if not invalid else "partial" if valid else "error"
    }


def enforce_selected_contexts(video_ids: Optional[list] = None) -> Dict[str, Any]:
    """
    Enforce that only selected video contexts can be queried.

    Args:
        video_ids: Optional list of video IDs to use. If None, uses selected contexts.
                  If provided, validates they are in selected contexts.

    Returns:
        Dict with:
        - "status": "success" or "error"
        - "video_ids": List of valid video IDs to use
        - "message": Error or info message
    """
    selected_videos = get_selected_video_ids()

    # If no videos selected at all
    if not selected_videos:
        return {
            "status": "error",
            "video_ids": [],
            "message": "No videos selected. Please select videos from Context Catalog tab first. Use get_selected_contexts() to check."
        }

    # If video_ids explicitly provided, validate they're in selected
    if video_ids is not None:
        # Check if provided IDs are in selected contexts
        not_selected = [vid for vid in video_ids if vid not in selected_videos]

        if not_selected:
            return {
                "status": "error",
                "video_ids": [],
                "message": f"Cannot query videos {not_selected} - they are not selected. Selected videos: {selected_videos}. Use get_selected_contexts() to see selected contexts."
            }

        # All provided IDs are in selected
        return {
            "status": "success",
            "video_ids": video_ids,
            "message": f"Using {len(video_ids)} selected video(s)"
        }

    # Use selected contexts (video_ids was None)
    return {
        "status": "success",
        "video_ids": selected_videos,
        "message": f"Using {len(selected_videos)} selected video(s)"
    }


# Backward compatibility functions
def set_current_video_id(video_id: str):
    """
    Set a single video ID (backward compatibility)

    Args:
        video_id: Video ID to set
    """
    set_selected_video_ids([video_id] if video_id else [])


def get_current_video_id() -> Optional[str]:
    """
    Get the first selected video ID (backward compatibility)

    Returns:
        First video ID or None
    """
    video_ids = get_selected_video_ids()
    return video_ids[0] if video_ids else None


# =============================================================================
# JSON SCHEMAS FOR TOOL OUTPUTS
# These schemas tell the agent exactly what structure to expect from each tool
# =============================================================================

QUERY_SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "no_results", "error"],
            "description": "Operation status"
        },
        "query": {
            "type": "string",
            "description": "The search query used"
        },
        "video_id": {
            "type": "string",
            "description": "Video that was searched"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of segments found"
        },
        "segments": {
            "type": "array",
            "description": "List of matching segments ordered by similarity",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer", "description": "Segment ID for use with get_segment_details"},
                    "start_time": {"type": "string", "description": "Start time like '0.0s'"},
                    "end_time": {"type": "string", "description": "End time like '30.0s'"},
                    "similarity_score": {"type": "string", "description": "Match score like '0.892'"},
                    "event_description": {"type": "string", "description": "AI-generated description of what's happening"},
                    "detected_objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of object types detected like ['tank', 'truck']"
                    }
                },
                "required": ["segment_id", "event_description"]
            }
        },
        "message": {
            "type": "string",
            "description": "Error or status message if applicable"
        }
    },
    "required": ["status"]
}

QUERY_BY_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "no_results", "error"],
            "description": "Operation status"
        },
        "object_type": {
            "type": "string",
            "description": "The object type that was searched"
        },
        "object_type_singular": {
            "type": "string",
            "description": "Singular form for display (e.g., 'tank')"
        },
        "object_type_plural": {
            "type": "string",
            "description": "Plural form for display (e.g., 'tanks')"
        },
        "timeline_text": {
            "type": "string",
            "description": "PRE-FORMATTED timeline ready to show user. Use this directly in response!"
        },
        "summary_text": {
            "type": "string",
            "description": "Summary string like 'Maximum 3 tanks at any point (2 segments)'"
        },
        "max_objects_at_any_point": {
            "type": "integer",
            "description": "Maximum count observed in any single segment"
        },
        "num_segments": {
            "type": "integer",
            "description": "Number of segments where object was detected"
        },
        "timeline": {
            "type": "array",
            "description": "Timeline entries for programmatic access",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "Video ID"},
                    "segment_id": {"type": "integer", "description": "Segment ID"},
                    "start_time_hms": {"type": "string", "description": "Start time as HH:MM:SS"},
                    "end_time_hms": {"type": "string", "description": "End time as HH:MM:SS"},
                    "object_count": {"type": "integer", "description": "Count of objects in this segment"},
                    "description": {"type": "string", "description": "What's happening in the segment"}
                }
            }
        },
        "message": {
            "type": "string",
            "description": "Error or status message if applicable"
        }
    },
    "required": ["status"]
}

QUERY_BY_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "no_results", "error"],
            "description": "Operation status"
        },
        "event_query": {
            "type": "string",
            "description": "The event keywords searched"
        },
        "video_id": {
            "type": "string",
            "description": "Video that was searched"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of segments found"
        },
        "segments": {
            "type": "array",
            "description": "List of matching segments",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer", "description": "Segment ID for use with get_segment_details"},
                    "start_time": {"type": "string", "description": "Start time like '0.0s'"},
                    "end_time": {"type": "string", "description": "End time like '30.0s'"},
                    "event_description": {"type": "string", "description": "AI-generated event description"},
                    "detected_objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of object types detected"
                    }
                },
                "required": ["segment_id", "event_description"]
            }
        },
        "message": {
            "type": "string",
            "description": "Error or status message if applicable"
        }
    },
    "required": ["status"]
}

VIDEO_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error"],
            "description": "Operation status"
        },
        "video_id": {
            "type": "string",
            "description": "Video ID"
        },
        "video_name": {
            "type": "string",
            "description": "Original filename"
        },
        "video_length": {
            "type": "string",
            "description": "Total duration like '120.0s'"
        },
        "num_segments": {
            "type": "integer",
            "description": "Total number of segments in video"
        },
        "object_counts": {
            "type": "object",
            "description": "Total counts by object type, e.g. {'tank': 12, 'truck': 5}",
            "additionalProperties": {"type": "integer"}
        },
        "stream_url": {
            "type": "string",
            "description": "Path to video file"
        },
        "player_url": {
            "type": "string",
            "description": "Path to video file"
        },
        "message": {
            "type": "string",
            "description": "Error message if applicable"
        }
    },
    "required": ["status"]
}

SEGMENT_DETAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error"],
            "description": "Operation status"
        },
        "video_id": {
            "type": "string",
            "description": "Video ID"
        },
        "segment_id": {
            "type": "integer",
            "description": "Segment ID"
        },
        "start_time": {
            "type": "string",
            "description": "Start time like '0.0s'"
        },
        "end_time": {
            "type": "string",
            "description": "End time like '30.0s'"
        },
        "event_description": {
            "type": "string",
            "description": "AI-generated description of what's happening"
        },
        "total_objects": {
            "type": "integer",
            "description": "Total count of all objects in segment"
        },
        "object_counts": {
            "type": "object",
            "description": "Counts by type, e.g. {'tank': 3, 'soldier': 2}",
            "additionalProperties": {"type": "integer"}
        },
        "objects": {
            "type": "array",
            "description": "List of individual detected objects",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string", "description": "Object type (tank/truck/soldier)"},
                    "confidence": {"type": "string", "description": "Detection confidence like '0.945'"},
                    "position": {"type": "string", "description": "Position like '(512, 384)'"}
                },
                "required": ["class", "confidence"]
            }
        },
        "message": {
            "type": "string",
            "description": "Error message if applicable"
        }
    },
    "required": ["status"]
}

SET_ACTIVE_VIDEO_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error"],
            "description": "Operation status"
        },
        "message": {
            "type": "string",
            "description": "Confirmation or error message"
        },
        "active_video_id": {
            "type": "string",
            "description": "The video ID now set as active"
        }
    },
    "required": ["status", "message"]
}


@tool
def get_selected_contexts() -> dict:
    """
    Get the currently selected video and PDF contexts from the Context Catalog.
    ALWAYS call this FIRST before querying to know what contexts are available.

    Returns:
        Dictionary with selected contexts:
        - status: "success" or "error"
        - selected_videos: List of selected video IDs
        - selected_pdfs: List of selected PDF filenames
        - num_videos: Number of selected videos
        - num_pdfs: Number of selected PDFs
        - message: Information about selected contexts
    """
    try:
        from tools.pdf_rag_tool import get_selected_pdf_files

        selected_videos = get_selected_video_ids()
        selected_pdfs = get_selected_pdf_files()

        if not selected_videos and not selected_pdfs:
            return {
                "status": "error",
                "selected_videos": [],
                "selected_pdfs": [],
                "num_videos": 0,
                "num_pdfs": 0,
                "message": "No contexts selected. User must select videos and/or PDFs from the Context Catalog tab before querying."
            }

        return {
            "status": "success",
            "selected_videos": selected_videos,
            "selected_pdfs": selected_pdfs,
            "num_videos": len(selected_videos),
            "num_pdfs": len(selected_pdfs),
            "message": f"Selected {len(selected_videos)} video(s) and {len(selected_pdfs)} PDF(s). Use these IDs for querying."
        }

    except Exception as e:
        return {
            "status": "error",
            "selected_videos": [],
            "selected_pdfs": [],
            "num_videos": 0,
            "num_pdfs": 0,
            "message": f"Error getting selected contexts: {str(e)}"
        }


@tool
def query_video_semantic(query: str, video_ids: Optional[list] = None, top_k: int = 5) -> dict:
    """
    Search video segments using semantic similarity to find matching visual content.
    Can search across multiple videos simultaneously.

    Args:
        query: Search query describing what to find (e.g. "tanks moving through forest")
        video_ids: List of video IDs to search. If not provided, uses currently selected videos
        top_k: Maximum number of results to return across all videos

    Returns:
        Dictionary with search results. Access fields directly like result['segments'].
        - status: "success", "no_results", or "error"
        - query: The search query used
        - video_ids: List of videos that were searched
        - num_results: Number of segments found
        - segments: List of matching segments with:
            - video_id: Video this segment belongs to
            - segment_id: Integer segment ID
            - start_time: Start time string
            - end_time: End time string
            - similarity_score: Match score (higher = better match)
            - event_description: What's happening in the segment
            - detected_objects: List of object types found

    Example:
        result = query_video_semantic("tanks moving", video_ids=["m-abc123", "m-xyz789"])
        for seg in result["segments"]:
            print(f"Video {seg['video_id']}, Segment {seg['segment_id']}: {seg['event_description']}")
    """
    try:
        embedding_generator = get_embedding_generator()

        # Enforce selected contexts
        context_check = enforce_selected_contexts(video_ids)
        if context_check["status"] == "error":
            return {
                "status": "error",
                "message": context_check["message"]
            }

        # Use validated video IDs from selected contexts
        video_ids = context_check["video_ids"]

        # Generate query embedding once
        query_embedding = embedding_generator.encode_text(query)

        # Collect results from all videos
        all_results = []
        for video_id in video_ids:
            try:
                # Get the collection for this video
                collection_name = get_collection_for_video(video_id)
                if not collection_name:
                    print(f"Warning: Could not find collection for video {video_id}")
                    continue

                # Get the correct manager for this video's collection
                videodb_manager = get_videodb_manager(collection_name)

                results = videodb_manager.query_by_similarity(
                    query_embedding=query_embedding,
                    video_id=video_id,
                    top_k=top_k * 2,  # Get more results per video for better cross-video ranking
                    threshold=0.6
                )

                # Add video_id to each result
                for result in results:
                    result["source_video_id"] = video_id
                    all_results.append(result)

            except Exception as e:
                print(f"Error querying video {video_id}: {e}")
                continue

        if not all_results:
            return {
                "status": "no_results",
                "message": f"No segments found matching query: {query}",
                "video_ids": video_ids,
                "query": query,
                "num_results": 0,
                "segments": []
            }

        # Sort all results by similarity score and take top_k
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = all_results[:top_k]

        # Format results
        formatted_results = []
        for result in top_results:
            formatted_results.append({
                "video_id": result["source_video_id"],
                "segment_id": result["segment_id"],
                "start_time": f"{result['start_time']:.1f}s",
                "end_time": f"{result['end_time']:.1f}s",
                "similarity_score": f"{result['similarity_score']:.3f}",
                "event_description": result["event_description"],
                "detected_objects": [obj.get("class") for obj in result.get("objects", [])]
            })

        return {
            "status": "success",
            "query": query,
            "video_ids": video_ids,
            "num_results": len(formatted_results),
            "segments": formatted_results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in semantic search: {str(e)}"
        }


def _format_seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@tool
def query_video_by_object(object_type: str, video_ids: Optional[list] = None, min_count: int = 1) -> dict:
    """
    Search video segments containing specific detected objects.
    Returns a TIMELINE showing when and how many objects appear in each segment.

    IMPORTANT: When user asks "How many X are in the video?", use the 'timeline_text' field
    to show segment-by-segment counts. Do NOT just report a single total number.

    Args:
        object_type: Type of object to search for (e.g., "truck", "tank", "soldier").
                    Valid types are configured in config/models_config.yaml target_classes.
        video_ids: List of video IDs to search. If not provided, uses currently selected videos
        min_count: Minimum number of objects required in a segment

    Returns:
        Dictionary with timeline-based results:
        - status: "success", "no_results", or "error"
        - object_type: The object type searched (e.g., "tank")
        - object_type_singular: Singular form for display (e.g., "tank")
        - object_type_plural: Plural form for display (e.g., "tanks")
        - timeline_text: PRE-FORMATTED timeline string ready to show user, e.g.:
            "00:00:00 ~ 00:00:30: 2 tanks appear
             00:00:30 ~ 00:01:00: 3 tanks appear"
        - summary_text: Summary string, e.g.: "Maximum 3 tanks at any point (2 segments)"
        - max_objects_at_any_point: Maximum count observed in any single segment
        - num_segments: Number of segments where object was detected
        - timeline: List of timeline entries for programmatic access:
            - start_time_hms: Start time as "HH:MM:SS"
            - end_time_hms: End time as "HH:MM:SS"
            - object_count: Number of objects in this segment
            - description: What's happening in the segment

    Example response to user:
        "{object_type_plural} appear in the video as follows:

        {timeline_text}

        Summary: {summary_text}"

    Example:
        result = query_video_by_object("tank", video_ids=["m-abc123"])
        # Use the pre-formatted timeline_text directly:
        answer = f"Tanks appear in the video as follows:\\n\\n{result['timeline_text']}\\n\\nSummary: {result['summary_text']}"
    """
    try:
        # Enforce selected contexts
        context_check = enforce_selected_contexts(video_ids)
        if context_check["status"] == "error":
            return {
                "status": "error",
                "message": context_check["message"]
            }

        # Use validated video IDs from selected contexts
        video_ids = context_check["video_ids"]

        # Validate object type - load valid classes from config
        valid_objects = get_target_classes()
        if object_type.lower() not in valid_objects:
            return {
                "status": "error",
                "message": f"Invalid object type. Must be one of: {valid_objects}"
            }

        # Collect results from all videos with temporal grouping
        all_results = []
        video_results = {}  # Group by video for temporal analysis

        for video_id in video_ids:
            try:
                # Get the collection for this video
                collection_name = get_collection_for_video(video_id)
                if not collection_name:
                    print(f"Warning: Could not find collection for video {video_id}")
                    continue

                # Get the correct manager for this video's collection
                videodb_manager = get_videodb_manager(collection_name)

                results = videodb_manager.query_by_object_type(
                    object_type=object_type.lower(),
                    video_id=video_id,
                    min_count=min_count
                )

                # Initialize video group
                if video_id not in video_results:
                    video_results[video_id] = []

                # Add video_id to each result and group
                for result in results:
                    result["source_video_id"] = video_id
                    all_results.append(result)
                    video_results[video_id].append(result)

            except Exception as e:
                print(f"Error querying video {video_id}: {e}")
                continue

        # Determine singular/plural forms
        obj_type_lower = object_type.lower()
        # Handle common pluralization
        if obj_type_lower.endswith('s'):
            obj_singular = obj_type_lower
            obj_plural = obj_type_lower
        else:
            obj_singular = obj_type_lower
            obj_plural = obj_type_lower + "s"

        if not all_results:
            return {
                "status": "no_results",
                "message": f"No {obj_plural} found in the video",
                "object_type": object_type,
                "object_type_singular": obj_singular,
                "object_type_plural": obj_plural,
                "timeline_text": f"No {obj_plural} were detected in this video.",
                "summary_text": f"No {obj_plural} detected",
                "max_objects_at_any_point": 0,
                "num_segments": 0,
                "timeline": []
            }

        # Sort all results by segment_id for timeline
        all_results.sort(key=lambda x: (x["source_video_id"], x["segment_id"]))

        # Build timeline with HH:MM:SS format
        timeline = []
        timeline_lines = []
        max_count = 0

        for result in all_results:
            start_seconds = result["start_time"]
            end_seconds = result["end_time"]
            count = result["object_count"]

            start_hms = _format_seconds_to_hms(start_seconds)
            end_hms = _format_seconds_to_hms(end_seconds)

            # Track max count
            if count > max_count:
                max_count = count

            # Build timeline entry
            timeline.append({
                "video_id": result["source_video_id"],
                "segment_id": result["segment_id"],
                "start_time_hms": start_hms,
                "end_time_hms": end_hms,
                "object_count": count,
                "description": result.get("event_description", "")
            })

            # Build timeline text line with proper singular/plural
            unit = obj_singular if count == 1 else obj_plural
            timeline_lines.append(f"{start_hms} ~ {end_hms}: {count} {unit} appear")

        # Create pre-formatted timeline text
        timeline_text = "\n".join(timeline_lines)

        # Create summary text
        num_segments = len(all_results)
        unit_summary = obj_singular if max_count == 1 else obj_plural
        summary_text = f"Maximum {max_count} {unit_summary} observed at any point ({num_segments} segment{'s' if num_segments != 1 else ''})"

        return {
            "status": "success",
            "object_type": object_type,
            "object_type_singular": obj_singular,
            "object_type_plural": obj_plural,
            "timeline_text": timeline_text,
            "summary_text": summary_text,
            "max_objects_at_any_point": max_count,
            "num_segments": num_segments,
            "timeline": timeline
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error querying by object: {str(e)}"
        }


@tool
def query_video_by_event(event_query: str, video_ids: Optional[list] = None, top_k: int = 5) -> dict:
    """
    Search video segments by event description keywords.
    Can search across multiple videos simultaneously.

    Args:
        event_query: Keywords to search in AI-generated descriptions (e.g. "moving", "stationary")
        video_ids: List of video IDs to search. If not provided, uses currently selected videos
        top_k: Maximum number of results across all videos

    Returns:
        Dictionary with search results. Access fields directly.
        - status: "success", "no_results", or "error"
        - event_query: Keywords searched
        - video_ids: List of videos searched
        - num_results: Number found
        - segments: List with:
            - video_id: Video this segment belongs to
            - segment_id: Integer ID
            - start_time: Start time
            - end_time: End time
            - event_description: Full description
            - detected_objects: List of object types

    Example:
        result = query_video_by_event("moving", video_ids=["m-abc123", "m-xyz789"])
        for seg in result["segments"]:
            print(f"Video {seg['video_id']}, Segment {seg['segment_id']}: {seg['event_description']}")
    """
    try:
        # Enforce selected contexts
        context_check = enforce_selected_contexts(video_ids)
        if context_check["status"] == "error":
            return {
                "status": "error",
                "message": context_check["message"]
            }

        # Use validated video IDs from selected contexts
        video_ids = context_check["video_ids"]

        # Collect results from all videos
        all_results = []
        for video_id in video_ids:
            try:
                # Get the collection for this video
                collection_name = get_collection_for_video(video_id)
                if not collection_name:
                    print(f"Warning: Could not find collection for video {video_id}")
                    continue

                # Get the correct manager for this video's collection
                videodb_manager = get_videodb_manager(collection_name)

                results = videodb_manager.query_by_event_description(
                    query=event_query,
                    video_id=video_id,
                    top_k=top_k * 2  # Get more results per video
                )

                # Add video_id to each result
                for result in results:
                    result["source_video_id"] = video_id
                    all_results.append(result)

            except Exception as e:
                print(f"Error querying video {video_id}: {e}")
                continue

        if not all_results:
            return {
                "status": "no_results",
                "message": f"No segments found matching: {event_query}",
                "video_ids": video_ids,
                "event_query": event_query,
                "num_results": 0,
                "segments": []
            }

        # Take top_k results
        all_results = all_results[:top_k]

        # Format results
        formatted_results = []
        for result in all_results:
            formatted_results.append({
                "video_id": result["source_video_id"],
                "segment_id": result["segment_id"],
                "start_time": f"{result['start_time']:.1f}s",
                "end_time": f"{result['end_time']:.1f}s",
                "event_description": result["event_description"],
                "detected_objects": [obj.get("class") for obj in result.get("objects", [])]
            })

        return {
            "status": "success",
            "event_query": event_query,
            "video_ids": video_ids,
            "num_results": len(formatted_results),
            "segments": formatted_results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error querying by event: {str(e)}"
        }


@tool
def get_video_summary(video_ids: Optional[list] = None) -> dict:
    """
    Get overall summary statistics for one or more videos.

    Args:
        video_ids: List of video IDs. If not provided, uses currently selected videos

    Returns:
        Dictionary with video summaries. Access fields directly.
        - status: "success" or "error"
        - num_videos: Number of videos summarized
        - summaries: List of summaries, each containing:
            - video_id: Video ID
            - video_name: Original filename
            - video_length: Duration as string
            - num_segments: Total number of segments
            - object_counts: Dict mapping object types to counts
            - stream_url: Path to video file
            - player_url: Path to video file

    Example:
        result = get_video_summary(video_ids=["m-abc123", "m-xyz789"])
        for summary in result["summaries"]:
            print(f"Video {summary['video_id']}: {summary['object_counts']}")
    """
    try:
        # Enforce selected contexts
        context_check = enforce_selected_contexts(video_ids)
        if context_check["status"] == "error":
            return {
                "status": "error",
                "message": context_check["message"]
            }

        # Use validated video IDs from selected contexts
        video_ids = context_check["video_ids"]

        # Collect summaries for all videos
        summaries = []
        for video_id in video_ids:
            try:
                # Get the collection for this video
                collection_name = get_collection_for_video(video_id)
                if not collection_name:
                    print(f"Warning: Could not find collection for video {video_id}")
                    continue

                # Get the correct manager for this video's collection
                videodb_manager = get_videodb_manager(collection_name)

                summary = videodb_manager.get_video_summary(video_id)

                if summary:
                    summaries.append({
                        "video_id": video_id,
                        "video_name": summary.get("video_name", ""),
                        "video_length": f"{summary.get('video_length', 0):.1f}s",
                        "num_segments": summary.get("num_segments", 0),
                        "object_counts": summary.get("object_counts", {}),
                        "stream_url": summary.get("stream_url", ""),
                        "player_url": summary.get("player_url", "")
                    })
            except Exception as e:
                print(f"Error getting summary for video {video_id}: {e}")
                continue

        if not summaries:
            return {
                "status": "error",
                "message": f"No data found for videos: {video_ids}"
            }

        return {
            "status": "success",
            "num_videos": len(summaries),
            "summaries": summaries
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting video summaries: {str(e)}"
        }


@tool
def get_segment_details(segment_id: int, video_id: Optional[str] = None) -> dict:
    """
    Get detailed information for a specific video segment.

    Args:
        segment_id: Integer ID of the segment (e.g. 0, 1, 2). Get this from query results
        video_id: Video ID. If not provided, uses currently active video

    Returns:
        Dictionary with segment details. Access fields directly like details['object_counts'].
        - status: "success" or "error"
        - video_id: Video ID
        - segment_id: Segment ID
        - start_time: Start time as string
        - end_time: End time as string
        - event_description: AI-generated description of what's happening
        - total_objects: Total count of all objects
        - object_counts: Dict mapping types to counts, e.g. {"tank": 3, "soldier": 2}
        - objects: List of individual objects with:
            - class: Object type ("tank", "truck", "soldier")
            - confidence: Detection confidence score
            - position: Position as string "(x, y)"

    Example:
        details = get_segment_details(segment_id=0, video_id="m-abc123")
        tank_count = details["object_counts"]["tank"]
        print(f"Segment has {tank_count} tanks")
        for obj in details["objects"]:
            if obj["class"] == "tank":
                print(f"Tank at {obj['position']} with confidence {obj['confidence']}")
    """
    try:
        # Get selected videos
        selected_videos = get_selected_video_ids()

        if not selected_videos:
            return {
                "status": "error",
                "message": "No videos selected. Please select videos from Context Catalog tab first. Use get_selected_contexts() to check."
            }

        # Determine video_id and validate it's in selected contexts
        if video_id is None:
            # Use first selected video if not specified
            video_id = selected_videos[0]
        else:
            # Validate provided video_id is in selected contexts
            if video_id not in selected_videos:
                return {
                    "status": "error",
                    "message": f"Cannot query video {video_id} - it is not selected. Selected videos: {selected_videos}. Use get_selected_contexts() to see selected contexts."
                }

        # Get the collection for this video
        collection_name = get_collection_for_video(video_id)
        if not collection_name:
            return {
                "status": "error",
                "message": f"Could not find collection for video {video_id}"
            }

        # Get the correct manager for this video's collection
        videodb_manager = get_videodb_manager(collection_name)

        # Get metadata
        metadata = videodb_manager.get_segment_metadata(video_id, segment_id)

        if metadata is None:
            return {
                "status": "error",
                "message": f"No data found for segment {segment_id} in video {video_id}"
            }

        # Count objects by type
        objects = metadata.get("objects", [])
        object_counts = {}
        for obj in objects:
            obj_class = obj.get("class", "unknown")
            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1

        return {
            "status": "success",
            "video_id": video_id,
            "segment_id": segment_id,
            "start_time": f"{metadata.get('start_time', 0):.1f}s",
            "end_time": f"{metadata.get('end_time', 0):.1f}s",
            "event_description": metadata.get("event_description", ""),
            "total_objects": len(objects),
            "object_counts": object_counts,
            "objects": [
                {
                    "class": obj.get("class"),
                    "confidence": f"{obj.get('confidence', 0):.3f}",
                    "position": f"({obj.get('x', 0):.0f}, {obj.get('y', 0):.0f})"
                }
                for obj in objects
            ]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting segment details: {str(e)}"
        }


@tool
def set_active_video(video_id: str) -> dict:
    """
    Set a single active video for subsequent queries (backward compatibility).
    For multiple videos, use set_active_videos instead.

    Args:
        video_id: Video ID to set as active (e.g. "m-abc123")

    Returns:
        Dictionary confirming the active video. Access fields directly.
        - status: "success" or "error"
        - message: Confirmation message
        - active_video_id: The video ID now set as active

    Example:
        result = set_active_video("m-abc123")
        if result["status"] == "success":
            print(f"Now querying video: {result['active_video_id']}")
    """
    try:
        # Validate video_id exists in database
        available_ids = get_available_video_ids()

        if not available_ids:
            return {
                "status": "error",
                "message": "No videos available in database. Upload and analyze videos first."
            }

        if video_id not in available_ids:
            return {
                "status": "error",
                "message": f"Video ID {video_id} does not exist. Available videos: {available_ids}. Use get_selected_contexts() to see available videos."
            }

        # Set as active
        set_current_video_id(video_id)

        return {
            "status": "success",
            "message": f"Active video set to: {video_id}",
            "active_video_id": video_id
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error setting active video: {str(e)}"
        }


@tool
def set_active_videos(video_ids: list) -> dict:
    """
    Set multiple active videos for subsequent queries.
    This allows querying across multiple videos simultaneously.

    Args:
        video_ids: List of video IDs to set as active (e.g. ["m-abc123", "m-xyz789"])

    Returns:
        Dictionary confirming the active videos. Access fields directly.
        - status: "success" or "error"
        - message: Confirmation message
        - active_video_ids: List of video IDs now set as active
        - num_videos: Number of videos selected

    Example:
        result = set_active_videos(["m-abc123", "m-xyz789"])
        if result["status"] == "success":
            print(f"Now querying {result['num_videos']} videos")
    """
    try:
        # Validate all video_ids exist in database
        available_ids = get_available_video_ids()

        if not available_ids:
            return {
                "status": "error",
                "message": "No videos available in database. Upload and analyze videos first."
            }

        # Check for invalid IDs
        invalid_ids = [vid for vid in video_ids if vid not in available_ids]

        if invalid_ids:
            return {
                "status": "error",
                "message": f"Invalid video IDs: {invalid_ids}. Available videos: {available_ids}. Use get_selected_contexts() to see available videos."
            }

        # Set as active
        set_selected_video_ids(video_ids)

        return {
            "status": "success",
            "message": f"Active videos set to: {', '.join(video_ids)}",
            "active_video_ids": video_ids,
            "num_videos": len(video_ids)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error setting active videos: {str(e)}"
        }


# =============================================================================
# ASSIGN OUTPUT SCHEMAS TO TOOLS
# This makes Smolagents show the agent the full structure with Returns documentation
# =============================================================================

# Apply schemas to all tools
query_video_semantic.output_schema = QUERY_SEMANTIC_SCHEMA
query_video_by_object.output_schema = QUERY_BY_OBJECT_SCHEMA
query_video_by_event.output_schema = QUERY_BY_EVENT_SCHEMA
get_video_summary.output_schema = VIDEO_SUMMARY_SCHEMA
get_segment_details.output_schema = SEGMENT_DETAILS_SCHEMA
set_active_video.output_schema = SET_ACTIVE_VIDEO_SCHEMA
