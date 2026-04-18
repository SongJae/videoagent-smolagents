"""
VideoDBManager: Interface for local video database operations
Uses the existing local_videodb module with FAISS for embeddings
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add local_videodb to path
local_videodb_path = Path(__file__).parent.parent / "local_videodb"
sys.path.insert(0, str(local_videodb_path))

# Import local video database
import local_videodb

try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Semantic search will be disabled.")
    faiss = None


class VideoDBManager:
    """
    Manages all video database operations using local_videodb
    Adds FAISS-based semantic search on top of file-based storage
    """

    def __init__(self, storage_path: str = "./local_videodb",
                 collection_name: str = "battlefield_reconnaissance"):
        """
        Initialize local video database

        Args:
            storage_path: Path to local storage directory
            collection_name: Name of the collection to use
        """
        print(f"Initializing local video database at: {storage_path}")

        # Connect to local videodb
        self.conn = local_videodb.connect(storage_path=storage_path)
        self.collection_name = collection_name

        # Get or create collection
        try:
            self.collection = self.conn.get_collection(collection_name)
        except:
            # Collection doesn't exist, create it
            self.collection = self.conn.create_collection(
                name=collection_name,
                description="Battlefield reconnaissance drone footage analysis"
            )

        # Initialize per-video FAISS indices
        self.faiss_indices = {}  # Dict mapping video_id to (index, embedding_map)
        self.storage_path = storage_path

        # Cache for compatibility
        self.video_cache = {}

        print(f"Local video database initialized. Collection: {collection_name}")

    def _get_video_faiss_path(self, video_id: str) -> Path:
        """Get FAISS index path for a specific video"""
        video_path = Path(self.storage_path) / "collections" / self.collection_name / "videos" / video_id
        faiss_path = video_path / "faiss_index"
        return faiss_path

    def _load_faiss_index(self, video_id: str):
        """Load or create FAISS index for a specific video"""
        if faiss is None:
            print("Warning: FAISS not available, semantic search disabled")
            return None, []

        faiss_path = self._get_video_faiss_path(video_id)
        faiss_path.mkdir(exist_ok=True, parents=True)

        index_file = faiss_path / "embeddings.index"
        map_file = faiss_path / "embedding_map.json"

        if index_file.exists() and map_file.exists():
            # Load existing index
            try:
                faiss_index = faiss.read_index(str(index_file))
                with open(map_file, 'r') as f:
                    embedding_map = json.load(f)
                print(f"Loaded FAISS index for video {video_id} with {faiss_index.ntotal} embeddings")
                return faiss_index, embedding_map
            except Exception as e:
                print(f"Error loading FAISS index for {video_id}: {e}, creating new one")
                return faiss.IndexFlatIP(1024), []
        else:
            # Create new index
            print(f"Created new FAISS index for video {video_id}")
            return faiss.IndexFlatIP(1024), []

    def _get_faiss_index(self, video_id: str):
        """Get FAISS index for a video (loads if not in cache)"""
        if video_id not in self.faiss_indices:
            faiss_index, embedding_map = self._load_faiss_index(video_id)
            self.faiss_indices[video_id] = (faiss_index, embedding_map)
        return self.faiss_indices[video_id]

    def _save_faiss_index(self, video_id: str):
        """Save FAISS index for a specific video to disk"""
        if faiss is None:
            return

        if video_id not in self.faiss_indices:
            return

        try:
            faiss_index, embedding_map = self.faiss_indices[video_id]
            faiss_path = self._get_video_faiss_path(video_id)
            faiss_path.mkdir(exist_ok=True, parents=True)

            index_file = faiss_path / "embeddings.index"
            map_file = faiss_path / "embedding_map.json"

            faiss.write_index(faiss_index, str(index_file))

            with open(map_file, 'w') as f:
                json.dump(embedding_map, f, indent=2)

            print(f"Saved FAISS index for video {video_id}")

        except Exception as e:
            print(f"Error saving FAISS index for {video_id}: {e}")

    def _save_all_faiss_indices(self):
        """Save all loaded FAISS indices to disk"""
        for video_id in self.faiss_indices.keys():
            self._save_faiss_index(video_id)

    def upload_video(self, video_path: str, name: Optional[str] = None,
                    description: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload video to local storage

        Args:
            video_path: Path to video file
            name: Optional video name
            description: Optional description

        Returns:
            Dict with video_id and video info
        """
        try:
            print(f"Uploading video: {video_path}")

            # Upload using local_videodb
            video = self.collection.upload(
                source=video_path,
                name=name or Path(video_path).name,
                description=description or ""
            )

            # Cache video info
            video_info = {
                "video_id": video.id,
                "video": video,
                "name": video.name,
                "length": video.length,
                "stream_url": video.stream_url,
                "player_url": video.player_url
            }

            self.video_cache[video.id] = video_info

            print(f"Video uploaded successfully. ID: {video.id}, Length: {video.length:.1f}s")

            return video_info

        except Exception as e:
            print(f"Error uploading video: {e}")
            raise

    def create_segments(self, video_id: str, segment_duration: int = 30) -> List[Dict[str, Any]]:
        """
        Create time-based segments for a video

        Args:
            video_id: Video ID
            segment_duration: Duration of each segment in seconds

        Returns:
            List of segment metadata dicts
        """
        try:
            # Get video
            video = self.conn.get_video(video_id, self.collection_name)

            # Extract scenes using time-based segmentation
            scene_collection = video.extract_scenes(
                extraction_type=local_videodb.SceneExtractionType.time_based,
                extraction_config={
                    "time": segment_duration,
                    "frame_count": 1,
                    "select_frames": ["middle"]
                }
            )

            # Convert scenes to segments
            segments = []
            for i, scene in enumerate(scene_collection.scenes):
                segment = {
                    "segment_id": i,
                    "video_id": video_id,
                    "start_time": scene.start,
                    "end_time": scene.end,
                    "duration": scene.end - scene.start,
                    "timeline": [(scene.start, scene.end)],
                    "scene_id": scene.id
                }
                segments.append(segment)

            print(f"Created {len(segments)} segments for video {video_id}")

            # Cache segments
            if video_id not in self.video_cache:
                self.video_cache[video_id] = {}
            self.video_cache[video_id]["segments"] = segments
            self.video_cache[video_id]["scene_collection"] = scene_collection

            return segments

        except Exception as e:
            print(f"Error creating segments: {e}")
            raise

    def store_segment_metadata(self, video_id: str, segment_id: int, metadata: Dict[str, Any]):
        """
        Store metadata for a segment

        Args:
            video_id: Video ID
            segment_id: Segment ID
            metadata: Metadata to store
        """
        try:
            # Get video metadata path
            video = self.conn.get_video(video_id, self.collection_name)
            storage = self.conn.storage

            # Store segment metadata in video's custom data
            video_meta = storage.get_video_metadata(self.collection_name, video_id)
            if "segments_metadata" not in video_meta:
                video_meta["segments_metadata"] = {}

            # Store metadata (including tracking data if available)
            segment_meta = {
                "segment_id": segment_id,
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "objects": metadata.get("objects", []),
                "object_count": len(metadata.get("objects", [])),
                "event_description": metadata.get("event_description", "")
            }

            # Add tracking-related fields if present
            if metadata.get("tracking_enabled"):
                segment_meta["tracking_enabled"] = metadata.get("tracking_enabled", False)
                segment_meta["num_objects_tracked"] = metadata.get("num_objects_tracked", 0)
                segment_meta["num_frames_processed"] = metadata.get("num_frames_processed", 0)
            if metadata.get("tracking_video_path"):
                segment_meta["tracking_video_path"] = metadata.get("tracking_video_path")
            if metadata.get("tracking_summary"):
                segment_meta["tracking_summary"] = metadata.get("tracking_summary")

            video_meta["segments_metadata"][str(segment_id)] = segment_meta

            storage.save_video_metadata(self.collection_name, video_id, video_meta)

            # Store embeddings in FAISS (per-video index)
            if faiss is not None:
                # Get or create FAISS index for this video
                faiss_index, embedding_map = self._get_faiss_index(video_id)

                # Store segment embedding
                if "segment_embedding" in metadata:
                    self._store_embedding(
                        video_id, segment_id,
                        metadata["segment_embedding"],
                        "segment"
                    )

                # Store description embedding
                if "description_embedding" in metadata:
                    self._store_embedding(
                        video_id, segment_id,
                        metadata["description_embedding"],
                        "description"
                    )

                # Store object embeddings
                if "object_embeddings" in metadata:
                    for i, emb in enumerate(metadata["object_embeddings"]):
                        self._store_embedding(
                            video_id, segment_id, emb, f"object_{i}"
                        )

            # Cache
            if video_id not in self.video_cache:
                self.video_cache[video_id] = {}
            if "segment_metadata" not in self.video_cache[video_id]:
                self.video_cache[video_id]["segment_metadata"] = {}
            self.video_cache[video_id]["segment_metadata"][segment_id] = metadata

            print(f"Stored metadata for segment {segment_id} of video {video_id}")

        except Exception as e:
            print(f"Error storing segment metadata: {e}")
            raise

    def _store_embedding(self, video_id: str, segment_id: int,
                        embedding: np.ndarray, embedding_type: str):
        """Store embedding in per-video FAISS index"""
        if faiss is None:
            return

        # Get the video's FAISS index
        faiss_index, embedding_map = self._get_faiss_index(video_id)

        # Ensure correct shape and type
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)

        # Add to FAISS
        faiss_idx = faiss_index.ntotal
        faiss_index.add(embedding)

        # Store mapping
        embedding_map.append({
            "video_id": video_id,
            "segment_id": segment_id,
            "embedding_type": embedding_type,
            "faiss_index": faiss_idx
        })

        # Update cache
        self.faiss_indices[video_id] = (faiss_index, embedding_map)

        # Save periodically
        if faiss_index.ntotal % 10 == 0:
            self._save_faiss_index(video_id)

    def get_segment_metadata(self, video_id: str, segment_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific segment"""
        try:
            # Try cache first
            if video_id in self.video_cache:
                if "segment_metadata" in self.video_cache[video_id]:
                    cached = self.video_cache[video_id]["segment_metadata"].get(segment_id)
                    if cached:
                        return cached

            # Load from storage
            storage = self.conn.storage
            video_meta = storage.get_video_metadata(self.collection_name, video_id)

            if video_meta and "segments_metadata" in video_meta:
                return video_meta["segments_metadata"].get(str(segment_id))

            return None

        except Exception as e:
            print(f"Error getting segment metadata: {e}")
            return None

    def get_all_segments(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all segments for a video"""
        try:
            # Try cache first
            if video_id in self.video_cache and "segments" in self.video_cache[video_id]:
                return self.video_cache[video_id]["segments"]

            # Load from video metadata
            storage = self.conn.storage
            video_meta = storage.get_video_metadata(self.collection_name, video_id)

            if video_meta and "segments_metadata" in video_meta:
                segments = []
                for seg_id, seg_meta in video_meta["segments_metadata"].items():
                    segments.append(seg_meta)
                return segments

            return []

        except Exception as e:
            print(f"Error getting segments: {e}")
            return []

    def query_by_similarity(self, query_embedding: np.ndarray,
                          video_id: Optional[str] = None,
                          top_k: int = 5,
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Query segments by embedding similarity

        Args:
            query_embedding: Query embedding vector
            video_id: Optional video ID to filter (required for per-video index)
            top_k: Number of results
            threshold: Similarity threshold

        Returns:
            List of matching segments
        """
        if faiss is None:
            print("Warning: FAISS not available")
            return []

        if video_id is None:
            print("Warning: video_id is required for per-video FAISS index")
            return []

        # Get the video's FAISS index
        faiss_index, embedding_map = self._get_faiss_index(video_id)

        if faiss_index.ntotal == 0:
            print(f"Warning: FAISS index empty for video {video_id}")
            return []

        try:
            # Prepare query
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype(np.float32)

            # Search
            scores, indices = faiss_index.search(query_embedding, min(top_k * 3, faiss_index.ntotal))

            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(embedding_map):
                    continue

                mapping = embedding_map[idx]

                # Filter by threshold
                if score < threshold:
                    continue

                # Get segment metadata
                seg_meta = self.get_segment_metadata(mapping["video_id"], mapping["segment_id"])
                if seg_meta:
                    result = {
                        "video_id": mapping["video_id"],
                        "segment_id": mapping["segment_id"],
                        "similarity_score": float(score),
                        "start_time": seg_meta.get("start_time"),
                        "end_time": seg_meta.get("end_time"),
                        "event_description": seg_meta.get("event_description", ""),
                        "objects": seg_meta.get("objects", [])
                    }
                    results.append(result)

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def query_by_object_type(self, object_type: str,
                           video_id: Optional[str] = None,
                           min_count: int = 1) -> List[Dict[str, Any]]:
        """Query segments by object type"""
        try:
            results = []

            # Get videos to search
            if video_id:
                video_ids = [video_id]
            else:
                video_ids = self.conn.storage.list_videos(self.collection_name)

            # Search through videos
            for vid in video_ids:
                video_meta = self.conn.storage.get_video_metadata(self.collection_name, vid)
                if not video_meta or "segments_metadata" not in video_meta:
                    continue

                for seg_id, seg_meta in video_meta["segments_metadata"].items():
                    objects = seg_meta.get("objects", [])
                    count = sum(1 for obj in objects if obj.get("class") == object_type)

                    if count >= min_count:
                        results.append({
                            "video_id": vid,
                            "segment_id": int(seg_id),
                            "object_type": object_type,
                            "object_count": count,
                            "start_time": seg_meta.get("start_time"),
                            "end_time": seg_meta.get("end_time"),
                            "event_description": seg_meta.get("event_description", ""),
                            "objects": [obj for obj in objects if obj.get("class") == object_type]
                        })

            # Sort by count
            results.sort(key=lambda x: x["object_count"], reverse=True)

            return results

        except Exception as e:
            print(f"Error querying by object type: {e}")
            return []

    def query_by_event_description(self, query: str,
                                  video_id: Optional[str] = None,
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """Query by event description keyword"""
        try:
            results = []
            query_lower = query.lower()

            # Get videos to search
            if video_id:
                video_ids = [video_id]
            else:
                video_ids = self.conn.storage.list_videos(self.collection_name)

            # Search
            for vid in video_ids:
                video_meta = self.conn.storage.get_video_metadata(self.collection_name, vid)
                if not video_meta or "segments_metadata" not in video_meta:
                    continue

                for seg_id, seg_meta in video_meta["segments_metadata"].items():
                    description = seg_meta.get("event_description", "").lower()

                    if query_lower in description:
                        results.append({
                            "video_id": vid,
                            "segment_id": int(seg_id),
                            "start_time": seg_meta.get("start_time"),
                            "end_time": seg_meta.get("end_time"),
                            "event_description": seg_meta.get("event_description", ""),
                            "objects": seg_meta.get("objects", [])
                        })

            return results[:top_k]

        except Exception as e:
            print(f"Error querying by event description: {e}")
            return []

    def get_video_summary(self, video_id: str, target_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get video summary with object counts.

        Args:
            video_id: Video ID to get summary for
            target_classes: Optional list of target classes to count. If provided,
                           initializes counts for these classes (even if 0).
                           All detected object classes will be counted regardless.

        Returns:
            Dict with video summary including object_counts
        """
        try:
            video = self.conn.get_video(video_id, self.collection_name)
            video_meta = self.conn.storage.get_video_metadata(self.collection_name, video_id)

            # Initialize object counts - dynamically count all detected classes
            # If target_classes provided, initialize those classes to 0 first
            object_counts = {}
            if target_classes:
                for cls in target_classes:
                    object_counts[cls.lower()] = 0

            num_segments = 0

            if video_meta and "segments_metadata" in video_meta:
                num_segments = len(video_meta["segments_metadata"])
                for seg_meta in video_meta["segments_metadata"].values():
                    for obj in seg_meta.get("objects", []):
                        obj_class = obj.get("class", "unknown")
                        # Count all detected classes, not just predefined ones
                        object_counts[obj_class] = object_counts.get(obj_class, 0) + 1

            return {
                "video_id": video_id,
                "video_name": video.name,
                "video_length": video.length,
                "num_segments": num_segments,
                "object_counts": object_counts,
                "stream_url": video.stream_url,
                "player_url": video.player_url
            }

        except Exception as e:
            print(f"Error getting video summary: {e}")
            return {}

    def export_metadata(self, video_id: str, output_path: str):
        """Export metadata to JSON"""
        try:
            video_meta = self.conn.storage.get_video_metadata(self.collection_name, video_id)

            # Remove embeddings (stored in FAISS)
            if video_meta and "segments_metadata" in video_meta:
                for seg_meta in video_meta["segments_metadata"].values():
                    seg_meta.pop("segment_embedding", None)
                    seg_meta.pop("description_embedding", None)
                    seg_meta.pop("object_embeddings", None)

            with open(output_path, 'w') as f:
                json.dump(video_meta, f, indent=2)

            print(f"Metadata exported to {output_path}")

        except Exception as e:
            print(f"Error exporting metadata: {e}")

    def cleanup(self):
        """Cleanup resources and save all FAISS indices"""
        try:
            print("Cleaning up video database...")
            self._save_all_faiss_indices()
            self.conn.cleanup()
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
