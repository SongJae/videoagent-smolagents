"""
VideoAnalysisSystem: Main orchestrator for video analysis pipeline
Coordinates video upload, segmentation, detection, embedding, and description
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm

from core_src.videodb_manager import VideoDBManager
from core_src.object_detection import ObjectDetectionProcessor
from core_src.embedding_generator import EmbeddingGenerator
from core_src.event_description import EventDescriptionGenerator


class VideoAnalysisSystem:
    """
    Main system for analyzing battlefield reconnaissance videos
    Orchestrates the entire pipeline from upload to storage
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 detector: Optional['ObjectDetectionProcessor'] = None,
                 embedding_generator: Optional['EmbeddingGenerator'] = None,
                 description_generator: Optional['EventDescriptionGenerator'] = None):
        """
        Initialize video analysis system

        Args:
            config_path: Path to configuration directory
            collection_name: Optional collection name (overrides config file)
            detector: Pre-loaded object detection processor (avoids reloading models)
            embedding_generator: Pre-loaded embedding generator (avoids reloading models)
            description_generator: Pre-loaded event description generator (avoids reloading models)
        """
        # Load configurations
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config"
        else:
            config_path = Path(config_path)

        self.config = self._load_configs(config_path)

        # Initialize components
        print("Initializing Video Analysis System...")

        # VideoDB Manager (using local storage)
        storage_path = self.config["videodb"]["storage"].get("local_path", "./data/local_videodb")

        # Use provided collection_name or fall back to config
        target_collection = collection_name or self.config["videodb"]["connection"]["collection_name"]
        print(f"Initializing local video database at: {storage_path}")

        self.videodb_manager = VideoDBManager(
            storage_path=storage_path,
            collection_name=target_collection
        )

        # Track whether we own the models (for cleanup purposes)
        # If models are pre-loaded (shared), we don't cleanup on destruction
        self._owns_models = (detector is None and embedding_generator is None and description_generator is None)

        # Use pre-loaded models if provided, otherwise create new instances
        if detector is not None:
            print("✓ Using pre-loaded object detector (SAM3)")
            self.detector = detector
        else:
            print("⚠️  Loading new object detector instance (SAM3)")
            obj_det_config = self.config["models"]["object_detection"]
            self.detector = ObjectDetectionProcessor(
                device=obj_det_config.get("device", "cuda"),
                target_classes=obj_det_config.get("target_classes"),
                sam3_config=obj_det_config.get("sam3_config"),
                gpus_to_use=obj_det_config.get("gpus_to_use")
            )

        if embedding_generator is not None:
            print("✓ Using pre-loaded embedding generator")
            self.embedding_generator = embedding_generator
        else:
            print("⚠️  Loading new embedding generator instance")
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.config["models"]["embedding_model"]["model_name"],
                device=self.config["models"]["embedding_model"]["device"],
                normalize=self.config["models"]["embedding_model"]["normalize"],
                batch_size=self.config["models"]["embedding_model"]["batch_size"]
            )

        if description_generator is not None:
            print("✓ Using pre-loaded description generator")
            self.description_generator = description_generator
        else:
            print("⚠️  Loading new description generator instance")
            self.description_generator = EventDescriptionGenerator(
                model_name=self.config["models"]["vlm_model"]["model_name"],
                device=self.config["models"]["vlm_model"]["device"],
                max_tokens=self.config["models"]["vlm_model"]["max_tokens"],
                temperature=self.config["models"]["vlm_model"]["temperature"],
                batch_size=self.config["models"]["vlm_model"]["batch_size"]
            )

        print("Video Analysis System initialized successfully")

    def _load_configs(self, config_path: Path) -> Dict[str, Any]:
        """Load all configuration files"""
        try:
            configs = {}

            # Load models config
            with open(config_path / "models_config.yaml", 'r') as f:
                configs["models"] = yaml.safe_load(f)

            # Load videodb config
            with open(config_path / "videodb_config.yaml", 'r') as f:
                configs["videodb"] = yaml.safe_load(f)

            # Load agent config
            with open(config_path / "agent_config.yaml", 'r') as f:
                configs["agent"] = yaml.safe_load(f)

            return configs

        except Exception as e:
            print(f"Error loading configs: {e}")
            raise

    def analyze_video(self, video_path: str,
                     segment_duration: Optional[int] = None,
                     video_name: Optional[str] = None,
                     video_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete video analysis pipeline

        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            video_name: Optional video name
            video_description: Optional video description

        Returns:
            Dict with video_id and analysis results
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting video analysis: {video_path}")
            print(f"{'='*60}\n")

            # Step 1: Upload video to VideoDB
            print("Step 1: Uploading video to VideoDB...")
            video_info = self.videodb_manager.upload_video(
                video_path=video_path,
                name=video_name,
                description=video_description
            )
            video_id = video_info["video_id"]
            print(f"✓ Video uploaded. ID: {video_id}\n")

            # Step 2: Create segments
            print("Step 2: Creating video segments...")
            if segment_duration is None:
                segment_duration = self.config["videodb"]["video_processing"]["default_segment_duration"]

            segments = self.videodb_manager.create_segments(
                video_id=video_id,
                segment_duration=segment_duration
            )
            print(f"✓ Created {len(segments)} segments\n")

            # Step 3: Process each segment
            print("Step 3: Processing segments...")
            results = []

            for segment in tqdm(segments, desc="Processing segments"):
                segment_result = self._process_segment(
                    video_path=video_path,
                    video_id=video_id,
                    segment=segment
                )
                results.append(segment_result)

            # Step 4: Generate video summary
            print("\nStep 4: Generating video summary...")
            # Pass target classes to ensure proper object counting
            target_classes = self.detector.get_target_classes()
            summary = self.videodb_manager.get_video_summary(video_id, target_classes=target_classes)

            print(f"\n{'='*60}")
            print("Video analysis completed successfully!")
            print(f"{'='*60}\n")

            print(f"Summary:")
            print(f"  - Video ID: {video_id}")
            print(f"  - Total segments: {len(segments)}")
            # Dynamically print object counts for all detected classes
            for obj_class, count in summary['object_counts'].items():
                print(f"  - Total {obj_class}s detected: {count}")
            print(f"  - Stream URL: {summary['stream_url']}")

            return {
                "video_id": video_id,
                "video_info": video_info,
                "segments": segments,
                "segment_results": results,
                "summary": summary
            }

        except Exception as e:
            print(f"Error analyzing video: {e}")
            raise

    def _process_segment(self, video_path: str,
                        video_id: str,
                        segment: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single segment with optional SAM3 tracking"""
        try:
            segment_id = segment["segment_id"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]

            # Use SAM3 tracking if available, otherwise use simple detection
            tracking_result = None
            if self.detector.tracking_enabled:
                # Get current collection name from videodb_manager
                collection_name = self.videodb_manager.collection._collection_id

                # Process with SAM3 tracking and auto-save to local_videodb
                tracking_result = self.detector.process_segment_with_tracking(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    video_id=video_id,
                    segment_id=segment_id,
                    save_tracking_video=True,
                    collection_name=collection_name
                )

                # Check if tracking was successful
                if "error" in tracking_result:
                    print(f"Warning: Tracking failed for segment {segment_id}, using simple detection")
                    detections, middle_frame = self.detector.process_segment_simple(
                        video_path=video_path,
                        start_time=start_time,
                        end_time=end_time
                    )
                    tracking_result = None  # Clear failed tracking result
                else:
                    # Tracking successful - get detections from middle frame for metadata/embeddings
                    # (Tracking video already saved, but we need detection data for statistics)
                    detections, middle_frame = self.detector.process_segment_simple(
                        video_path=video_path,
                        start_time=start_time,
                        end_time=end_time
                    )
                    print(f"✓ Segment {segment_id} processed with SAM3 tracking")
                    print(f"  Objects detected: {len(detections)}")
                    if "tracking_video_path" in tracking_result:
                        print(f"  Tracking video: {tracking_result['tracking_video_path']}")
            else:
                # Use simple detection (middle frame only)
                detections, middle_frame = self.detector.process_segment_simple(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=end_time
                )

            if middle_frame is None:
                print(f"Warning: No frame extracted for segment {segment_id}")
                return {"segment_id": segment_id, "error": "No frame extracted"}

            # 2. Generate segment embedding
            segment_embedding = self.embedding_generator.encode_image(middle_frame)

            # 3. Extract object crops and generate embeddings
            object_embeddings = []
            if len(detections) > 0:
                crops = self.detector.extract_object_crops(middle_frame, detections)
                for crop, det in crops:
                    obj_embedding = self.embedding_generator.encode_image(crop)
                    object_embeddings.append(obj_embedding)

            # 4. Generate event description
            context = {
                "timestamp": f"{start_time:.1f}s - {end_time:.1f}s",
                "objects": detections
            }
            event_description = self.description_generator.describe_with_context(
                image=middle_frame,
                context=context
            )

            # 5. Generate description embedding
            description_embedding = self.embedding_generator.encode_text(event_description)

            # 6. Store all metadata
            metadata = {
                "segment_id": segment_id,
                "start_time": start_time,
                "end_time": end_time,
                "objects": detections,
                "object_count": len(detections),
                "segment_embedding": segment_embedding,
                "object_embeddings": object_embeddings,
                "event_description": event_description,
                "description_embedding": description_embedding
            }

            # Add tracking data if available
            if tracking_result:
                metadata["tracking_enabled"] = True
                metadata["num_objects_tracked"] = tracking_result.get("num_objects_tracked", 0)
                metadata["num_frames_processed"] = tracking_result.get("num_frames_processed", 0)
                if "tracking_video_path" in tracking_result:
                    metadata["tracking_video_path"] = tracking_result["tracking_video_path"]
                if "tracking_data" in tracking_result:
                    # Store summarized tracking data (not all frames to save space)
                    metadata["tracking_summary"] = {
                        "total_frames": len(tracking_result["tracking_data"]),
                        "avg_objects_per_frame": sum(
                            frame.get("num_objects", 0) for frame in tracking_result["tracking_data"]
                        ) / len(tracking_result["tracking_data"]) if tracking_result["tracking_data"] else 0
                    }

            self.videodb_manager.store_segment_metadata(
                video_id=video_id,
                segment_id=segment_id,
                metadata=metadata
            )

            return {
                "segment_id": segment_id,
                "object_count": len(detections),
                "event_description": event_description
            }

        except Exception as e:
            print(f"Error processing segment {segment.get('segment_id', '?')}: {e}")
            return {
                "segment_id": segment.get("segment_id"),
                "error": str(e)
            }

    def query_video(self, video_id: str, query: str,
                   query_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        Query video segments

        Args:
            video_id: Video ID
            query: Query text
            query_type: Type of query (semantic/object/event)

        Returns:
            List of matching segments
        """
        try:
            if query_type == "semantic":
                # Generate query embedding
                query_embedding = self.embedding_generator.encode_text(query)

                # Search by similarity
                results = self.videodb_manager.query_by_similarity(
                    query_embedding=query_embedding,
                    video_id=video_id,
                    top_k=5,
                    threshold=0.7
                )

            elif query_type == "object":
                # Query by object type
                # Extract object type from query (simple keyword matching)
                object_types = ["truck", "tank", "soldier"]
                object_type = None
                for obj in object_types:
                    if obj in query.lower():
                        object_type = obj
                        break

                if object_type:
                    results = self.videodb_manager.query_by_object_type(
                        object_type=object_type,
                        video_id=video_id
                    )
                else:
                    results = []

            elif query_type == "event":
                # Query by event description
                results = self.videodb_manager.query_by_event_description(
                    query=query,
                    video_id=video_id,
                    top_k=5
                )

            else:
                results = []

            return results

        except Exception as e:
            print(f"Error querying video: {e}")
            return []

    def export_results(self, video_id: str, output_path: str):
        """Export all analysis results to JSON"""
        try:
            self.videodb_manager.export_metadata(video_id, output_path)
            print(f"Results exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting results: {e}")

    def get_summary(self, video_id: str) -> Dict[str, Any]:
        """Get video summary"""
        target_classes = self.detector.get_target_classes()
        return self.videodb_manager.get_video_summary(video_id, target_classes=target_classes)

    def get_target_classes(self) -> List[str]:
        """
        Get the list of target classes configured for object detection.

        Returns:
            List of class names (e.g., ["soldier", "tank", "truck"])
        """
        return self.detector.get_target_classes()

    def switch_collection(self, collection_name: str):
        """
        Switch to a different collection without reloading models

        Args:
            collection_name: Name of the collection to switch to
        """
        print(f"Switching to collection: {collection_name}")

        # Clean up current videodb manager
        if hasattr(self.videodb_manager, 'cleanup'):
            self.videodb_manager.cleanup()

        # Create new videodb manager for the new collection
        storage_path = self.config["videodb"]["storage"].get("local_path", "./data/local_videodb")
        self.videodb_manager = VideoDBManager(
            storage_path=storage_path,
            collection_name=collection_name
        )

        print(f"✓ Switched to collection: {collection_name}")

    def cleanup(self):
        """Cleanup all resources"""
        try:
            print("Cleaning up resources...")

            # Only cleanup models if we own them (not shared from model_manager)
            if self._owns_models:
                print("Cleaning up owned model instances...")
                self.detector.cleanup()
                self.embedding_generator.cleanup()
                self.description_generator.cleanup()
            else:
                print("Skipping model cleanup (using shared model instances)")

            # Always cleanup videodb manager
            if hasattr(self.videodb_manager, 'cleanup'):
                self.videodb_manager.cleanup()

            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


# Convenience function
def create_video_analysis_system(config_path: Optional[str] = None) -> VideoAnalysisSystem:
    """
    Create and initialize VideoAnalysisSystem

    Args:
        config_path: Path to configuration directory

    Returns:
        Initialized VideoAnalysisSystem
    """
    return VideoAnalysisSystem(config_path=config_path)
