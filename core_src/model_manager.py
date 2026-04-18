"""
ModelManager: Centralized model loading and management
Pre-loads all models before UI or analysis starts
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from tqdm import tqdm
import torch

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class ModelManager:
    """
    Centralized manager for all models in the system
    Handles pre-loading, caching, and cleanup
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Model Manager

        Args:
            config_path: Path to configuration directory
        """
        if config_path is None:
            config_path = parent_dir / "config"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self.config = self._load_configs()

        # Model instances
        self.object_detector = None
        self.embedding_generator = None
        self.description_generator = None
        self.agent_model = None
        self.video_system = None
        self.agent = None

        # Loading status
        self.models_loaded = False
        self.loading_errors = []

    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        try:
            configs = {}

            # Load models config
            with open(self.config_path / "models_config.yaml", 'r') as f:
                configs["models"] = yaml.safe_load(f)

            # Load videodb config
            with open(self.config_path / "videodb_config.yaml", 'r') as f:
                configs["videodb"] = yaml.safe_load(f)

            # Load agent config
            with open(self.config_path / "agent_config.yaml", 'r') as f:
                configs["agent"] = yaml.safe_load(f)

            return configs

        except Exception as e:
            print(f"Error loading configs: {e}")
            raise

    def load_all_models(self, load_agent: bool = True, progress_callback=None):
        """
        Load all models with progress tracking

        Args:
            load_agent: Whether to load the agent model (optional, as it's large)
            progress_callback: Optional callback function for progress updates
        """
        print("\n" + "="*60)
        print("Loading Models...")
        print("="*60 + "\n")

        models_to_load = [
            ("Object Detection & Tracking (SAM3)", self._load_object_detector),
            ("Embedding Model (PE-Core)", self._load_embedding_generator),
            ("Event Description (SmolVLM2)", self._load_description_generator),
        ]

        if load_agent:
            models_to_load.append(("Agent Model (EXAONE)", self._load_agent_model))

        # Load models with progress bar
        with tqdm(total=len(models_to_load), desc="Loading models") as pbar:
            for model_name, load_func in models_to_load:
                try:
                    print(f"\n📦 Loading {model_name}...")
                    if progress_callback:
                        progress_callback(f"Loading {model_name}...")

                    load_func()

                    print(f"✅ {model_name} loaded successfully")
                    pbar.update(1)

                except Exception as e:
                    error_msg = f"❌ Error loading {model_name}: {e}"
                    print(error_msg)
                    self.loading_errors.append((model_name, str(e)))
                    pbar.update(1)

        if self.loading_errors:
            print("\n⚠️  Some models failed to load:")
            for model_name, error in self.loading_errors:
                print(f"  - {model_name}: {error}")
        else:
            print("\n✅ All models loaded successfully!")
            self.models_loaded = True

        print("\n" + "="*60 + "\n")

    def _load_object_detector(self):
        """Load SAM3-based object detection and tracking model"""
        from core_src.object_detection import ObjectDetectionProcessor

        obj_det_config = self.config["models"]["object_detection"]

        self.object_detector = ObjectDetectionProcessor(
            device=obj_det_config.get("device", "cuda"),
            target_classes=obj_det_config.get("target_classes"),
            sam3_config=obj_det_config.get("sam3_config"),
            gpus_to_use=obj_det_config.get("gpus_to_use"),
            confidence_threshold=obj_det_config.get("confidence_threshold", 0.3)
        )

    def _load_embedding_generator(self):
        """Load embedding model"""
        from core_src.embedding_generator import EmbeddingGenerator

        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config["models"]["embedding_model"]["model_name"],
            device=self.config["models"]["embedding_model"]["device"],
            normalize=self.config["models"]["embedding_model"]["normalize"],
            batch_size=self.config["models"]["embedding_model"]["batch_size"]
        )

    def _load_description_generator(self):
        """Load event description model"""
        from core_src.event_description import EventDescriptionGenerator

        self.description_generator = EventDescriptionGenerator(
            model_name=self.config["models"]["vlm_model"]["model_name"],
            device=self.config["models"]["vlm_model"]["device"],
            max_tokens=self.config["models"]["vlm_model"]["max_tokens"],
            temperature=self.config["models"]["vlm_model"]["temperature"],
            batch_size=self.config["models"]["vlm_model"]["batch_size"]
        )

    def _load_agent_model(self):
        """Load agent model (EXAONE)"""
        from agent.model_loader import load_exaone_model

        self.agent_model = load_exaone_model(self.config["models"]["agent_model"])

    def get_video_system(self):
        """
        Get or create VideoAnalysisSystem with pre-loaded models

        Returns:
            VideoAnalysisSystem instance with pre-loaded models
        """
        if self.video_system is None:
            from core_src.video_analysis_system import VideoAnalysisSystem

            # Create system but replace its models with pre-loaded ones
            self.video_system = VideoAnalysisSystem.__new__(VideoAnalysisSystem)
            self.video_system.config = self.config

            # Initialize VideoDB Manager
            from core_src.videodb_manager import VideoDBManager
            storage_path = self.config["videodb"]["storage"].get("local_path", "./data/local_videodb")
            self.video_system.videodb_manager = VideoDBManager(
                storage_path=storage_path,
                collection_name=self.config["videodb"]["connection"]["collection_name"]
            )

            # Use pre-loaded models
            self.video_system.detector = self.object_detector
            self.video_system.embedding_generator = self.embedding_generator
            self.video_system.description_generator = self.description_generator

            print("VideoAnalysisSystem initialized with pre-loaded models")

        return self.video_system

    def get_agent(self):
        """
        Get or create BattlefieldReconnaissanceAgent with pre-loaded model

        Returns:
            BattlefieldReconnaissanceAgent instance with pre-loaded model
        """
        if self.agent is None:
            from agent.battlefield_agent import BattlefieldReconnaissanceAgent
            from smolagents import CodeAgent

            # Import skill-based tools (dynamically routed via SkillManager)
            from skills.skill_tool import (
                get_available_skills,
                get_skill_actions,
                invoke_skill,
                videodb_query_skill,
                pdf_rag_skill,
                wargame_query_skill,
                final_response_skill
            )

            # Create agent manually using pre-loaded model
            self.agent = BattlefieldReconnaissanceAgent.__new__(BattlefieldReconnaissanceAgent)
            self.agent.config_path = self.config_path
            self.agent.config = self.config

            # Use pre-loaded model
            self.agent.model = self.agent_model

            # Prepare skill-based tools
            # All skills are dynamically discovered and executed via SkillManager
            # Adding new skills only requires creating a skill directory with SKILL.md and executor.py
            self.agent.tools = [
                # Skill discovery (dynamic - auto-discovers new skills)
                get_available_skills,
                get_skill_actions,
                # Generic skill invocation (dynamic - routes to any skill)
                invoke_skill,
                # Direct skill tools (convenience wrappers - optional)
                videodb_query_skill,    # Video database queries
                pdf_rag_skill,          # PDF document search
                wargame_query_skill,    # Tactical map queries
                final_response_skill    # Report generation
            ]

            # Load custom instructions
            custom_instructions = ""
            instructions_file = self.config_path / "agent_custom_instructions.txt"
            if instructions_file.exists():
                try:
                    with open(instructions_file, 'r', encoding='utf-8') as f:
                        custom_instructions = f.read()
                    print(f"Loaded custom instructions ({len(custom_instructions)} chars)")
                except Exception as e:
                    print(f"Warning: Could not load custom instructions: {e}")

            # Create CodeAgent
            self.agent.agent = CodeAgent(
                model=self.agent.model,
                tools=self.agent.tools,
                executor_type=self.config["agent"]["code_agent"]["executor_type"],
                max_steps=self.config["agent"]["code_agent"]["max_steps"],
                planning_interval=self.config["agent"]["code_agent"]["planning_interval"],
                stream_outputs=self.config["agent"]["code_agent"]["stream_outputs"],
                additional_authorized_imports=self.config["agent"]["code_agent"]["additional_authorized_imports"],
                instructions=custom_instructions
            )

            print("BattlefieldReconnaissanceAgent initialized with pre-loaded model")

        return self.agent

    def check_models_ready(self) -> bool:
        """
        Check if all required models are loaded

        Returns:
            True if models are ready, False otherwise
        """
        return (
            self.object_detector is not None and
            self.embedding_generator is not None and
            self.description_generator is not None
        )

    def check_agent_ready(self) -> bool:
        """
        Check if agent is ready

        Returns:
            True if agent is ready, False otherwise
        """
        return self.agent_model is not None

    def cleanup(self):
        """Cleanup all model resources"""
        print("Cleaning up models...")

        try:
            if self.object_detector:
                self.object_detector.cleanup()

            if self.embedding_generator:
                self.embedding_generator.cleanup()

            if self.description_generator:
                self.description_generator.cleanup()

            if self.video_system:
                self.video_system.cleanup()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("✅ Model cleanup completed")

        except Exception as e:
            print(f"Error during cleanup: {e}")


# Global model manager instance (singleton pattern)
_global_model_manager = None


def get_global_model_manager(config_path: Optional[str] = None) -> ModelManager:
    """
    Get or create global model manager instance

    Args:
        config_path: Path to configuration directory

    Returns:
        Global ModelManager instance
    """
    global _global_model_manager

    if _global_model_manager is None:
        _global_model_manager = ModelManager(config_path=config_path)

    return _global_model_manager


def reset_global_model_manager():
    """Reset global model manager (for testing)"""
    global _global_model_manager
    if _global_model_manager:
        _global_model_manager.cleanup()
    _global_model_manager = None
