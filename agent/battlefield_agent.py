"""
Battlefield Agent: CodeAgent setup for battlefield reconnaissance analysis
Integrates EXAONE model with custom tools for video analysis
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Add smolagents to path
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

from smolagents import CodeAgent, VLLMModel

# Import tools
from tools.pdf_rag_tool import pdf_rag_search, add_pdf_to_rag
from tools.videodb_query_tool import (
    get_selected_contexts,
    query_video_semantic,
    query_video_by_object,
    query_video_by_event,
    get_video_summary,
    get_segment_details,
    set_active_video,
    set_active_videos
)
from tools.wargame_query_tool import (
    get_tactical_situation,
    get_friendly_units,
    get_hostile_units,
    get_unit_details,
    get_unit_waypoints,
    get_units_by_type
)

from agent.model_loader import load_exaone_model


class BattlefieldReconnaissanceAgent:
    """
    Battlefield Reconnaissance Agent
    Analyzes drone footage and answers questions about battlefield situations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Battlefield Reconnaissance Agent

        Args:
            config_path: Path to configuration directory
        """
        # Load configurations
        if config_path is None:
            config_path = parent_dir / "config"
        else:
            config_path = Path(config_path)

        self.config = self._load_config(config_path)

        # Load EXAONE model
        print("Loading EXAONE model for agent...")
        self.model = load_exaone_model(self.config["agent_model"])

        # Prepare tools - ALL map operations via MCP
        self.tools = [
            # Context management (CALL FIRST)
            get_selected_contexts,
            # PDF RAG tools
            pdf_rag_search,
            add_pdf_to_rag,
            # VideoDB query tools
            query_video_semantic,
            query_video_by_object,
            query_video_by_event,
            get_video_summary,
            get_segment_details,
            set_active_video,
            set_active_videos,  # Multi-video support
            # Tactical map / wargame tools
            get_tactical_situation,
            get_friendly_units,
            get_hostile_units,
            get_unit_details,
            get_unit_waypoints,
            get_units_by_type
        ]

        # Load custom instructions for battlefield reconnaissance
        custom_instructions = self._load_custom_instructions(config_path)

        # Create CodeAgent
        print("Creating CodeAgent...")
        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            executor_type=self.config["code_agent"]["executor_type"],
            max_steps=self.config["code_agent"]["max_steps"],
            planning_interval=self.config["code_agent"]["planning_interval"],
            stream_outputs=self.config["code_agent"]["stream_outputs"],
            additional_authorized_imports=self.config["code_agent"]["additional_authorized_imports"],
            instructions=custom_instructions  # Add custom instructions
        )

        print("Battlefield Reconnaissance Agent initialized successfully")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration files"""
        try:
            configs = {}

            # Load models config
            with open(config_path / "models_config.yaml", 'r') as f:
                configs.update(yaml.safe_load(f))

            # Load agent config
            with open(config_path / "agent_config.yaml", 'r') as f:
                configs.update(yaml.safe_load(f))

            return configs

        except Exception as e:
            print(f"Error loading configs: {e}")
            raise

    def _load_custom_instructions(self, config_path: Path) -> str:
        """Load custom instructions for the agent"""
        try:
            instructions_file = config_path / "agent_custom_instructions.txt"
            if instructions_file.exists():
                with open(instructions_file, 'r', encoding='utf-8') as f:
                    instructions = f.read()
                print(f"Loaded custom instructions ({len(instructions)} chars)")
                return instructions
            else:
                print("No custom instructions file found, using default behavior")
                return ""

        except Exception as e:
            print(f"Error loading custom instructions: {e}")
            return ""

    def run(self, task: str, **kwargs) -> Any:
        """
        Run agent on a task

        Args:
            task: Task description
            **kwargs: Additional arguments for agent.run()

        Returns:
            Agent output
        """
        try:
            result = self.agent.run(task, **kwargs)
            return result

        except Exception as e:
            print(f"Error running agent: {e}")
            raise

    def get_agent(self) -> CodeAgent:
        """Get the underlying CodeAgent instance"""
        return self.agent

    def set_video_context(self, video_id: str):
        """
        Set the current video context for the agent

        Args:
            video_id: Video ID to use as context
        """
        from tools.videodb_query_tool import set_current_video_id
        set_current_video_id(video_id)
        print(f"Video context set to: {video_id}")

    def add_pdf_context(self, pdf_path: str):
        """
        Add PDF document to RAG system for context

        Args:
            pdf_path: Path to PDF file
        """
        from tools.pdf_rag_tool import get_rag_system
        rag_system = get_rag_system()
        num_chunks = rag_system.add_pdf(pdf_path)
        print(f"Added {num_chunks} chunks from {pdf_path} to RAG system")


def create_battlefield_agent(config_path: Optional[str] = None) -> BattlefieldReconnaissanceAgent:
    """
    Convenience function to create a Battlefield Reconnaissance Agent

    Args:
        config_path: Path to configuration directory

    Returns:
        BattlefieldReconnaissanceAgent instance
    """
    return BattlefieldReconnaissanceAgent(config_path=config_path)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_battlefield_agent()

    # Set video context (example)
    # agent.set_video_context("m-xxxxx")

    # Example query
    # result = agent.run("How many tanks were detected in the video?")
    # print(result)
