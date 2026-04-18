"""
Model Loader: Load and configure EXAONE-4.0-32B-AWQ for agent use
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Add smolagents to path
parent_dir = Path(__file__).parent.parent
smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

from smolagents import VLLMModel


def load_exaone_model(config: Optional[Dict[str, Any]] = None) -> VLLMModel:
    """
    Load EXAONE-4.0-32B-AWQ model for agent use

    Args:
        config: Optional configuration dict for model parameters

    Returns:
        VLLMModel instance
    """
    # Load default config if not provided
    if config is None:
        config_path = parent_dir / "config" / "models_config.yaml"
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config["agent_model"]

    print(f"Loading EXAONE model: {config['model_id']}")

    # Configure VLLM model
    # NOTE: temperature and max_tokens should NOT be passed to VLLMModel.__init__()
    # They are extracted from kwargs in generate() method and passed to SamplingParams
    # Passing them as kwargs causes them to be added to completion_kwargs and then
    # passed to LLM.generate() which doesn't accept them, causing errors
    model = VLLMModel(
        model_id=config["model_id"],
        model_kwargs={
            "tensor_parallel_size": config.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.9),
            "dtype": config.get("dtype", "auto"),
            "quantization": config.get("quantization", "awq"),
            "max_model_len": config.get("max_model_len", 4096),
            "trust_remote_code": True,
        },
        # Do NOT pass generation parameters here - they should be passed when calling generate()
        # VLLMModel.generate() extracts them from kwargs and creates SamplingParams
        # Default values in generate() are: temperature=0.0, max_tokens=2048, n=1
    )

    print("EXAONE model loaded successfully")

    return model


def load_model_from_config_file(config_path: str) -> VLLMModel:
    """
    Load model from configuration file

    Args:
        config_path: Path to models_config.yaml

    Returns:
        VLLMModel instance
    """
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        config = full_config["agent_model"]

    return load_exaone_model(config)
