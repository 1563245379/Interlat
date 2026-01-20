"""
Configuration file for Math Evaluator
=====================================

This file contains default configurations and settings for the math evaluator.
You can modify these values or create custom configurations.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EvaluationConfig:
    """Configuration class for evaluation settings"""

    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"

    # Generation settings
    max_length: int = 2048
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Dataset settings
    dataset_name: str = "hendrycks/MATH"
    split: str = "test"
    num_samples: Optional[int] = None
    samples_per_question: int = 1

    # Evaluation settings
    output_dir: str = "./results"
    save_detailed_results: bool = True
    save_summary: bool = True
    log_level: str = "INFO"

    # Advanced settings
    use_cache: bool = True
    seed: int = 42
    batch_size: int = 1  # Currently only supports 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    def save(self, filepath: str):
        """Save config to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'EvaluationConfig':
        """Load config from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configurations for different use cases
DEFAULT_CONFIG = EvaluationConfig()

FAST_CONFIG = EvaluationConfig(
    max_length=512,
    max_new_tokens=256,
    num_samples=100,
    samples_per_question=1
)

HIGH_QUALITY_CONFIG = EvaluationConfig(
    temperature=0.0,
    do_sample=False,
    max_length=4096,
    max_new_tokens=2048,
    samples_per_question=3
)

DEBUG_CONFIG = EvaluationConfig(
    num_samples=5,
    samples_per_question=1,
    log_level="DEBUG"
)

# Available model presets
# Available model presets
MODEL_PRESETS = {
    "qwen2.5-7b": {
        "model_name": "Qwen/Qwen2.5-7B",
        "tokenizer_name": "Qwen/Qwen2.5-7B",
        "max_length": 4096,
        "max_new_tokens": 2048,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "use_cache": True
    },
    "qwen2.5-0.5b": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "tokenizer_name": "Qwen/Qwen2.5-0.5B",
        "max_length": 2048,
        "max_new_tokens": 1024,
        "torch_dtype": "float16",
        "device": "auto",
        "use_cache": True
    },
    "llama3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "tokenizer_name": "meta-llama/Llama-3.1-8B",
        "max_length": 4096,
        "max_new_tokens": 2048,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "use_cache": True
    }
}

# Dataset presets
DATASET_PRESETS = {
    "math": {
        "dataset_name": "hendrycks/MATH",
        "split": "test"
    },
}

def get_config(config_name: str = "default") -> EvaluationConfig:
    """Get predefined configuration by name"""
    configs = {
        "default": DEFAULT_CONFIG,
        "fast": FAST_CONFIG,
        "high_quality": HIGH_QUALITY_CONFIG,
        "debug": DEBUG_CONFIG
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]

def create_custom_config(**kwargs) -> EvaluationConfig:
    """Create custom configuration with specified parameters"""
    config = EvaluationConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return config