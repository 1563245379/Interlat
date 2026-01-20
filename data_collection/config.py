#!/usr/bin/env python3
"""
Configuration Management for Data Collection
===========================================

Centralized configuration management for different data collection scenarios.
Provides pre-defined configurations and easy customization options.

Usage:
    from config import get_config, DataCollectionConfig

    # Use predefined config
    config = get_config("math_high_quality")

    # Create custom config
    config = DataCollectionConfig.create_custom(
        temperature=0.8,
        max_new_tokens=2000
    )
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-related configuration"""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "float32"  # "float32", "float16", "bfloat16"
    trust_remote_code: bool = True
    device_strategy: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class GenerationConfig:
    """Text generation configuration"""
    temperature: float = 0.8
    max_new_tokens: int = 1500
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_beams: int = 1
    max_hidden_states: int = 10000


@dataclass
class DatasetConfig:
    """Dataset-specific configuration"""
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    dataset_split: str = "train"  # "train", "test", "validation"


@dataclass
class StorageConfig:
    """Data storage configuration"""
    output_dir: str = "collected_data"
    max_shard_size_gb: float = 2.0
    compression: str = "zstd"  # "zstd", "gzip", "snappy"
    save_huggingface: bool = True
    save_parquet: bool = True


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    backend: str = "nccl"  # "nccl", "gloo", "mpi"
    timeout_minutes: int = 300
    init_method: str = "env://"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_to_file: bool = True
    log_file_prefix: str = "data_collection"
    progress_update_frequency: int = 10


@dataclass
class DataCollectionConfig:
    """Complete data collection configuration"""
    model: ModelConfig
    generation: GenerationConfig
    dataset: DatasetConfig
    storage: StorageConfig
    distributed: DistributedConfig
    logging: LoggingConfig

    # Task-specific settings
    task_type: str = "general"  # "math", "alfworld", "general"

    @classmethod
    def create_default(cls) -> 'DataCollectionConfig':
        """Create default configuration"""
        return cls(
            model=ModelConfig(),
            generation=GenerationConfig(),
            dataset=DatasetConfig(),
            storage=StorageConfig(),
            distributed=DistributedConfig(),
            logging=LoggingConfig()
        )

    @classmethod
    def create_custom(cls, **kwargs) -> 'DataCollectionConfig':
        """Create custom configuration with overrides"""
        config = cls.create_default()

        # Apply overrides
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like "generation.temperature"
                section, param = key.split('.', 1)
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, value)
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'DataCollectionConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            model=ModelConfig(**data['model']),
            generation=GenerationConfig(**data['generation']),
            dataset=DatasetConfig(**data['dataset']),
            storage=StorageConfig(**data['storage']),
            distributed=DistributedConfig(**data['distributed']),
            logging=LoggingConfig(**data['logging']),
            task_type=data.get('task_type', 'general')
        )


# Predefined configurations for different scenarios
PREDEFINED_CONFIGS = {
    "default": {
        "description": "Standard configuration for general data collection",
        "config": DataCollectionConfig.create_default()
    },

    "math_high_quality": {
        "description": "High quality configuration for mathematical reasoning",
        "config": DataCollectionConfig.create_custom(
            **{
                "task_type": "math",
                "generation.temperature": 0.7,
                "generation.max_new_tokens": 2000,
                "generation.top_p": 0.9,
                "storage.output_dir": "math_reasoning_high_quality",
                "dataset.dataset_split": "train"
            }
        )
    },

    "math_diverse": {
        "description": "More diverse sampling for mathematical reasoning",
        "config": DataCollectionConfig.create_custom(
            **{
                "task_type": "math",
                "generation.temperature": 1.0,
                "generation.max_new_tokens": 1500,
                "generation.top_p": 0.95,
                "storage.output_dir": "math_reasoning_diverse",
                "dataset.dataset_split": "train"
            }
        )
    },

    "alfworld_creative": {
        "description": "Creative planning for ALFWorld household tasks",
        "config": DataCollectionConfig.create_custom(
            **{
                "task_type": "alfworld",
                "generation.temperature": 1.2,
                "generation.max_new_tokens": 1500,
                "generation.top_p": 0.9,
                "storage.output_dir": "alfworld_creative_planning",
                "batch_size": 1
            }
        )
    },

    "fast_prototyping": {
        "description": "Fast configuration for prototyping and testing",
        "config": DataCollectionConfig.create_custom(
            **{
                "model.model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "generation.max_new_tokens": 512,
                "storage.max_shard_size_gb": 0.5,
                "storage.output_dir": "prototype_data",
                "logging.log_level": "DEBUG"
            }
        )
    },

    "large_scale": {
        "description": "Configuration for large-scale distributed collection",
        "config": DataCollectionConfig.create_custom(
            **{
                "generation.max_new_tokens": 2000,
                "storage.max_shard_size_gb": 5.0,
                "storage.output_dir": "large_scale_collection",
                "distributed.timeout_minutes": 600,
                "dataset.num_workers": 8
            }
        )
    },

    "cpu_only": {
        "description": "CPU-only configuration for environments without GPUs",
        "config": DataCollectionConfig.create_custom(
            **{
                "model.model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "model.device_strategy": "cpu",
                "model.torch_dtype": "float32",
                "generation.max_new_tokens": 800,
                "distributed.backend": "gloo",
                "storage.output_dir": "cpu_collection"
            }
        )
    }
}


def get_config(config_name: str = "default") -> DataCollectionConfig:
    """
    Get a predefined configuration by name

    Args:
        config_name: Name of the predefined configuration

    Returns:
        DataCollectionConfig object

    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in PREDEFINED_CONFIGS:
        available = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return PREDEFINED_CONFIGS[config_name]["config"]


def list_available_configs() -> Dict[str, str]:
    """
    List all available predefined configurations

    Returns:
        Dictionary mapping config names to descriptions
    """
    return {
        name: info["description"]
        for name, info in PREDEFINED_CONFIGS.items()
    }


def create_config_for_model(model_path: str, task_type: str = "general") -> DataCollectionConfig:
    """
    Create configuration optimized for a specific model

    Args:
        model_path: Path to the model
        task_type: Type of task ("math", "alfworld", "general")

    Returns:
        Optimized DataCollectionConfig
    """
    # Model-specific optimizations
    model_optimizations = {}

    model_lower = model_path.lower()
    if "0.5b" in model_lower or "small" in model_lower:
        model_optimizations.update({
            "generation.max_new_tokens": 800,
            "storage.max_shard_size_gb": 1.0
        })
    elif "7b" in model_lower or "8b" in model_lower:
        model_optimizations.update({
            "generation.max_new_tokens": 1500,
            "storage.max_shard_size_gb": 3.0
        })
    elif "13b" in model_lower or "14b" in model_lower or "larger" in model_lower:
        model_optimizations.update({
            "generation.max_new_tokens": 2000,
            "storage.max_shard_size_gb": 5.0
        })

    # Task-specific optimizations
    if task_type == "math":
        model_optimizations.update({
            "generation.temperature": 0.7,
            "generation.top_p": 0.9
        })
    elif task_type == "alfworld":
        model_optimizations.update({
            "generation.temperature": 1.2,
            "generation.top_p": 0.95
        })

    # Set model path and task type
    model_optimizations.update({
        "model.model_path": model_path,
        "task_type": task_type,
        "storage.output_dir": f"{task_type}_data_{Path(model_path).name.replace('/', '_')}"
    })

    return DataCollectionConfig.create_custom(**model_optimizations)


# Utility functions for backwards compatibility
def get_math_config(temperature: float = 0.7,
                   max_new_tokens: int = 1500) -> DataCollectionConfig:
    """Get configuration for math data collection"""
    return DataCollectionConfig.create_custom(
        task_type="math",
        **{
            "generation.temperature": temperature,
            "generation.max_new_tokens": max_new_tokens,
            "storage.output_dir": f"math_data_temp_{temperature}"
        }
    )


def get_alfworld_config(temperature: float = 1.2,
                       max_new_tokens: int = 1500) -> DataCollectionConfig:
    """Get configuration for ALFWorld data collection"""
    return DataCollectionConfig.create_custom(
        task_type="alfworld",
        **{
            "generation.temperature": temperature,
            "generation.max_new_tokens": max_new_tokens,
            "storage.output_dir": f"alfworld_data_temp_{temperature}"
        }
    )


if __name__ == "__main__":
    # Demo usage
    print("Available configurations:")
    for name, desc in list_available_configs().items():
        print(f"  {name}: {desc}")

    print("\nExample configuration:")
    config = get_config("math_high_quality")
    print(json.dumps(config.to_dict(), indent=2))