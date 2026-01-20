#!/usr/bin/env python3
"""
Base Data Collection Framework
=============================

A comprehensive framework for collecting model reasoning data across different tasks.
This module provides the core infrastructure for distributed data collection,
model loading, and hidden state extraction.

Features:
- Distributed training support with fault tolerance
- Flexible data storage (HuggingFace datasets, Parquet files)
- Automatic hidden state extraction and processing
- Configurable generation parameters
- Robust error handling and logging

Author: AI Research Team
Version: 2.0
"""

import os
import sys
import json
import pickle
import time
import argparse
import random
import logging
import socket
import datetime
import math
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Data processing
import datasets
from datasets import Features, Value, Sequence, load_from_disk
from datasets import Dataset as HFDataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s",
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)

class CollectedSample:
    """
    Data structure for a single collected sample

    Attributes:
        task_id: Unique identifier for the task
        task_content: The original task/question content
        plan_output: Generated plan or solution
        hidden_states: Model hidden states during generation
        task_metadata: Additional task information (type, level, etc.)
        generation_metadata: Generation parameters and timestamps
    """

    def __init__(self,
                 task_id: str,
                 task_content: str,
                 plan_output: str,
                 hidden_states: np.ndarray,
                 task_metadata: Optional[Dict] = None,
                 generation_metadata: Optional[Dict] = None):
        self.task_id = task_id
        self.task_content = task_content
        self.plan_output = plan_output
        self.hidden_states = hidden_states
        self.task_metadata = task_metadata or {}
        self.generation_metadata = generation_metadata or {}
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format"""
        return {
            "task_id": self.task_id,
            "task": self.task_content,
            "plan": self.plan_output,
            "hidden_state": self.hidden_states,
            "task_type": self.task_metadata.get("type", "unknown"),
            "task_level": self.task_metadata.get("level", "unknown"),
            "timestamp": self.timestamp,
            **self.generation_metadata
        }

class DataStorageManager:
    """
    Manages data storage in multiple formats

    Supports:
    - HuggingFace Dataset format
    - Parquet files with automatic sharding
    - Distributed data collection and merging
    """

    def __init__(self,
                 output_dir: str = "collected_data",
                 max_shard_size_gb: float = 2.0,
                 compression: str = "zstd"):
        self.output_dir = Path(output_dir)
        self.max_shard_size_gb = max_shard_size_gb
        self.compression = compression

        # Global data collection
        self.collected_samples: List[CollectedSample] = []

    def add_sample(self, sample: CollectedSample):
        """Add a collected sample to the storage"""
        self.collected_samples.append(sample)

    def estimate_sample_size(self, sample_dict: Dict) -> int:
        """Estimate memory size of a sample in bytes"""
        hidden_states = sample_dict["hidden_state"]
        if isinstance(hidden_states, np.ndarray):
            bytes_hidden = hidden_states.size * 4  # float32
        else:
            bytes_hidden = sum(len(row) for row in hidden_states) * 4

        # Estimate text fields
        text_fields = ["task", "task_id", "plan", "task_type", "task_level"]
        bytes_text = sum(
            len(str(sample_dict.get(field, "")).encode("utf-8"))
            for field in text_fields
        )

        return int((bytes_hidden + bytes_text) * 1.2)  # 20% overhead

    def write_parquet_shard(self, samples: List[Dict], output_path: Path):
        """Write a shard of samples to Parquet format"""
        # Define schema
        hidden_state_type = pa.list_(pa.list_(pa.float32()))
        schema = pa.schema([
            pa.field('task', pa.string()),
            pa.field('task_id', pa.string()),
            pa.field('plan', pa.string()),
            pa.field('task_type', pa.string()),
            pa.field('task_level', pa.string()),
            pa.field('hidden_state', hidden_state_type),
            pa.field('timestamp', pa.string()),
        ])

        # Prepare data
        data = {
            'task': [s['task'] for s in samples],
            'task_id': [s['task_id'] for s in samples],
            'plan': [s['plan'] for s in samples],
            'task_type': [s.get('task_type', 'unknown') for s in samples],
            'task_level': [s.get('task_level', 'unknown') for s in samples],
            'hidden_state': [
                (s['hidden_state'].tolist() if isinstance(s['hidden_state'], np.ndarray)
                 else s['hidden_state']) for s in samples
            ],
            'timestamp': [s.get('timestamp', '') for s in samples],
        }

        # Create table and write
        table = pa.table(data, schema=schema)
        pq.write_table(table, output_path,
                      compression=self.compression,
                      use_dictionary=True)

    def save_to_huggingface_format(self, samples: List[Dict]) -> str:
        """Save samples as HuggingFace Dataset"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define features
        features = Features({
            'task': Value('string'),
            'task_id': Value('string'),
            'plan': Value('string'),
            'task_type': Value('string'),
            'task_level': Value('string'),
            'hidden_state': Sequence(Sequence(Value('float32'))),
            'timestamp': Value('string'),
        })

        # Prepare data dict
        data_dict = {
            "task": [s['task'] for s in samples],
            "task_id": [s['task_id'] for s in samples],
            "plan": [s['plan'] for s in samples],
            "task_type": [s.get('task_type', 'unknown') for s in samples],
            "task_level": [s.get('task_level', 'unknown') for s in samples],
            "hidden_state": [
                (s['hidden_state'].astype(np.float32).tolist()
                 if isinstance(s['hidden_state'], np.ndarray)
                 else s['hidden_state']) for s in samples
            ],
            "timestamp": [s.get('timestamp', '') for s in samples],
        }

        # Create and save dataset
        dataset = HFDataset.from_dict(data_dict, features=features)
        hf_dir = self.output_dir / "huggingface_dataset"
        dataset.save_to_disk(str(hf_dir))

        return str(hf_dir)

    def save_to_parquet_shards(self, samples: List[Dict]) -> str:
        """Save samples as sharded Parquet files"""
        shards_dir = self.output_dir / "parquet_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        max_bytes = int(self.max_shard_size_gb * (1024 ** 3)) - 64 * 1024 * 1024
        shard_samples = []
        shard_bytes = 0
        shard_idx = 0

        for sample in samples:
            estimated_size = self.estimate_sample_size(sample)

            # Write current shard if adding this sample would exceed limit
            if shard_bytes + estimated_size > max_bytes and shard_samples:
                shard_path = shards_dir / f"data-{shard_idx:05d}.parquet"
                self.write_parquet_shard(shard_samples, shard_path)
                logging.info(f"Wrote shard {shard_idx}: {shard_path}")

                shard_idx += 1
                shard_samples = []
                shard_bytes = 0

            shard_samples.append(sample)
            shard_bytes += estimated_size

        # Write final shard
        if shard_samples:
            shard_path = shards_dir / f"data-{shard_idx:05d}.parquet"
            self.write_parquet_shard(shard_samples, shard_path)
            logging.info(f"Wrote final shard {shard_idx}: {shard_path}")

        return str(shards_dir)

    def finalize_storage(self):
        """Convert all collected samples to final storage formats"""
        if not self.collected_samples:
            logging.warning("No samples collected to save")
            return

        # Convert to dict format
        samples_dict = [sample.to_dict() for sample in self.collected_samples]

        # Save in multiple formats
        hf_path = self.save_to_huggingface_format(samples_dict)
        parquet_path = self.save_to_parquet_shards(samples_dict)

        logging.info(f"Data saved - HuggingFace: {hf_path}, Parquet: {parquet_path}")
        logging.info(f"Total samples collected: {len(self.collected_samples)}")

class DistributedDataCollector:
    """
    Handles distributed data collection across multiple GPUs/nodes
    """

    def __init__(self, storage_manager: DataStorageManager):
        self.storage_manager = storage_manager
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.temp_dir = Path("temp_distributed_data")

    def setup_distributed_environment(self) -> Tuple[int, int, int]:
        """Initialize distributed training environment with robust error handling"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int, default=-1,
                          help="Local rank for distributed training")
        args, _ = parser.parse_known_args()

        # Debug environment
        env_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        logging.info("Distributed environment variables:")
        for var in env_vars:
            logging.info(f"{var}: {os.environ.get(var, 'Not set')}")
        logging.info(f"Hostname: {socket.gethostname()}")
        logging.info(f"Available GPUs: {torch.cuda.device_count()}")

        # Determine rank and world size
        if 'LOCAL_RANK' not in os.environ and args.local_rank != -1:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

        self.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        self.rank = int(os.environ.get('RANK', self.local_rank))
        self.world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))

        logging.info(f"Using rank: {self.rank}, local_rank: {self.local_rank}, world_size: {self.world_size}")

        # Initialize process group
        try:
            if self.world_size > 1:
                torch.cuda.set_device(self.local_rank)
                dist.init_process_group(
                    backend='nccl',
                    timeout=datetime.timedelta(minutes=300),
                    init_method=os.environ.get('INIT_METHOD', 'env://'),
                    world_size=self.world_size,
                    rank=self.rank
                )

                # Update from process group
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                logging.info(f"Process group initialized. Final rank: {self.rank}, world_size: {self.world_size}")
            else:
                logging.info("Single process mode - no distributed setup needed")

        except Exception as e:
            logging.error(f"Failed to initialize distributed training: {e}")
            logging.info("Falling back to single process mode")
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1

        return self.rank, self.world_size, self.local_rank

    def save_rank_data(self):
        """Save current rank's data to temporary file"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.temp_dir / f"rank_{self.rank}.pkl"

        samples_data = [sample.to_dict() for sample in self.storage_manager.collected_samples]
        with open(temp_file, "wb") as f:
            pickle.dump(samples_data, f)

        logging.info(f"Rank {self.rank} saved {len(samples_data)} samples to {temp_file}")

    def merge_distributed_data(self):
        """Merge data from all ranks (called by rank 0)"""
        if self.rank != 0:
            return

        logging.info("Starting data merge from all ranks...")
        all_samples = []

        # Collect data from all ranks
        for rank_id in range(self.world_size):
            temp_file = self.temp_dir / f"rank_{rank_id}.pkl"
            if temp_file.exists():
                with open(temp_file, "rb") as f:
                    rank_samples = pickle.load(f)
                    all_samples.extend(rank_samples)
                logging.info(f"Loaded {len(rank_samples)} samples from rank {rank_id}")
            else:
                logging.warning(f"Missing data file for rank {rank_id}: {temp_file}")

        # Convert back to CollectedSample objects
        collected_objects = []
        for sample_dict in all_samples:
            sample = CollectedSample(
                task_id=sample_dict["task_id"],
                task_content=sample_dict["task"],
                plan_output=sample_dict["plan"],
                hidden_states=sample_dict["hidden_state"],
                task_metadata={
                    "type": sample_dict.get("task_type", "unknown"),
                    "level": sample_dict.get("task_level", "unknown")
                }
            )
            collected_objects.append(sample)

        # Update storage manager and finalize
        self.storage_manager.collected_samples = collected_objects
        self.storage_manager.finalize_storage()

        # Cleanup temporary files
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logging.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def finalize_distributed_collection(self):
        """Complete the distributed data collection process"""
        if self.world_size <= 1:
            # Single process - directly finalize
            self.storage_manager.finalize_storage()
            return

        # Multi-process - save individual rank data
        self.save_rank_data()

        # Wait for all processes
        if self.world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f"Barrier synchronization failed: {e}")

        # Rank 0 merges all data
        self.merge_distributed_data()

        # Final synchronization
        if self.world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f"Final barrier failed: {e}")

        logging.info(f"Rank {self.rank} completed data collection")

    def cleanup_distributed(self):
        """Clean up distributed resources"""
        if self.world_size > 1:
            try:
                dist.destroy_process_group()
                logging.info(f"Rank {self.rank} cleaned up process group")
            except Exception as e:
                logging.error(f"Failed to destroy process group: {e}")

class ModelManager:
    """
    Handles model loading and generation with robust error handling
    """

    def __init__(self,
                 model_path: str,
                 device_id: int = 0,
                 torch_dtype: torch.dtype = torch.float32):
        self.model_path = model_path
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.torch_dtype = torch_dtype

        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with error handling"""
        logging.info(f"Loading model from: {self.model_path}")

        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map={"": self.device},
                trust_remote_code=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Set pad token if needed
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            logging.info(f"Model successfully loaded on device: {self.device}")
            return self.model, self.tokenizer

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def generate_with_hidden_states(self,
                                  prompt: str,
                                  max_new_tokens: int = 1500,
                                  temperature: float = 0.8,
                                  do_sample: bool = True,
                                  top_p: float = 0.9,
                                  max_hidden_states: int = 10000) -> Tuple[str, np.ndarray]:
        """
        Generate text and extract hidden states

        Returns:
            Tuple of (generated_text, hidden_states_array)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")

        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate with hidden states
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][input_length:],
            skip_special_tokens=True
        )

        # Extract hidden states from last layer of each generation step
        step_hidden_states = []
        generation_steps = outputs.hidden_states

        # Limit the number of hidden states to save memory
        start_idx = max(0, len(generation_steps) - max_hidden_states)

        for step_idx in range(start_idx, len(generation_steps)):
            # Get last layer hidden states [batch_size, sequence_length, hidden_size]
            last_layer_hidden = generation_steps[step_idx][-1]
            # Take the hidden state of the newly generated token (last position)
            new_token_hidden = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]
            step_hidden_states.append(new_token_hidden)

        # Stack to get [batch_size, num_steps, hidden_size]
        if step_hidden_states:
            hidden_states_tensor = torch.stack(step_hidden_states, dim=1)

            # Convert to numpy and remove batch dimension if batch_size=1
            if hidden_states_tensor.size(0) == 1:
                hidden_states_array = hidden_states_tensor.squeeze(0).cpu().numpy()
            else:
                hidden_states_array = hidden_states_tensor.cpu().numpy()
        else:
            # Fallback: empty array with correct shape
            hidden_dim = self.model.config.hidden_size
            hidden_states_array = np.empty((0, hidden_dim), dtype=np.float32)

        return generated_text, hidden_states_array

class TaskDataset(Dataset):
    """
    Generic dataset wrapper for task data
    """

    def __init__(self, data_items: List[Any]):
        self.data_items = data_items

    def __len__(self) -> int:
        return len(self.data_items)

    def __getitem__(self, idx: int) -> Any:
        return self.data_items[idx]

# Base class for specific data collectors
class BaseDataCollectionPipeline:
    """
    Base class for implementing specific data collection pipelines

    Subclasses should implement:
    - load_dataset_for_task()
    - create_prompt_for_task()
    - process_task_batch()
    """

    def __init__(self,
                 model_path: str,
                 output_dir: str = "collected_data",
                 temperature: float = 0.8,
                 max_new_tokens: int = 1500,
                 batch_size: int = 1):

        # Initialize components
        self.storage_manager = DataStorageManager(output_dir)
        self.distributed_collector = DistributedDataCollector(self.storage_manager)

        # Setup distributed environment
        self.rank, self.world_size, self.local_rank = (
            self.distributed_collector.setup_distributed_environment()
        )

        # Initialize model manager
        self.model_manager = ModelManager(
            model_path=model_path,
            device_id=self.local_rank
        )

        # Generation parameters
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Setup logging with rank info
        for handler in logging.root.handlers:
            handler.setFormatter(logging.Formatter(
                f"%(asctime)s - %(levelname)s - [Rank {self.rank}] %(message)s"
            ))

    def initialize(self):
        """Initialize the pipeline - load models and datasets"""
        logging.info("Initializing data collection pipeline...")

        # Load model and tokenizer
        self.model, self.tokenizer = self.model_manager.load_model_and_tokenizer()

        # Load dataset (implemented by subclasses)
        self.dataset = self.load_dataset_for_task()

        # Distribute dataset across processes
        self.my_dataset = self.distribute_dataset()

        logging.info(f"Pipeline initialized. Processing {len(self.my_dataset)} samples.")

    def distribute_dataset(self) -> List[Any]:
        """Distribute dataset across processes"""
        if self.world_size <= 1:
            return self.dataset

        total_samples = len(self.dataset)
        samples_per_process = math.ceil(total_samples / self.world_size)
        start_idx = self.rank * samples_per_process
        end_idx = min(start_idx + samples_per_process, total_samples)

        my_samples = self.dataset[start_idx:end_idx] if hasattr(self.dataset, '__getitem__') else list(self.dataset)[start_idx:end_idx]

        logging.info(f"Rank {self.rank} processing samples {start_idx} to {end_idx-1} ({len(my_samples)} total)")
        return my_samples

    def run_collection(self):
        """Run the main data collection process"""
        logging.info("Starting data collection...")

        # Create dataloader
        task_dataset = TaskDataset(self.my_dataset)
        dataloader = DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues with GPU
        )

        # Process batches
        total_processed = 0
        with tqdm(total=len(dataloader), desc=f"Rank {self.rank} Collection") as pbar:
            for batch_idx, batch_items in enumerate(dataloader):
                try:
                    processed_count = self.process_task_batch(batch_items, batch_idx)
                    total_processed += processed_count

                    pbar.update(1)
                    pbar.set_postfix({
                        "Processed": total_processed,
                        "Collected": len(self.storage_manager.collected_samples)
                    })

                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    continue

        logging.info(f"Data collection completed. Processed {total_processed} samples.")

    def finalize(self):
        """Finalize the collection process"""
        logging.info("Finalizing data collection...")

        # Finalize distributed collection
        self.distributed_collector.finalize_distributed_collection()

        # Cleanup
        self.distributed_collector.cleanup_distributed()

        logging.info("Data collection pipeline completed successfully!")

    # Abstract methods to be implemented by subclasses
    def load_dataset_for_task(self) -> List[Any]:
        """Load the dataset for the specific task"""
        raise NotImplementedError("Subclasses must implement load_dataset_for_task()")

    def create_prompt_for_task(self, task_item: Any) -> str:
        """Create a prompt for the given task item"""
        raise NotImplementedError("Subclasses must implement create_prompt_for_task()")

    def process_task_batch(self, batch_items: List[Any], batch_idx: int) -> int:
        """Process a batch of task items and collect data"""
        raise NotImplementedError("Subclasses must implement process_task_batch()")

if __name__ == "__main__":
    # This is a base framework - specific implementations should inherit from BaseDataCollectionPipeline
    logging.info("Base data collection framework loaded successfully")
    logging.info("Create specific collectors by inheriting from BaseDataCollectionPipeline")