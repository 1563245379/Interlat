import time
import random
import string
import numpy as np
import pandas as pd
import torch
import datasets
from datasets import load_dataset, Split
from tqdm import tqdm

class HiddenStateLoader:
    def __init__(self, dataset_name, max_samples=0, task_ids=None):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.task_ids = set(task_ids) if task_ids is not None else None

        # Automatically load data during initialization
        self._load_data()

    def _load_data(self):
        print(f"Loading tensor data from {self.dataset_name}")
        self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.TRAIN)

        print(f"Loaded {len(self.dataset)} records.")

        def optimized_nested_convert(nested_array):
            """Optimized nested array conversion"""
            try:
                if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                    list_data = nested_array.tolist()
                    numpy_array = np.array(list_data, dtype=np.float32)
                    return torch.from_numpy(numpy_array).to(torch.bfloat16)
                elif isinstance(nested_array, np.ndarray):
                    return torch.from_numpy(nested_array.astype(np.float32)).to(torch.bfloat16)
                elif isinstance(nested_array, list):
                    numpy_array = np.array(nested_array, dtype=np.float32)
                    return torch.from_numpy(numpy_array).to(torch.bfloat16)
                else:
                    return torch.tensor(nested_array, dtype=torch.bfloat16)
            except Exception as e:
                print(f"Conversion failed: {e}")
                return None

        # Directly iterate over the HuggingFace dataset — no pandas intermediate copy
        self.id_to_data = {}
        success_count = 0
        skip_count = 0

        total = len(self.dataset)
        if self.task_ids is not None:
            print(f"Filtering: only loading {len(self.task_ids)} task_ids out of {total} records")

        for row in tqdm(self.dataset, total=total, desc="Building id_to_data"):
            tid = row['task_id']
            # Skip if we have a filter and this id is not needed
            if self.task_ids is not None and tid not in self.task_ids:
                skip_count += 1
                continue
            tensor = optimized_nested_convert(row['hidden_state'])
            if tensor is not None:
                self.id_to_data[tid] = {
                    'hidden_state': tensor,
                    'plan': row['plan'],
                }
                success_count += 1
                # Early stop if max_samples is set and we have enough
                if self.max_samples > 0 and success_count >= self.max_samples:
                    break

        print(f"Successfully converted: {success_count}/{total} arrays (skipped: {skip_count})")

        # Verify results
        if self.id_to_data:
            sample_key = next(iter(self.id_to_data))
            sample_data = self.id_to_data[sample_key]
            print(f"Sample tensor shape: {sample_data['hidden_state'].shape}")
            print(f"Sample tensor dtype: {sample_data['hidden_state'].dtype}")
            print(f"Sample plan: {sample_data['plan'][:100]}...")

    # Query function is very efficient
    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']