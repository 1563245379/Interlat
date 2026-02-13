"""
Math Data Collection with Command Line Arguments

This script collects mathematical reasoning data using a language model, with all parameters
configurable via command line arguments instead of environment variables.

Usage:
    python math_collection_args.py --mode train --output_dir ./output --temperature 0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import glob
import json
import pickle
import time
import argparse
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import sys
import datasets
from datasets import Features, Value, Sequence, load_from_disk
from datasets import Dataset as HFDataset
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import re
import socket
import datetime
import math
from pathlib import Path

# Global variable to collect all data
all_data = []

def save_train_data(task, task_id, plan, hidden_state, task_type, task_level):
    """Collect task, task_id, plan, hidden_state, task_type, and task_level into a global list"""
    # hidden_state is expected to be [T, H] or [1, T, H]; normalize to [T, H]
    if hidden_state.dim() == 3 and hidden_state.size(0) == 1:
        hidden_np = hidden_state.cpu().numpy().squeeze(0).astype(np.float32)  # [T, H]
    else:
        hidden_np = hidden_state.cpu().numpy().astype(np.float32)             # [T, H]

    entry = {
        "task": task,
        "task_id": task_id,
        "plan": plan,
        "hidden_state": hidden_np,             # [T, H]
        "task_type": str(task_type),           # store as string for compatibility
        "task_level": str(task_level),
    }
    all_data.append(entry)
    return None


# Required dependencies
import pyarrow as pa
import pyarrow.parquet as pq

# Checkpoint management
def save_checkpoint(processed_ids, checkpoint_path):
    """Save checkpoint with processed sample IDs"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)

def load_checkpoint(checkpoint_path):
    """Load checkpoint and return set of processed sample IDs"""
    if not os.path.exists(checkpoint_path):
        return set()
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_ids', []))
    except Exception as e:
        print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}")
        return set()

def incremental_save_data(data_batch, output_dir, rank, save_counter, args):
    """Incrementally save collected data"""
    if len(data_batch) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    incremental_dir = os.path.join(output_dir, f"incremental_rank_{rank}")
    os.makedirs(incremental_dir, exist_ok=True)
    
    filename = os.path.join(incremental_dir, f"batch_{save_counter:05d}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(data_batch, f)
    print(f"Rank {rank} incrementally saved {len(data_batch)} samples to {filename}")

def load_incremental_data(output_dir, rank):
    """Load all incrementally saved data for a specific rank"""
    incremental_dir = os.path.join(output_dir, f"incremental_rank_{rank}")
    if not os.path.exists(incremental_dir):
        return []
    
    all_data = []
    files = sorted(glob.glob(os.path.join(incremental_dir, "batch_*.pkl")))
    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                batch_data = pickle.load(f)
                all_data.extend(batch_data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    print(f"Rank {rank} loaded {len(all_data)} samples from incremental saves")
    return all_data

def _estimate_entry_bytes(entry: dict) -> int:
    """Conservatively estimate the byte size of one sample for shard size control"""
    hs = entry["hidden_state"]
    if isinstance(hs, np.ndarray):
        bytes_hs = hs.size * 4  # float32
    else:
        bytes_hs = sum(len(row) for row in hs) * 4

    text_keys = ["task", "task_id", "plan", "task_type", "task_level"]
    bytes_txt = 0
    for k in text_keys:
        v = entry.get(k, "")
        if v is None:
            v = ""
        bytes_txt += len(str(v).encode("utf-8"))

    return int((bytes_hs + bytes_txt) * 1.2)  # +20% overhead

def _write_parquet_shard(rows: list, out_path: str):
    """Write rows to a single Parquet file"""
    if len(rows) == 0:
        return
    
    hs_type = pa.list_(pa.list_(pa.float32()))
    schema = pa.schema([
        pa.field('task', pa.string()),
        pa.field('task_id', pa.string()),
        pa.field('plan', pa.string()),
        pa.field('task_type', pa.string()),
        pa.field('task_level', pa.string()),
        pa.field('hidden_state', hs_type),
    ])

    tasks        = [r['task'] for r in rows]
    task_ids     = [r['task_id'] for r in rows]
    plans        = [r['plan'] for r in rows]
    task_types   = [r['task_type'] for r in rows]
    task_levels  = [r['task_level'] for r in rows]
    hidden_lists = [
        (r['hidden_state'].tolist() if isinstance(r['hidden_state'], np.ndarray) else r['hidden_state'])
        for r in rows
    ]

    table = pa.table({
        'task': pa.array(tasks, type=pa.string()),
        'task_id': pa.array(task_ids, type=pa.string()),
        'plan': pa.array(plans, type=pa.string()),
        'task_type': pa.array(task_types, type=pa.string()),
        'task_level': pa.array(task_levels, type=pa.string()),
        'hidden_state': pa.array(hidden_lists, type=hs_type),
    }, schema=schema)

    pq.write_table(table, out_path, compression="zstd", use_dictionary=True)

def convert_to_hf_dataset(
    data,
    output_dir="final_output",
    parquet_max_gb: float = 2.0,
    write_full: bool = True,
    full_filename: str = "data_full.parquet",
    write_shards: bool = True,
    shards_subdir: str = "parquet_shards",
):
    """
    Convert collected data into a HuggingFace Dataset and output:
      1) A single large Parquet file
      2) Multiple size-limited Parquet shards
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build HuggingFace Dataset
    features = Features({
        'task': Value('string'),
        'task_id': Value('string'),
        'plan': Value('string'),
        'task_type': Value('string'),
        'task_level': Value('string'),
        'hidden_state': Sequence(Sequence(Value('float32'))),  # [T, H]
    })
    hf_dataset = HFDataset.from_dict({
        "task":        [d['task'] for d in data],
        "task_id":     [d['task_id'] for d in data],
        "plan":        [d['plan'] for d in data],
        "task_type":   [d['task_type'] for d in data],
        "task_level":  [d['task_level'] for d in data],
        "hidden_state":[
            (d['hidden_state'].astype(np.float32).tolist()
             if isinstance(d['hidden_state'], np.ndarray)
             else d['hidden_state'])
            for d in data
        ],
    }, features=features)

    hf_dir = os.path.join(output_dir, "hf_dataset")
    hf_dataset.save_to_disk(hf_dir)
    print(f"✅ HuggingFace Dataset saved to {hf_dir}")

    # 2) Write a single large Parquet file
    if write_full:
        full_path = os.path.join(output_dir, full_filename)
        _write_parquet_shard(data, full_path)
        print(f"✅ Single Parquet file saved: {full_path}")

    # 3) Write size-limited Parquet shards
    if write_shards:
        shards_dir = os.path.join(output_dir, shards_subdir)
        os.makedirs(shards_dir, exist_ok=True)

        max_bytes = int(parquet_max_gb * (1024 ** 3)) - 64 * 1024 * 1024
        shard_rows, shard_bytes, shard_idx = [], 0, 0

        for entry in data:
            est = _estimate_entry_bytes(entry)

            if est >= max_bytes and shard_rows:
                out_path = os.path.join(shards_dir, f"data-{shard_idx:05d}.parquet")
                _write_parquet_shard(shard_rows, out_path)
                print(f"✅ Wrote shard #{shard_idx} -> {out_path}")
                shard_idx += 1
                shard_rows, shard_bytes = [], 0

            if shard_bytes + est > max_bytes and shard_rows:
                out_path = os.path.join(shards_dir, f"data-{shard_idx:05d}.parquet")
                _write_parquet_shard(shard_rows, out_path)
                print(f"✅ Wrote shard #{shard_idx} -> {out_path}")
                shard_idx += 1
                shard_rows, shard_bytes = [], 0

            shard_rows.append(entry)
            shard_bytes += est

        if shard_rows:
            out_path = os.path.join(shards_dir, f"data-{shard_idx:05d}.parquet")
            _write_parquet_shard(shard_rows, out_path)
            print(f"✅ Wrote shard #{shard_idx} -> {out_path}")

        print(f"✅ Parquet shards saved to {shards_dir} (≤ {parquet_max_gb}GB each)")


def streaming_merge_to_parquet(
    output_dir,
    world_size,
    temp_dir="temp_rank_data",
    parquet_max_gb=2.0,
    batch_size=1000,
):
    """
    流式合并所有rank的增量数据，直接写入Parquet，避免内存溢出
    
    Args:
        output_dir: 输出目录
        world_size: 进程数量
        temp_dir: 临时目录
        parquet_max_gb: 每个Parquet文件的最大大小（GB）
        batch_size: 每次处理的样本数量
    """
    print("🔄 Starting streaming merge to Parquet...")
    
    # 收集所有增量文件路径
    all_incremental_files = []
    for rank in range(world_size):
        incremental_dir = os.path.join(output_dir, f"incremental_rank_{rank}")
        if os.path.exists(incremental_dir):
            files = sorted(glob.glob(os.path.join(incremental_dir, "batch_*.pkl")))
            all_incremental_files.extend(files)
            print(f"  Found {len(files)} incremental files from rank {rank}")
    
    # 加载temp_dir中的最终rank数据
    for rank in range(world_size):
        rank_file = os.path.join(temp_dir, f"rank_{rank}.pkl")
        if os.path.exists(rank_file):
            all_incremental_files.append(rank_file)
            print(f"  Found final data file from rank {rank}")
    
    if len(all_incremental_files) == 0:
        print("⚠️  No data files found to merge")
        return 0
    
    print(f"📦 Total {len(all_incremental_files)} files to process")
    
    # 准备输出目录
    shards_dir = os.path.join(output_dir, "parquet_shards")
    os.makedirs(shards_dir, exist_ok=True)
    
    max_bytes = int(parquet_max_gb * (1024 ** 3)) - 64 * 1024 * 1024
    shard_rows = []
    shard_bytes = 0
    shard_idx = 0
    total_samples = 0
    
    # 流式处理每个文件
    for file_idx, filepath in enumerate(tqdm(all_incremental_files, desc="Merging files")):
        try:
            with open(filepath, 'rb') as f:
                data_batch = pickle.load(f)
            
            if not isinstance(data_batch, list):
                print(f"⚠️  Skipping invalid file: {filepath}")
                continue
            
            # 分批处理数据
            for entry in data_batch:
                est = _estimate_entry_bytes(entry)
                
                # 检查是否需要写入当前shard
                if shard_bytes + est > max_bytes and shard_rows:
                    out_path = os.path.join(shards_dir, f"data-{shard_idx:05d}.parquet")
                    _write_parquet_shard(shard_rows, out_path)
                    total_samples += len(shard_rows)
                    print(f"  ✅ Wrote shard #{shard_idx} with {len(shard_rows)} samples -> {out_path}")
                    shard_idx += 1
                    shard_rows = []
                    shard_bytes = 0
                
                shard_rows.append(entry)
                shard_bytes += est
            
            # 定期清理内存
            if (file_idx + 1) % 10 == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            continue
    
    # 写入最后一个shard
    if shard_rows:
        out_path = os.path.join(shards_dir, f"data-{shard_idx:05d}.parquet")
        _write_parquet_shard(shard_rows, out_path)
        total_samples += len(shard_rows)
        print(f"  ✅ Wrote final shard #{shard_idx} with {len(shard_rows)} samples -> {out_path}")
    
    print(f"✅ Streaming merge complete: {total_samples} total samples in {shard_idx + 1} shards")
    return total_samples


def create_hf_dataset_from_parquet(output_dir, shards_subdir="parquet_shards"):
    """
    从Parquet文件创建HuggingFace Dataset，避免大内存占用
    """
    shards_dir = os.path.join(output_dir, shards_subdir)
    parquet_files = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    
    if len(parquet_files) == 0:
        print(f"⚠️  No parquet files found in {shards_dir}")
        return None
    
    print(f"📚 Creating HuggingFace Dataset from {len(parquet_files)} parquet files...")
    
    # 使用Parquet文件创建Dataset
    from datasets import load_dataset as hf_load_dataset
    
    dataset = hf_load_dataset('parquet', data_files=parquet_files, split='train')
    
    # 保存为HuggingFace格式
    hf_dir = os.path.join(output_dir, "hf_dataset")
    dataset.save_to_disk(hf_dir)
    print(f"✅ HuggingFace Dataset saved to {hf_dir} with {len(dataset)} samples")
    
    return dataset


def save_rank_data(rank, all_data, temp_dir="temp_rank_data"):
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"rank_{rank}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Rank {rank} saved data to {filename}")

def finalize_data_save_and_merge(rank, world_size, output_dir="final_output", temp_dir="temp_rank_data", args=None):
    """使用流式处理合并数据，避免内存溢出"""
    # 加载该rank的所有增量数据
    incremental_data = load_incremental_data(output_dir, rank)
    
    # 添加内存中剩余的数据（如果有）
    if len(all_data) > 0:
        print(f"Rank {rank}: Adding {len(all_data)} samples from memory")
        incremental_data.extend(all_data)
    
    print(f"Rank {rank}: Total {len(incremental_data)} samples to merge")
    
    if world_size <= 1:
        # 单进程模式：使用流式处理
        print(f"Rank {rank}: Using streaming merge for single process")
        
        # 先保存当前rank的数据（如果还没保存）
        if len(incremental_data) > 0:
            final_file = os.path.join(output_dir, "incremental_rank_0", "final.pkl")
            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            with open(final_file, 'wb') as f:
                pickle.dump(incremental_data, f)
        
        # 使用流式合并
        total_samples = streaming_merge_to_parquet(
            output_dir=output_dir,
            world_size=1,
            temp_dir=temp_dir,
            parquet_max_gb=args.parquet_max_gb if args else 2.0,
            batch_size=args.streaming_batch_size if (args and hasattr(args, 'streaming_batch_size')) else 1000,
        )
        
        # 从Parquet创建HuggingFace Dataset
        if args and args.write_full:
            create_hf_dataset_from_parquet(output_dir)
        
        return total_samples

    # 多进程模式：保存当前rank的合并数据
    save_rank_data(rank, incremental_data, temp_dir=temp_dir)
    dist.barrier()

    if rank == 0:
        print("Start streaming merge from all ranks...")
        
        # 使用流式合并
        total_samples = streaming_merge_to_parquet(
            output_dir=output_dir,
            world_size=world_size,
            temp_dir=temp_dir,
            parquet_max_gb=args.parquet_max_gb if args else 2.0,
            batch_size=args.streaming_batch_size if (args and hasattr(args, 'streaming_batch_size')) else 1000,
        )
        
        # 从Parquet创建HuggingFace Dataset
        if args and args.write_full:
            create_hf_dataset_from_parquet(output_dir)
        
        # 清理临时目录
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Temporary directory removed: {temp_dir}")
        
        print(f"✅ Merge complete: {total_samples} total samples")
    else:
        print(f"Rank {rank} finished saving data.")


def setup_distributed(args):
    """Initialize distributed training environment"""
    print("Environment variables before setup:")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"Hostname: {socket.gethostname()}")

    gpu_count = torch.cuda.device_count()
    print(f"Available GPU count: {gpu_count}")

    if 'LOCAL_RANK' not in os.environ and args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size or gpu_count))

    print(f"Using rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=datetime.timedelta(minutes=args.distributed_timeout),
            init_method=args.init_method,
            world_size=world_size,
            rank=rank
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process group initialized. Rank: {rank}, World Size: {world_size}")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        import traceback
        traceback.print_exc()
        rank = 0
        local_rank = 0
        world_size = 1
        print("Falling back to single-process mode")

    return rank, world_size


class MMDataset(Dataset):
    """Dataset class for MMLU data"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_model(model_path, rank, torch_dtype="float32"):
    """Load model and prepare for DDP"""
    device = torch.device(f"cuda:{rank}")

    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_obj = dtype_mapping.get(torch_dtype, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_obj,
        device_map={"": device}
    )

    print(f"Model initialized on GPU {rank}.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def agent_generate(model, tokenizer, texts, device, args):
    """支持批量推理的生成函数
    Args:
        texts: 可以是单个字符串或字符串列表
    Returns:
        如果输入是单个文本，返回(generated_text, hidden_seq)
        如果输入是文本列表，返回(generated_texts, hidden_seqs)列表
    """
    # 处理单个文本的情况
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False
    
    # 为所有文本添加chat模板
    formatted_texts = ['<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n' for text in texts]
    
    # 批量tokenize
    inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_lengths = attention_mask.sum(dim=1)  # 每个样本的实际长度
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

    # 处理每个样本的生成结果
    results = []
    for idx in range(len(texts)):
        input_length = input_lengths[idx].item()
        generated_text = tokenizer.decode(
            outputs.sequences[idx][input_length:], skip_special_tokens=True
        )

        # 提取hidden states
        step_hiddens = []
        steps = outputs.hidden_states
        start_index = max(0, len(steps) - args.max_hidden_states)
        for i in range(start_index, len(steps)):
            last_layer = steps[i][-1]
            h_last = last_layer[idx:idx+1, -1, :]  # 保持batch维度
            step_hiddens.append(h_last)

        hidden_seq = torch.stack(step_hiddens, dim=1)
        if hidden_seq.size(0) == 1:
            hidden_seq = hidden_seq.squeeze(0)

        results.append((generated_text, hidden_seq))
    
    # 如果是单个输入，返回单个结果
    if single_input:
        return results[0]
    
    return results


def infer_chain_batch(model, tokenizer, batch_data, device, args):
    """批量执行推理链并保存样本
    Args:
        batch_data: list of dicts with keys: task, task_solution, task_id, task_type, task_level
    """
    from textwrap import dedent

    # Use custom prompt if provided, otherwise use default
    if args.custom_prompt:
        prompt_template = args.custom_prompt
    else:
        prompt_template = r"""
        You are a mathematical problem-solving planner.

        When you receive a math problem (Question), your task is to output a high-level solution plan (Plan)
        that guides another model to solve the problem in detail.

        IMPORTANT RULES:
        1. Provide a plan only, not the final answer.
        2. Keep the plan abstract and general.
        3. Do not copy or reference any existing solution steps.
        4. Use the exact output format specified.

        Question:
        {question}
        """.strip()

    def build_plan_prompt(question: str) -> str:
        return dedent(prompt_template).format(question=question).strip()

    # 构建批量prompts
    prompts = [build_plan_prompt(item['task']) for item in batch_data]
    
    # 批量生成
    results = agent_generate(model, tokenizer, prompts, device, args)
    
    # 保存每个样本
    for idx, (generated_text, hidden_seq) in enumerate(results):
        item = batch_data[idx]
        plan = generated_text

        if args.verbose:
            print(f"Sample {item['task_id']} output: {generated_text[:100]}...")

        save_train_data(
            task=item['task'],
            task_id=item['task_id'],
            plan=plan,
            hidden_state=hidden_seq,
            task_type=item['task_type'],
            task_level=item['task_level'],
        )

def infer_chain(model, tokenizer, task, task_solution, task_id, task_type, task_level, device, args):
    """单个样本推理（保持向后兼容）"""
    batch_data = [{
        'task': task,
        'task_solution': task_solution,
        'task_id': task_id,
        'task_type': task_type,
        'task_level': task_level
    }]
    infer_chain_batch(model, tokenizer, batch_data, device, args)


def evaluate(model, tokenizer, dataloader, device, rank, world_size, args, base_offset: int, processed_ids: set):
    """Evaluate model performance with checkpoint resume and incremental save
    Args:
        processed_ids: set of already processed task IDs to skip
    """
    task_num = 0
    correct_count = 0
    save_counter = 0
    temp_batch_data = []  # 临时存储待保存的数据

    # 计算checkpoint路径
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_rank_{rank}.json")
    
    pbar = tqdm(total=len(dataloader), desc=f"GPU {rank} processing")

    try:
        batch_buffer = []  # 用于批量推理的缓冲区
        
        for batch_idx, task_item in enumerate(dataloader):
            # 获取当前batch的所有样本
            batch_size = len(task_item['problem'])
            
            for i in range(batch_size):
                global_idx = base_offset + batch_idx * args.batch_size + i + 1
                task_id = f'MATH_{global_idx}'
                
                # 检查是否已处理
                if task_id in processed_ids:
                    if args.verbose:
                        print(f"Skipping already processed task: {task_id}")
                    continue
                
                task = task_item['problem'][i]
                task_level = task_item['level'][i]
                task_type = task_item['type'][i]
                task_solution = task_item['solution'][i]
                
                batch_buffer.append({
                    'task': task,
                    'task_solution': task_solution,
                    'task_id': task_id,
                    'task_type': task_type,
                    'task_level': task_level
                })
                
                # 当缓冲区达到推理batch size时进行批量推理
                if len(batch_buffer) >= args.inference_batch_size:
                    infer_chain_batch(model, tokenizer, batch_buffer, device, args)
                    task_num += len(batch_buffer)
                    
                    # 更新已处理ID列表
                    for item in batch_buffer:
                        processed_ids.add(item['task_id'])
                    
                    # 增量保存检查
                    if args.save_every > 0 and task_num % args.save_every == 0:
                        # 保存当前收集的数据
                        incremental_save_data(all_data[-args.save_every:], args.output_dir, rank, save_counter, args)
                        save_counter += 1
                        
                        # 保存checkpoint
                        save_checkpoint(processed_ids, checkpoint_path)
                        print(f"GPU {rank}: Saved checkpoint at {task_num} tasks")
                    
                    batch_buffer = []
            
            pbar.update(1)
        
        # 处理剩余的样本
        if batch_buffer:
            infer_chain_batch(model, tokenizer, batch_buffer, device, args)
            task_num += len(batch_buffer)
            for item in batch_buffer:
                processed_ids.add(item['task_id'])
            
            # 最后保存一次
            if args.save_every > 0:
                incremental_save_data(all_data[-len(batch_buffer):], args.output_dir, rank, save_counter, args)
                save_checkpoint(processed_ids, checkpoint_path)
                print(f"GPU {rank}: Final checkpoint saved")
                
    except Exception as e:
        print(f"GPU {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
        # 出错时也保存checkpoint
        if args.save_every > 0:
            save_checkpoint(processed_ids, checkpoint_path)
            print(f"GPU {rank}: Emergency checkpoint saved")
    finally:
        pbar.close()

    return correct_count, task_num


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Math Data Collection with Configurable Parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path", type=str,
        default="Interlat_preview/models/Qwen2.5-7B",
        help="Path to the model (HuggingFace model name or local path)"
    )
    model_group.add_argument(
        "--torch_dtype", type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="PyTorch dtype for the model"
    )

    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        "--mode", type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to use (train or test)"
    )
    data_group.add_argument(
        "--subjects", type=str, nargs="+",
        default=['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'],
        help="Math subjects to include"
    )
    data_group.add_argument(
        "--output_dir", type=str,
        help="Output directory for collected data (default: auto-generated)"
    )

    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature for text generation"
    )
    gen_group.add_argument(
        "--max_new_tokens", type=int, default=1500,
        help="Maximum number of new tokens to generate"
    )
    gen_group.add_argument(
        "--do_sample", action="store_true", default=True,
        help="Whether to use sampling for generation"
    )
    gen_group.add_argument(
        "--no_sample", dest="do_sample", action="store_false",
        help="Disable sampling (use greedy decoding)"
    )
    gen_group.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    gen_group.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling parameter"
    )
    gen_group.add_argument(
        "--num_beams", type=int, default=1,
        help="Number of beams for beam search"
    )
    gen_group.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="Repetition penalty for generation"
    )
    gen_group.add_argument(
        "--max_hidden_states", type=int, default=10000,
        help="Maximum number of hidden states to collect"
    )

    # Prompt customization
    prompt_group = parser.add_argument_group('Prompt Configuration')
    prompt_group.add_argument(
        "--custom_prompt", type=str,
        help="Custom prompt template (use {question} placeholder)"
    )
    prompt_group.add_argument(
        "--prompt_file", type=str,
        help="Path to file containing custom prompt template"
    )

    # Distributed training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    dist_group.add_argument(
        "--world_size", type=int, default=None,
        help="World size for distributed training (auto-detected if not specified)"
    )
    dist_group.add_argument(
        "--distributed_backend", type=str, default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend"
    )
    dist_group.add_argument(
        "--distributed_timeout", type=int, default=300,
        help="Distributed training timeout in minutes"
    )
    dist_group.add_argument(
        "--init_method", type=str, default="env://",
        help="Initialization method for distributed training"
    )

    # Storage configuration
    storage_group = parser.add_argument_group('Storage Configuration')
    storage_group.add_argument(
        "--temp_dir", type=str, default="temp_rank_data",
        help="Temporary directory for rank data during distributed processing"
    )
    storage_group.add_argument(
        "--parquet_max_gb", type=float, default=2.0,
        help="Maximum size of parquet shards in GB"
    )
    storage_group.add_argument(
        "--streaming_batch_size", type=int, default=1000,
        help="Batch size for streaming merge (larger = faster but more memory)"
    )
    storage_group.add_argument(
        "--write_full", action="store_true", default=True,
        help="Write a single large parquet file"
    )
    storage_group.add_argument(
        "--no_write_full", dest="write_full", action="store_false",
        help="Don't write a single large parquet file"
    )
    storage_group.add_argument(
        "--write_shards", action="store_true", default=True,
        help="Write parquet shards"
    )
    storage_group.add_argument(
        "--no_write_shards", dest="write_shards", action="store_false",
        help="Don't write parquet shards"
    )

    # Miscellaneous
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging"
    )
    misc_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for data loading (DataLoader batch size)"
    )
    misc_group.add_argument(
        "--inference_batch_size", type=int, default=4,
        help="Batch size for model inference (推理时的批处理大小)"
    )
    misc_group.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of workers for data loading"
    )
    
    # Checkpoint and incremental save
    checkpoint_group = parser.add_argument_group('Checkpoint and Incremental Save')
    checkpoint_group.add_argument(
        "--save_every", type=int, default=100,
        help="Save checkpoint every N samples (0 to disable incremental save)"
    )
    checkpoint_group.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to checkpoint file for resume (auto-generated if not specified)"
    )
    checkpoint_group.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from checkpoint if available"
    )

    return parser


def validate_arguments(args):
    """Validate and process arguments"""
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./math_{args.mode}_data_temp_{args.temperature}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load custom prompt from file if specified
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            args.custom_prompt = f.read().strip()
        print(f"Loaded custom prompt from: {args.prompt_file}")

    # Validate subjects
    valid_subjects = {
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    }
    invalid_subjects = set(args.subjects) - valid_subjects
    if invalid_subjects:
        raise ValueError(f"Invalid subjects: {invalid_subjects}. Valid subjects: {valid_subjects}")

    return args


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_arguments(args)

    # Print configuration
    print("=" * 50)
    print("Math Data Collection Configuration")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Subjects: {args.subjects}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Torch dtype: {args.torch_dtype}")
    if args.custom_prompt:
        print(f"Custom prompt: {'Yes' if args.custom_prompt else 'No'}")
    print("=" * 50)

    # Initialize distributed environment
    rank, world_size = setup_distributed(args)
    print(f"After setup: Rank = {rank}, World Size = {world_size}")

    print(f"Running in {args.mode} mode with temperature={args.temperature}")

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f"cuda:{local_rank}")

    model, tokenizer = load_model(args.model_path, local_rank, args.torch_dtype)

    full_train_dataset = concatenate_datasets([
        load_dataset('EleutherAI/hendrycks_math', config, split=args.mode)
        for config in args.subjects
    ])

    # full_train_dataset = full_train_dataset[:100]

    print(f"Loaded {len(full_train_dataset)} samples")

    if world_size > 1:
        total_samples = len(full_train_dataset)
        samples_per_worker = math.ceil(total_samples / world_size)
        start_idx = rank * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        my_indices = list(range(start_idx, end_idx))
        my_dataset = full_train_dataset.select(my_indices)
        base_offset = start_idx
        print(
            f"GPU {rank} handling samples {start_idx} to {end_idx-1}, "
            f"total: {len(my_dataset)} | base_offset={base_offset}"
        )
    else:
        my_dataset = full_train_dataset
        base_offset = 0
        print(f"Single process mode - handling all {len(my_dataset)} samples | base_offset={base_offset}")

    mm_dataset = MMDataset(my_dataset)
    dataloader = DataLoader(
        mm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"GPU {rank} number of samples to process: {len(dataloader)}")

    # 加载checkpoint（如果启用）
    processed_ids = set()
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_rank_{rank}.json")
        processed_ids = load_checkpoint(checkpoint_path)
        print(f"GPU {rank}: Loaded checkpoint with {len(processed_ids)} processed samples")
        
        # 同时加载已保存的增量数据
        existing_data = load_incremental_data(args.output_dir, rank)
        print(f"GPU {rank}: Loaded {len(existing_data)} samples from incremental saves")

    correct_count, task_num = evaluate(
        model, tokenizer, dataloader, device, rank, world_size, args,
        base_offset=base_offset,
        processed_ids=processed_ids
    )

    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            print(f"Barrier error on GPU {rank}: {e}")

    finalize_data_save_and_merge(
        rank, world_size,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        args=args
    )

    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error destroying process group on GPU {rank}: {e}")

    print(f"GPU {rank} completed processing {task_num} tasks.")


if __name__ == "__main__":
    main()