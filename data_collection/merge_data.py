"""
Merge Data Script

This script merges incrementally saved data files from multiple ranks into a unified dataset.
It supports streaming merge to avoid memory overflow.

Usage:
    python merge_data.py --input_dir ./math_train_data_temp_0.8 --output_dir ./merged_data
    python merge_data.py --input_dir ./output --world_size 4 --parquet_max_gb 2.0
"""

import os
import glob
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Required dependencies
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Value, Sequence, load_dataset as hf_load_dataset


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
        print(f"⚠️  No data to write to {out_path}")
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


def collect_all_data_files(input_dir, world_size=None):
    """
    Collect all incremental data files from the input directory
    
    Args:
        input_dir: Input directory containing incremental_rank_* folders
        world_size: Number of ranks (auto-detect if None)
    
    Returns:
        List of file paths
    """
    all_files = []
    
    # Auto-detect world_size if not provided
    if world_size is None:
        rank_dirs = glob.glob(os.path.join(input_dir, "incremental_rank_*"))
        if rank_dirs:
            world_size = len(rank_dirs)
            print(f"Auto-detected {world_size} ranks")
        else:
            world_size = 1
            print("No incremental_rank_* folders found, assuming single rank")
    
    # Collect files from each rank
    for rank in range(world_size):
        incremental_dir = os.path.join(input_dir, f"incremental_rank_{rank}")
        if os.path.exists(incremental_dir):
            files = sorted(glob.glob(os.path.join(incremental_dir, "*.pkl")))
            all_files.extend(files)
            print(f"  Rank {rank}: Found {len(files)} data files")
        else:
            print(f"  Rank {rank}: No incremental directory found")
    
    # Also check temp_rank_data directory if it exists
    temp_dir = os.path.join(input_dir, "temp_rank_data")
    if os.path.exists(temp_dir):
        temp_files = glob.glob(os.path.join(temp_dir, "rank_*.pkl"))
        if temp_files:
            all_files.extend(temp_files)
            print(f"  Found {len(temp_files)} temporary rank files")
    
    return all_files


def streaming_merge_to_parquet(
    all_files,
    output_dir,
    parquet_max_gb=2.0,
    output_name="merged_data",
):
    """
    Stream merge all data files to Parquet format
    
    Args:
        all_files: List of pickle file paths to merge
        output_dir: Output directory for parquet files
        parquet_max_gb: Maximum size per parquet shard in GB
        output_name: Base name for output files
    
    Returns:
        Total number of samples merged
    """
    if len(all_files) == 0:
        print("⚠️  No data files to merge")
        return 0
    
    print(f"🔄 Starting streaming merge of {len(all_files)} files...")
    
    # Prepare output directory
    shards_dir = os.path.join(output_dir, "parquet_shards")
    os.makedirs(shards_dir, exist_ok=True)
    
    max_bytes = int(parquet_max_gb * (1024 ** 3)) - 64 * 1024 * 1024
    shard_rows = []
    shard_bytes = 0
    shard_idx = 0
    total_samples = 0
    
    # Process each file
    for file_idx, filepath in enumerate(tqdm(all_files, desc="Merging files")):
        try:
            with open(filepath, 'rb') as f:
                data_batch = pickle.load(f)
            
            if not isinstance(data_batch, list):
                print(f"⚠️  Skipping invalid file: {filepath}")
                continue
            
            # Process each entry
            for entry in data_batch:
                est = _estimate_entry_bytes(entry)
                
                # Check if we need to write current shard
                if shard_bytes + est > max_bytes and shard_rows:
                    out_path = os.path.join(shards_dir, f"{output_name}-{shard_idx:05d}.parquet")
                    _write_parquet_shard(shard_rows, out_path)
                    total_samples += len(shard_rows)
                    print(f"  ✅ Wrote shard #{shard_idx} with {len(shard_rows)} samples")
                    shard_idx += 1
                    shard_rows = []
                    shard_bytes = 0
                
                shard_rows.append(entry)
                shard_bytes += est
            
            # Periodic garbage collection
            if (file_idx + 1) % 10 == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write final shard
    if shard_rows:
        out_path = os.path.join(shards_dir, f"{output_name}-{shard_idx:05d}.parquet")
        _write_parquet_shard(shard_rows, out_path)
        total_samples += len(shard_rows)
        print(f"  ✅ Wrote final shard #{shard_idx} with {len(shard_rows)} samples")
    
    print(f"✅ Streaming merge complete: {total_samples} total samples in {shard_idx + 1} shards")
    return total_samples


def create_hf_dataset_from_parquet(output_dir, shards_subdir="parquet_shards"):
    """
    Create HuggingFace Dataset from Parquet files
    
    Args:
        output_dir: Directory containing parquet shards
        shards_subdir: Subdirectory name for parquet shards
    
    Returns:
        HuggingFace Dataset object
    """
    shards_dir = os.path.join(output_dir, shards_subdir)
    parquet_files = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    
    if len(parquet_files) == 0:
        print(f"⚠️  No parquet files found in {shards_dir}")
        return None
    
    print(f"📚 Creating HuggingFace Dataset from {len(parquet_files)} parquet files...")
    
    # Load dataset from parquet files
    dataset = hf_load_dataset('parquet', data_files=parquet_files, split='train')
    
    # Save as HuggingFace format
    hf_dir = os.path.join(output_dir, "hf_dataset")
    dataset.save_to_disk(hf_dir)
    print(f"✅ HuggingFace Dataset saved to {hf_dir} with {len(dataset)} samples")
    
    return dataset


def print_dataset_info(dataset):
    """Print information about the merged dataset"""
    if dataset is None:
        return
    
    print("\n" + "=" * 50)
    print("Dataset Information")
    print("=" * 50)
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")
    
    # Sample statistics
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        for key, value in sample.items():
            if key == 'hidden_state':
                if isinstance(value, list):
                    print(f"  {key}: shape {len(value)} x {len(value[0]) if value else 0}")
                else:
                    print(f"  {key}: {type(value)}")
            else:
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"  {key}: {val_str}")
    
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge incrementally saved data files into a unified dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing incremental data files"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for merged data (default: input_dir/merged)"
    )
    parser.add_argument(
        "--world_size", type=int, default=None,
        help="Number of ranks (auto-detect if not specified)"
    )
    parser.add_argument(
        "--parquet_max_gb", type=float, default=2.0,
        help="Maximum size of each parquet shard in GB"
    )
    parser.add_argument(
        "--output_name", type=str, default="merged_data",
        help="Base name for output files"
    )
    parser.add_argument(
        "--create_hf_dataset", action="store_true", default=True,
        help="Create HuggingFace Dataset from parquet files"
    )
    parser.add_argument(
        "--no_hf_dataset", dest="create_hf_dataset", action="store_false",
        help="Don't create HuggingFace Dataset"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "merged")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 50)
    print("Data Merge Configuration")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max parquet size: {args.parquet_max_gb} GB")
    print(f"Output name: {args.output_name}")
    print(f"Create HF Dataset: {args.create_hf_dataset}")
    print("=" * 50 + "\n")
    
    # Collect all data files
    print("📂 Collecting data files...")
    all_files = collect_all_data_files(args.input_dir, args.world_size)
    
    if len(all_files) == 0:
        print("❌ No data files found. Please check the input directory.")
        return
    
    print(f"\n📦 Found {len(all_files)} total files to merge\n")
    
    # Merge to parquet
    total_samples = streaming_merge_to_parquet(
        all_files=all_files,
        output_dir=args.output_dir,
        parquet_max_gb=args.parquet_max_gb,
        output_name=args.output_name,
    )
    
    if total_samples == 0:
        print("❌ No samples were merged.")
        return
    
    # Create HuggingFace Dataset
    dataset = None
    if args.create_hf_dataset:
        print("\n")
        dataset = create_hf_dataset_from_parquet(args.output_dir)
        if dataset:
            print_dataset_info(dataset)
    
    print(f"\n🎉 Merge complete! Total {total_samples} samples saved to {args.output_dir}")
    print(f"   - Parquet shards: {args.output_dir}/parquet_shards/")
    if args.create_hf_dataset and dataset:
        print(f"   - HF Dataset: {args.output_dir}/hf_dataset/")


if __name__ == "__main__":
    main()
