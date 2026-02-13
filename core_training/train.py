#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Interlat Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Interlat Core Training Module

Main training script for Interlat models with hidden state integration.
"""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union, List
import os
import pandas as pd
import torch
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainerCallback
from transformers import EarlyStoppingCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import IntervalStrategy
import matplotlib.pyplot as plt
import torch.distributed as dist
from contextlib import suppress

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
import random
import time
from tqdm import tqdm
import string

import pyarrow as pa
import datasets

from arguments import ModelArguments, DataArguments, TrainingArguments
from callbacks import (
    OptimizerDebugCallback,
    ParameterChangeCallback,
    SaveMHAStateCallback,
    EvalReportCallback,
    GradientLoggingCallback,
    LossRecorderCallback
)

from hidden_model.custom_model import ModelWithInsertedHiddenState
from data_processor import make_supervised_data_module

os.environ["WANDB_DISABLED"] = "true"

# 🔧 Fix Triton cache directory issues for distributed training
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# 🔧 Set NCCL environment variables for better distributed training stability
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# Set a reasonable NCCL timeout (5 minutes) to avoid infinite hangs
os.environ["NCCL_TIMEOUT"] = "300"
# 🔧 Disable PyTorch 2.4+ eager NCCL communicator initialization
os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "0"
# 🔧 Fix NCCL P2P deadlock: GPUs on different PCIe switches cannot do P2P in Docker
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
# 🔧 Use network socket transport as fallback
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0,lo")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# Optional: if you have a Hugging Face token, you can set it here (not required)
# os.environ["HF_TOKEN"] = "your_hf_token_here"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE = -100
EPS = 1e-8  # Prevent log(0)

print(IGNORE_TOKEN_ID)

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
# NOTE: torch.cuda.manual_seed / manual_seed_all moved into train()
# to avoid premature CUDA context creation on all GPUs from all processes,
# which causes 'CUDA device busy or unavailable' race conditions.

def detect_precision() -> str:
    """
    Automatically detect the best supported precision on the current device.
    Returns: "bf16", "fp16", or "no"
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU mode, mixed precision disabled")
        return "no"

    device = torch.cuda.get_device_properties(0)

    if device.major >= 8:
        print(f"GPU {device.name} supports bfloat16, using bf16 mixed precision")
        return "bf16"
    elif device.supports_fp16:
        print(f"GPU {device.name} supports float16, using fp16 mixed precision")
        return "fp16"
    else:
        print(f"GPU {device.name} does not support mixed precision, using default precision")
        return "no"


class GradientSafeWrapper(nn.Module):
    """Gradient-safe wrapper to prevent gradient explosion"""

    def __init__(self, module, max_grad=1.0):
        super().__init__()
        self.module = module
        self.max_grad = max_grad

    def forward(self, x):
        out = self.module(x)
        if self.training:
            out.register_hook(lambda grad: torch.clamp(grad, -self.max_grad, self.max_grad))
        return out


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def _get_local_rank() -> int:
    """Determine local_rank from environment or command-line args."""
    import sys
    lr = int(os.environ.get("LOCAL_RANK", -1))
    if lr == -1:
        for arg in sys.argv:
            if arg.startswith("--local_rank="):
                lr = int(arg.split("=")[1])
                break
    return lr


def _patch_eager_connect():
    """
    Monkey-patch PyTorch 2.4+'s ProcessGroupNCCL to disable eager_connect_single_device.

    In Docker containers, the eager NCCL connection causes 'CUDA device busy'
    errors when multiple processes simultaneously create CUDA contexts.
    This patch makes eager_connect_single_device a no-op, falling back to
    lazy connection which is more robust in containerized environments.
    """
    try:
        from torch.distributed import ProcessGroupNCCL
        if hasattr(ProcessGroupNCCL, 'eager_connect_single_device'):
            ProcessGroupNCCL.eager_connect_single_device = lambda self, device: None
            print("[Patch] Disabled ProcessGroupNCCL.eager_connect_single_device")
            return
    except (ImportError, AttributeError):
        pass

    # Fallback: patch at the C++ binding level via distributed_c10d
    try:
        import torch.distributed.distributed_c10d as c10d
        _orig_helper = c10d._new_process_group_helper

        def _patched_helper(*args, **kwargs):
            pg, store = _orig_helper(*args, **kwargs)
            # The original would call pg.eager_connect_single_device(device)
            # after this returns, but we've already patched the method above.
            # This fallback patches the returned object directly.
            if hasattr(pg, 'eager_connect_single_device'):
                pg.eager_connect_single_device = lambda device: None
            return pg, store

        c10d._new_process_group_helper = _patched_helper
        print("[Patch] Patched _new_process_group_helper to skip eager connect")
    except Exception as e:
        print(f"[Patch] Warning: could not patch eager connect: {e}")


def train():
    global local_rank

    # 🔧 Patch out the problematic eager_connect_single_device before anything
    # triggers distributed initialization (parse_args → TrainingArgs → init_process_group)
    _patch_eager_connect()

    # Set device for this rank before parse_args triggers _setup_devices.
    # Stagger CUDA init across ranks to avoid Docker CUDA driver race conditions.
    lr = _get_local_rank()
    if lr >= 0 and torch.cuda.is_available():
        # Each rank waits (rank * 2s) so GPU contexts are created sequentially
        if lr > 0:
            time.sleep(lr * 2.0)
        # Retry set_device with backoff
        for attempt in range(5):
            try:
                torch.cuda.set_device(lr)
                print(f"[Init] Rank {lr} bound to GPU {lr} ({torch.cuda.get_device_name(lr)})")
                break
            except RuntimeError as e:
                if attempt < 4:
                    wait = (attempt + 1) * 2
                    print(f"[Init] Rank {lr} set_device failed (attempt {attempt+1}/5), "
                          f"retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set local_rank from training_args (DeepSpeed/Trainer canonical source)
    local_rank = training_args.local_rank
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        rank0_print(f"🔧 Process {local_rank} using GPU {local_rank}")

    # 🔧 Set CUDA seeds AFTER local_rank is determined and device is set
    # (doing this at module level causes all processes to init all GPUs simultaneously)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    if local_rank in (-1, 0):
        print("[Before overrides]")
        print("save_total_limit =", training_args.save_total_limit, flush=True)
        print("save_steps       =", training_args.save_steps, flush=True)
        print("load_best_model_at_end =", training_args.load_best_model_at_end, flush=True)

    training_args.evaluation_strategy = IntervalStrategy.STEPS
    training_args.save_strategy = IntervalStrategy.STEPS
    training_args.load_best_model_at_end = False
    local_rank = training_args.local_rank

    if local_rank in (-1, 0):
        print("[After overrides]")
        print("save_total_limit =", training_args.save_total_limit, flush=True)
        print("save_steps       =", training_args.save_steps, flush=True)
        print("load_best_model_at_end =", training_args.load_best_model_at_end, flush=True)

    training_args.max_grad_norm = 3
    print(f"Output_dir: {training_args.output_dir}")

    # Initialize loss callback
    loss_recorder = LossRecorderCallback(
        log_path=os.path.join(training_args.output_dir, "loss_log.csv"),
        plot_path=os.path.join(training_args.output_dir, "loss_curve.png")
    )

    # Automatically detect precision
    precision_mode = detect_precision()
    if precision_mode == "bf16":
        training_args.bf16 = True
        training_args.fp16 = False
    elif precision_mode == "fp16":
        training_args.fp16 = True
        training_args.bf16 = False
    else:
        training_args.fp16 = False
        training_args.bf16 = False

    model_path = model_args.model_name_or_path
    print(f'model_path: {model_path}')

    print("Loading model from Hugging Face or local path...")
    model_args.model_name_or_path = model_path
    print(f"Model_path: {model_args.model_name_or_path}")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load base model
    rank0_print(f"Loading model from {model_args.model_name_or_path}")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    # 🔧 Safe dtype conversion
    def safe_to_bfloat16(model):
        print("🔧 [Safe Convert] Converting model to bfloat16...")
        converted_params = 0
        for name, module in model.named_modules():
            for param_name, param in list(module.named_parameters(recurse=False)):
                if param.dtype != torch.bfloat16:
                    new_param = torch.nn.Parameter(
                        param.data.to(torch.bfloat16),
                        requires_grad=param.requires_grad
                    )
                    setattr(module, param_name, new_param)
                    converted_params += 1
                    if converted_params <= 5:
                        print(f"  🔧 Converted param: {name}.{param_name}")
        for buffer in model.buffers():
            if buffer.dtype != torch.bfloat16:
                buffer.data = buffer.data.to(torch.bfloat16)
        print(f"🔧 [Safe Convert] Converted {converted_params} parameters to bfloat16")

    safe_to_bfloat16(base_model)

    # Create model with prepended hidden states if needed
    use_position_tracking = model_args.prepended_length > 0 and model_args.prepend_position == "first_human"

    if model_args.prepended_length > 0:
        hidden_size = base_model.config.hidden_size
        model_args.prepended_learnable = False

        model = ModelWithInsertedHiddenState(
            base_model,
            model_args.prepended_length,
            hidden_size,
            prepended_learnable=model_args.prepended_learnable,
            plan_similarity_weight=model_args.plan_similarity_weight,
            random_contrast_weight=model_args.random_contrast_weight,
            prepended_input_dim=hidden_size,
        )
        rank0_print(f"Created model with {model_args.prepended_length} prepended hidden states")
        rank0_print(f"Prepended states learnable: {model_args.prepended_learnable}")
        rank0_print(f"Prepend position: {model_args.prepend_position}")
        rank0_print("Using hidden states from dataset, not random initialization")
    else:
        model = base_model

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token != tokenizer.unk_token and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.unk_token = tokenizer.pad_token

    print(f"use_position_tracking is: {use_position_tracking}")

    # Add special tokens
    if use_position_tracking:
        special_tokens_dict = {'additional_special_tokens': ['<FIRST_HUMAN_END>', '<bop>', '<eop>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"DEBUG: Added {num_added_toks} special tokens.")
        print(f"DEBUG: Tokenizer vocab size after adding special tokens: {len(tokenizer)}")

        debug_bop_id = tokenizer.convert_tokens_to_ids('<bop>')
        debug_eop_id = tokenizer.convert_tokens_to_ids('<eop>')
        print(f"DEBUG: ID for <bop>: {debug_bop_id}, type: {type(debug_bop_id)}")
        print(f"DEBUG: ID for <eop>: {debug_eop_id}, type: {type(debug_eop_id)}")

        if debug_bop_id is None or debug_eop_id is None:
            print("ERROR: <bop> or <eop> token IDs are None after addition.")
        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))
            print("DEBUG: Model embedding layer resized.")

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_path=model_args.model_name_or_path,
        use_position_tracking=use_position_tracking,
        prepended_length=model_args.prepended_length
    )

    # 🔧 Do NOT manually move model to GPU here.
    # DeepSpeed / HuggingFace Trainer handles device placement automatically.
    # Manual placement before Trainer init causes NCCL deadlock.
    tokenizer.padding_side = "right"
    model.tokenizer = tokenizer

    training_args.save_safetensors = False
    training_args.remove_unused_columns = False
    training_args.per_device_train_batch_size = 2
    training_args.per_device_eval_batch_size = 1   # eval runs 3 forward passes, needs less batch
    training_args.gradient_accumulation_steps = 2  # effective batch = 2 * 2 * 4gpus = 16

    # NOTE: Model is already loaded as bfloat16, no need to convert again.
    # Redundant safe_to_bfloat16 removed to avoid parameter aliasing issues.

    # Ensure eval_strategy is consistent with evaluation_strategy
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    setattr(training_args, "eval_strategy", training_args.evaluation_strategy)
    print("evaluation_strategy =", training_args.evaluation_strategy)
    print("eval_strategy =", getattr(training_args, "eval_strategy", None))
    print("save_strategy =", training_args.save_strategy)
    print("eval_steps    =", training_args.eval_steps)
    print("save_steps    =", training_args.save_steps)

    # Start trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[
            loss_recorder,
            EvalReportCallback(
                tokenizer=tokenizer,
                eval_dataset=data_module["eval_dataset"],
                data_collator=data_module["data_collator"],
                num_samples=3,
            ),
            OptimizerDebugCallback(),
            SaveMHAStateCallback(),
            GradientLoggingCallback(),
            ParameterChangeCallback(model_reference=model),
        ],
    )

    trainer.train()

    # Final fix for lm_head (if needed)
    model.base_model.lm_head.weight = torch.nn.Parameter(model.base_model.lm_head.weight.clone())

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()