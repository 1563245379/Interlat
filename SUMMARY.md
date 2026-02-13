# Interlat Project Context

## Overview
**Interlat** is a novel multi-agent communication framework enabling agents to communicate **entirely in latent space**, bypassing natural language. It supports heterogeneous agents and aggressive compression (up to 24x).

*   **Paper:** [arXiv:2511.09149](https://arxiv.org/abs/2511.09149)
*   **Key Tasks:** ALFWorld (Embodied Planning), MATH (Symbolic Reasoning).
*   **Tech Stack:** Python, PyTorch, Transformers, DeepSpeed, Accelerate.

## Environment Setup

1.  **Create Conda Environment:**
    ```bash
    conda create -n interlat python=3.8 -y
    conda activate interlat
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Optional: Install in dev mode
    pip install -e .
    ```

3.  **Environment Variables:**
    Set `HF_HOME` to cache models effectively.
    ```bash
    export HF_HOME=/path/to/huggingface
    ```

## Quick Start

The easiest way to run the project is via the `quick_start.sh` script.

*   **Quick Demo (ALFWorld, Small Model):**
    ```bash
    ./scripts/quick_start.sh
    ```

*   **Math Task:**
    ```bash
    ./scripts/quick_start.sh --task math
    ```

*   **Full Workflow (Large Model, All Data):**
    ```bash
    ./scripts/quick_start.sh --task alfworld --model-size large --full
    ```

## Core Workflows

### 1. Data Collection
Collect hidden states from a teacher model.

*   **ALFWorld:**
    ```bash
    ./scripts/collect_alfworld.sh \
        --dataset_path ./datasets/alfworld_dataset.json \
        --output_dir ./data/alfworld_hidden_states
    ```
*   **Math:**
    ```bash
    ./scripts/collect_math.sh \
        --mode train \
        --subjects algebra geometry \
        --output_dir ./data/math_hidden_states
    ```
*   **Python Direct:**
    ```bash
    python data_collection/collect_data.py [alfworld|math] --dataset_path ...
    ```

### 2. Training
Train models to understand and generate hidden states.

*   **Script:**
    ```bash
    ./scripts/train_model.sh \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --data "./data/training_data.json" \
        --hidden-data "./data/hidden_states" \
        --epochs 10
    ```
*   **Python Direct:**
    ```bash
    python core_training/train.py --model_name_or_path ...
    ```

### 3. Compression Training
Distill latents into a smaller dimension (e.g., K=128).

*   **Script:**
    ```bash
    ./scripts/train_compression.sh \
        --teacher-model ./trained_models/teacher_model \
        --hidden-repo your_hidden_states_dataset \
        --K 128
    ```

### 4. Evaluation

*   **ALFWorld:**
    ```bash
    python eval/alfworld/eval_agent/main.py \
        --model_path ./trained_models/alfworld_model \
        --dataset_path ./data/alfworld_hidden_states \
        --split dev
    ```
*   **Math:**
    ```bash
    python eval/math/math_evaluator.py \
        --model_name ./trained_models/math_model \
        --dataset hendrycks/MATH \
        --split test
    ```

## Configuration system
The project uses a structured configuration system in `data_collection/config.py`.
You can use predefined configs like:
*   `default`: Standard config.
*   `math_high_quality`: Optimized for math reasoning (temp 0.7).
*   `alfworld_creative`: Optimized for planning (temp 1.2).
*   `fast_prototyping`: For quick tests.

## Key Directories
*   `core_training/`: Main training logic, model definitions (`hidden_model`), and data processing.
*   `compression_training/`: Logic for latent compression (teacher-student).
*   `data_collection/`: Scripts for extracting hidden states from LLMs.
*   `eval/`: Evaluation scripts for supported benchmarks.
*   `scripts/`: Shell scripts for orchestrating workflows.
