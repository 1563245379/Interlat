# Interlat: Latent Space Communication Framework

Interlat is a novel multi-agent communication framework that enables agents to collaborate **entirely in latent space**, bypassing natural language as the communication medium. By sharing last-layer hidden states instead of discrete tokens, Interlat allows for richer, more information-preserving interactions between agents.

## Project Overview

- **Purpose**: Enables language-free inter-agent communication via temporally aligned hidden states.
- **Backbone Models**: Supports Qwen2.5 (0.5B, 7B) and LLaMA3.1 (8B) models.
- **Key Benchmarks**: Evaluated on ALFWorld (embodied planning) and MATH (symbolic reasoning).
- **Architecture**:
    - **Data Collection**: Extracts hidden states and plans from LLMs during reasoning tasks.
    - **Hidden Model Integration**: Prepends hidden state representations to model embeddings using a modular processor (`AdaptiveProjection`, `MultiheadAttention`, `LayerNorm`).
    - **Core Training**: Uses a multi-objective approach with Cross-Entropy loss, Plan Similarity loss (KL/Cosine), and Random Contrast loss to align latent communication with task objectives.
    - **Latent Compression**: Distillation-based framework to reduce communication length by up to 24x.

## Building and Running

### Environment Setup
The project uses a Conda environment named `interlat`.
```bash
source /workspace/miniconda/bin/activate interlat
pip install -r requirements.txt
```

### Key Workflow Steps

1. **Collect Hidden States**:
   Extract hidden states and generate plans for specific tasks.
   ```bash
   ./scripts/collect_math.sh --mode train --subjects algebra --output_dir ./data/math_hidden_states
   ```

2. **Data Conversion** (if needed):
   Convert collected hidden states (Arrow/Parquet) and match with ground truth solutions for training.
   ```bash
   python convert_math_data.py
   ```

3. **Train Model**:
   Fine-tune a model to integrate and utilize latent representations.
   ```bash
   ./scripts/train_model.sh --model Qwen/Qwen2.5-7B-Instruct --data ./data/training_data.json --hidden-data ./data/hidden_states
   ```

4. **Evaluate Model**:
   ```bash
   python eval/math/math_evaluator.py --model_name ./trained_models/math_model --dataset hendrycks/MATH --split test
   ```

5. **Latent Compression**:
   Train a student model to use compressed latent states from a teacher.
   ```bash
   ./scripts/train_compression.sh --teacher-model ./trained_models/teacher_model --hidden-repo your_hidden_states_dataset --K 128
   ```

## Development Conventions

- **Directory Structure**:
    - `core_training/`: Core training logic, arguments, and custom model definitions.
    - `data_collection/`: Scripts for extracting hidden states from benchmarks.
    - `eval/`: Task-specific evaluation suites.
    - `scripts/`: Shell scripts for end-to-end workflows.
    - `data/`: Intended storage for training JSONs and hidden state Parquet shards.
- **Data Format**:
    - Training text data: JSON with standard conversation format (`human`, `gpt` turns).
    - Latent data: Sharded Parquet files containing `task_id`, `plan`, and `hidden_state` (float32/bfloat16).
- **Special Tokens**: Uses `<FIRST_HUMAN_END>`, `<bop>`, and `<eop>` to mark hidden state insertion points in the token sequence.
- **Precision**: Prefers `bfloat16` for model weights and latent processing to ensure stability and efficiency.
- **Distributed Training**: Supports Multi-GPU training via DeepSpeed or standard DistributedDataParallel (DDP).

## Gemini Added Memories
- When using a conda environment, the command `source /workspace/miniconda/bin/activate interlat` must be executed first to initialize conda.
- When running this project for the first time, execute the following commands to set up the Hugging Face cache:
  ```bash
  export TRITON_CACHE_DIR=/workspace/cache
  export HF_HOME=/workspace/cache
  export TRANSFORMERS_CACHE=$HF_HOME
  export HF_DATASETS_CACHE=$HF_HOME
  ```
