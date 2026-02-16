#!/bin/bash
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

# Interlat Compression Training Script
# This script trains compressed latent models using teacher-student distillation

set -e  # Exit on error
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set PYTHONPATH to include necessary modules
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/core_training:$(pwd)/compression_training

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Default configurations
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-./trained_models/teacher_model}"
DATA_PATH="${DATA_PATH:-./data/training_data.json}"
HIDDEN_REPO="${HIDDEN_REPO:-your_hidden_states_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./compressed_models}"
K="${K:-128}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --student-model)
            STUDENT_MODEL="$2"
            shift 2
            ;;
        --teacher-model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --hidden-repo)
            HIDDEN_REPO="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --K)
            K="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --bf16)
            BF16="--bf16"
            shift
            ;;
        --gradient-checkpointing)
            GRAD_CHECKPOINT="--gradient_checkpointing"
            shift
            ;;
        --help|-h)
            echo "Interlat Compression Training Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --student-model <name>         Student model path or HuggingFace name"
            echo "  --teacher-model <path>         Teacher model path (from core training)"
            echo "  --data-path <path>            Training data JSON path"
            echo "  --hidden-repo <name>          HuggingFace dataset with hidden states"
            echo "  --output-dir <path>           Output directory for compressed model"
            echo "  --K <int>                     Compression length (default: 128)"
            echo "  --epochs <int>                Number of training epochs (default: 3)"
            echo "  --batch-size <int>            Training batch size (default: 2)"
            echo "  --learning-rate <float>       Learning rate (default: 5e-5)"
            echo "  --bf16                        Use bfloat16 precision"
            echo "  --gradient-checkpointing      Enable gradient checkpointing"
            echo "  --help, -h                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --teacher-model ./trained_models/qwen_model"
            echo "  $0 --K 256 --bf16 --gradient-checkpointing"
            echo "  $0 --student-model Qwen/Qwen2.5-7B-Instruct --epochs 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================================"
echo -e "${BLUE}Interlat Compression Training${NC}"
echo "========================================================"
echo "Student Model: $STUDENT_MODEL"
echo "Teacher Model: $TEACHER_MODEL"
echo "Data Path: $DATA_PATH"
echo "Hidden Repo: $HIDDEN_REPO"
echo "Output Dir: $OUTPUT_DIR"
echo "Compression K: $K"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "========================================================"

# Validate teacher model exists
if [[ ! "$TEACHER_MODEL" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]] && [[ ! -d "$TEACHER_MODEL" ]]; then
    print_error "Teacher model not found: $TEACHER_MODEL"
    echo "Please provide a valid local path or train a teacher model first using:"
    echo "  ./scripts/train_model.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")/logs"

# Set up logging
LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/compression_$(date +%Y%m%d_%H%M%S).log"

print_step "1" "Starting Compression Training"
echo "Logs will be saved to: $LOG_FILE"

# Build command
CMD_ARGS=(
    --student_model_path "$STUDENT_MODEL"
    --teacher_model_path "$TEACHER_MODEL"
    --data_path "$DATA_PATH"
    --hf_hidden_repo "$HIDDEN_REPO"
    --output_dir "$OUTPUT_DIR"
    --K "$K"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --logging_steps 10
    --save_steps 200
    --eval_steps 200
    --warmup_ratio 0.03
)

# Add optional flags
if [[ -n "$BF16" ]]; then
    CMD_ARGS+=($BF16)
fi

if [[ -n "$GRAD_CHECKPOINT" ]]; then
    CMD_ARGS+=($GRAD_CHECKPOINT)
fi

# Run compression training
python compression_training/compress.py \
    "${CMD_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

# Check if training was successful
if [[ $? -eq 0 ]]; then
    print_success "Compression training completed successfully!"
    echo "📁 Compressed model saved to: $OUTPUT_DIR"
    echo "📊 Logs saved to: $LOG_FILE"

    # Show output structure
    echo ""
    echo "Compressed model structure:"
    find "$OUTPUT_DIR" -name "pytorch_model*.bin" -o -name "config.json" -o -name "tokenizer*" | head -10 | sed 's/^/  /'

    echo ""
    echo "🚀 You can now use the compressed model for evaluation!"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate compressed model performance:"
    echo "   python eval/alfworld/eval_agent/main.py --model_path $OUTPUT_DIR"
    echo "2. Compare with full-size model latency and accuracy"
    echo "3. Deploy in multi-agent scenarios for efficiency gains"
else
    print_error "Compression training failed. Check the logs for details."
    echo "📊 Logs: $LOG_FILE"
    exit 1
fi