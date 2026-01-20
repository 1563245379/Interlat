#!/bin/bash

# ===================================================
# Open Source Training Script for Hidden State Model
# ===================================================

set -e  # Exit on any error

# Color definitions for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default configuration
DEFAULT_MODEL_PATH="Qwen/Qwen2.5-7B"
DEFAULT_OUTPUT_DIR="./output"
DEFAULT_DATA_PATH="./data/train_data.json"
DEFAULT_HIDDEN_DATA="./data/hidden_states"
DEFAULT_EPOCHS=10
DEFAULT_BATCH_SIZE=2
DEFAULT_LEARNING_RATE="1e-5"
DEFAULT_MAX_LENGTH=2048
DEFAULT_PREPENDED_LENGTH=1000

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help                  Show this help message
    -m, --model PATH            Model path or name (default: $DEFAULT_MODEL_PATH)
    -d, --data PATH             Training data path (default: $DEFAULT_DATA_PATH)
    -H, --hidden-data PATH      Hidden states data path (default: $DEFAULT_HIDDEN_DATA)
    -o, --output DIR            Output directory (default: $DEFAULT_OUTPUT_DIR)
    -e, --epochs NUM            Number of training epochs (default: $DEFAULT_EPOCHS)
    -b, --batch-size NUM        Per device batch size (default: $DEFAULT_BATCH_SIZE)
    -l, --learning-rate RATE    Learning rate (default: $DEFAULT_LEARNING_RATE)
    --max-length NUM            Model max length (default: $DEFAULT_MAX_LENGTH)
    --prepended-length NUM      Prepended sequence length (default: $DEFAULT_PREPENDED_LENGTH)
    --gpus NUM                  Number of GPUs to use (default: auto-detect)
    --no-deepspeed             Disable DeepSpeed (use native PyTorch DDP)
    --dry-run                  Show the command without executing

Examples:
    # Basic training with default settings
    $0

    # Training with custom model and data paths
    $0 --model meta-llama/Meta-Llama-3.1-8B --data ./my_data.json

    # Training with custom output directory and more epochs
    $0 --output ./my_output --epochs 20 --batch-size 4

EOF
}

# Parse command line arguments
parse_args() {
    MODEL_PATH="$DEFAULT_MODEL_PATH"
    DATA_PATH="$DEFAULT_DATA_PATH"
    HIDDEN_DATA="$DEFAULT_HIDDEN_DATA"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    NUM_EPOCHS="$DEFAULT_EPOCHS"
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    LEARNING_RATE="$DEFAULT_LEARNING_RATE"
    MAX_LENGTH="$DEFAULT_MAX_LENGTH"
    PREPENDED_LENGTH="$DEFAULT_PREPENDED_LENGTH"
    USE_DEEPSPEED=true
    DRY_RUN=false
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            -d|--data)
                DATA_PATH="$2"
                shift 2
                ;;
            -H|--hidden-data)
                HIDDEN_DATA="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -e|--epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -l|--learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --max-length)
                MAX_LENGTH="$2"
                shift 2
                ;;
            --prepended-length)
                PREPENDED_LENGTH="$2"
                shift 2
                ;;
            --gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --no-deepspeed)
                USE_DEEPSPEED=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate inputs
validate_inputs() {
    # Check if required files exist
    if [[ ! -f "$DATA_PATH" ]]; then
        print_error "Training data file not found: $DATA_PATH"
        print_info "Please provide a valid training data path using --data option"
        exit 1
    fi

    if [[ ! -d "$HIDDEN_DATA" && ! -f "$HIDDEN_DATA" ]]; then
        print_warning "Hidden states data not found: $HIDDEN_DATA"
        print_info "Training will continue without hidden states data"
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Validate numeric inputs
    if ! [[ "$NUM_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$NUM_EPOCHS" -le 0 ]]; then
        print_error "Invalid number of epochs: $NUM_EPOCHS"
        exit 1
    fi

    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -le 0 ]]; then
        print_error "Invalid batch size: $BATCH_SIZE"
        exit 1
    fi

    if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
        print_error "Invalid number of GPUs: $NUM_GPUS"
        exit 1
    fi

    print_success "Input validation passed"
}

# Create DeepSpeed configuration
create_deepspeed_config() {
    local config_file="$OUTPUT_DIR/ds_config.json"

    print_info "Creating DeepSpeed configuration: $config_file"

    cat > "$config_file" << 'EOF'
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "weight_decay": 0.0,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}
EOF

    echo "$config_file"
}

# Build training command
build_train_command() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local run_name="${timestamp}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}"

    # Base training arguments
    local args=(
        "--data_path" "$DATA_PATH"
        "--model_name_or_path" "$MODEL_PATH"
        "--output_dir" "$OUTPUT_DIR"
        "--prepended_length" "$PREPENDED_LENGTH"
        "--prepended_learnable" "false"
        "--num_train_epochs" "$NUM_EPOCHS"
        "--per_device_train_batch_size" "$BATCH_SIZE"
        "--per_device_eval_batch_size" "$BATCH_SIZE"
        "--evaluation_strategy" "steps"
        "--save_strategy" "steps"
        "--eval_steps" "500"
        "--save_steps" "500"
        "--load_best_model_at_end" "false"
        "--save_total_limit" "3"
        "--learning_rate" "$LEARNING_RATE"
        "--weight_decay" "0.0"
        "--warmup_ratio" "0.03"
        "--plan_similarity_weight" "0.1"
        "--random_contrast_weight" "2.0"
        "--lr_scheduler_type" "cosine"
        "--logging_steps" "10"
        "--model_max_length" "$MAX_LENGTH"
        "--gradient_checkpointing" "true"
        "--lazy_preprocess" "false"
        "--run_name" "$run_name"
        "--report_to" "none"
    )

    # Add hidden data if available
    if [[ -d "$HIDDEN_DATA" || -f "$HIDDEN_DATA" ]]; then
        args+=("--hidden_data" "$HIDDEN_DATA")
    fi

    # Configure distributed training
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        if [[ "$USE_DEEPSPEED" == true ]]; then
            local ds_config=$(create_deepspeed_config)
            args+=("--deepspeed" "$ds_config")
            train_cmd="torchrun --nproc_per_node=$NUM_GPUS train.py"
        else
            train_cmd="torchrun --nproc_per_node=$NUM_GPUS train.py"
        fi
    elif [[ "$NUM_GPUS" -eq 1 ]]; then
        if [[ "$USE_DEEPSPEED" == true ]]; then
            local ds_config=$(create_deepspeed_config)
            args+=("--deepspeed" "$ds_config")
        fi
        train_cmd="python train.py"
    else
        print_warning "No GPUs detected, using CPU training"
        train_cmd="python train.py"
    fi

    # Build the complete command
    FULL_COMMAND="$train_cmd ${args[*]}"
}

# Print training configuration
print_config() {
    print_info "=== Training Configuration ==="
    echo "Model Path:          $MODEL_PATH"
    echo "Data Path:           $DATA_PATH"
    echo "Hidden Data:         $HIDDEN_DATA"
    echo "Output Directory:    $OUTPUT_DIR"
    echo "Number of Epochs:    $NUM_EPOCHS"
    echo "Batch Size:          $BATCH_SIZE"
    echo "Learning Rate:       $LEARNING_RATE"
    echo "Max Length:          $MAX_LENGTH"
    echo "Prepended Length:    $PREPENDED_LENGTH"
    echo "Number of GPUs:      $NUM_GPUS"
    echo "Use DeepSpeed:       $USE_DEEPSPEED"
    echo "================================="
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi

    # Check if training script exists
    if [[ ! -f "train.py" ]]; then
        print_error "train.py not found in current directory"
        print_info "Please make sure you're running this script from the project root"
        exit 1
    fi

    # Check GPU availability if using GPUs
    if [[ "$NUM_GPUS" -gt 0 ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            print_warning "nvidia-smi not found. GPU training may not work properly"
        fi
    fi

    print_success "Dependencies check passed"
}

# Main function
main() {
    print_info "Starting training script..."

    parse_args "$@"
    validate_inputs
    check_dependencies
    build_train_command
    print_config

    print_info "Full training command:"
    echo "$FULL_COMMAND"
    echo

    if [[ "$DRY_RUN" == true ]]; then
        print_info "Dry run mode - command not executed"
        exit 0
    fi

    print_info "Starting training..."
    echo "Press Ctrl+C to stop training"
    echo

    # Execute the training command
    eval "$FULL_COMMAND"

    if [[ $? -eq 0 ]]; then
        print_success "Training completed successfully!"
        print_info "Model saved to: $OUTPUT_DIR"
    else
        print_error "Training failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"