# Training Scripts Documentation

This directory contains open-source training scripts for the hidden state model. All company-specific configurations and sensitive information have been removed.

## Files Overview

- `train.sh` - Advanced training script with command-line arguments
- `submit.sh` - Interactive training setup script
- `README_TRAINING.md` - This documentation file

## Quick Start

### Option 1: Using the Interactive Script (Recommended for beginners)

```bash
# Run the interactive setup script
./submit.sh
```

This script will guide you through:
- Task selection (alfworld, math, or custom)
- Model selection (Qwen, LLaMA, or custom)
- Hidden states configuration
- Output directory setup
- GPU detection and training execution

### Option 2: Using the Advanced Script (For experienced users)

```bash
# Basic usage with defaults
./train.sh

# Custom training with specific parameters
./train.sh --model Qwen/Qwen2.5-7B --data ./my_data.json --epochs 20 --batch-size 4

# Training with hidden states data
./train.sh --model meta-llama/Meta-Llama-3.1-8B --hidden-data ./hidden_states --output ./my_output

# See all available options
./train.sh --help
```

## Prerequisites

### System Requirements
- Python 3.8+
- PyTorch 1.12+
- transformers library
- deepspeed (for distributed training)
- CUDA-capable GPU (recommended)

### Installation
```bash
# Install required packages
pip install torch transformers deepspeed accelerate

# Or install from requirements.txt if available
pip install -r requirements.txt
```

## Data Format

### Training Data
Your training data should be in JSON format:
```json
[
  {
    "input": "Your input text here",
    "output": "Expected output text here",
    "instruction": "Optional instruction"
  },
  ...
]
```

### Hidden States Data
If you have hidden states data, organize it as:
```
hidden_states/
├── sample_1.pt
├── sample_2.pt
└── ...
```

Each `.pt` file should contain a PyTorch tensor with hidden states.

## Configuration Options

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model path or HuggingFace model name | Qwen/Qwen2.5-7B |
| `--data` | Training data path | ./data/train_data.json |
| `--output` | Output directory | ./output |
| `--epochs` | Number of training epochs | 10 |
| `--batch-size` | Per device batch size | 2 |
| `--learning-rate` | Learning rate | 1e-5 |
| `--max-length` | Maximum sequence length | 2048 |
| `--gpus` | Number of GPUs to use | auto-detect |

### Advanced Parameters

- `--prepended-length`: Length of prepended hidden states (default: 1000)
- `--no-deepspeed`: Disable DeepSpeed and use native PyTorch DDP
- `--dry-run`: Show the command without executing

## Training Examples

### 1. Basic Text Generation Training
```bash
./train.sh \
  --model Qwen/Qwen2.5-7B \
  --data ./data/conversation_data.json \
  --epochs 5 \
  --batch-size 4 \
  --output ./models/conversation_model
```

### 2. Math Problem Solving
```bash
./train.sh \
  --model meta-llama/Meta-Llama-3.1-8B \
  --data ./data/math_problems.json \
  --hidden-data ./hidden_states/math \
  --epochs 15 \
  --learning-rate 5e-6
```

### 3. Multi-GPU Training
```bash
./train.sh \
  --model Qwen/Qwen2.5-7B \
  --data ./data/large_dataset.json \
  --gpus 4 \
  --batch-size 8 \
  --epochs 20
```

## Output Structure

After training, your output directory will contain:
```
output/
├── checkpoint-500/          # Training checkpoints
├── checkpoint-1000/
├── final_model/             # Final trained model
├── training_log.txt         # Training logs
├── ds_config.json          # DeepSpeed configuration
└── run_training.sh         # Generated training command
```

## Monitoring Training

### Using TensorBoard (if available)
```bash
tensorboard --logdir ./output --port 6006
```

### Checking Logs
```bash
# Real-time log monitoring
tail -f ./output/training_log.txt

# Search for specific metrics
grep "loss" ./output/training_log.txt
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Enable gradient checkpointing (enabled by default)
   - Use CPU offloading with DeepSpeed

2. **Model Not Found**
   - Ensure the model path is correct
   - Check internet connection for HuggingFace models
   - Verify model access permissions

3. **Data Loading Errors**
   - Check JSON format of training data
   - Ensure data paths are correct
   - Verify file permissions

4. **GPU Detection Issues**
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify CUDA installation
   - Use `--gpus 0` to force CPU training

### Performance Tips

1. **Optimize Batch Size**: Use the largest batch size that fits in memory
2. **Mixed Precision**: bf16 is enabled by default for better performance
3. **Gradient Accumulation**: Automatically handled by DeepSpeed
4. **CPU Offloading**: Enabled by default to save GPU memory

## Advanced Configuration

### Custom DeepSpeed Configuration
If you need custom DeepSpeed settings, modify the generated `ds_config.json` file before training.

### Environment Variables
```bash
# Set specific CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Adjust memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review training logs for error messages
3. Ensure all dependencies are properly installed
4. Verify data format and file paths

## License

This training framework is designed to be open-source friendly. Please ensure compliance with the licenses of the specific models you choose to train.