# Math Problem Evaluation Tool

An open-source tool for evaluating language models on mathematical reasoning tasks. This tool extracts answers from LaTeX-formatted model outputs and provides comprehensive evaluation metrics.

## Features

- 🧮 **LaTeX Answer Extraction**: Automatically extracts answers from `\boxed{}` and `\fbox{}` LaTeX commands
- 📊 **Comprehensive Evaluation**: Detailed logging and result reporting
- 🔄 **Multiple Sampling**: Support for multiple samples per question for robust evaluation
- 🤖 **Model Agnostic**: Works with any HuggingFace compatible model
- 📈 **Progress Tracking**: Real-time accuracy tracking with progress bars
- 💾 **Result Storage**: Saves both detailed results and summary statistics

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Evaluate a model on the MATH dataset:

```bash
python math_evaluator.py --model_name microsoft/DialoGPT-medium --device cuda
```

### Advanced Usage

```bash
python math_evaluator.py \
    --model_name your-model-name \
    --device auto \
    --max_length 2048 \
    --temperature 0.1 \
    --do_sample \
    --dataset hendrycks/MATH \
    --split test \
    --num_samples 100 \
    --samples_per_question 3 \
    --output_dir ./my_results
```

## Command Line Arguments

### Model Configuration
- `--model_name`: HuggingFace model name or local path (required)
- `--tokenizer_name`: Tokenizer name (defaults to model_name)
- `--device`: Device to use ('auto', 'cuda', 'cpu')

### Generation Parameters
- `--max_length`: Maximum generation length (default: 2048)
- `--temperature`: Sampling temperature (default: 0.1)
- `--do_sample`: Enable sampling for generation

### Evaluation Parameters
- `--dataset`: HuggingFace dataset name (default: 'hendrycks/MATH')
- `--split`: Dataset split to use (default: 'test')
- `--num_samples`: Number of questions to evaluate (default: all)
- `--samples_per_question`: Multiple samples per question (default: 1)
- `--output_dir`: Output directory for results (default: './results')

## Output Format

The tool generates two main output files:

### 1. Detailed Results (`detailed_results.jsonl`)
Each line contains a complete evaluation record:
```json
{
  "question_id": "0_0",
  "question": "What is 2+2?",
  "model_output": "The answer is \\boxed{4}",
  "predicted_answer": "4",
  "ground_truth": "4",
  "is_correct": true,
  "timestamp": "2024-01-01T12:00:00",
  "sample_id": 0
}
```

### 2. Summary Statistics (`summary.json`)
Overall evaluation metrics:
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "dataset": "hendrycks/MATH",
  "split": "test",
  "total_questions": 5000,
  "samples_per_question": 1,
  "total_samples": 5000,
  "correct_samples": 1250,
  "accuracy": 0.25,
  "evaluation_time": "2024-01-01T12:30:00"
}
```

## Supported Datasets

The tool is designed to work with math datasets that have the following structure:
- `problem`: The math question
- `solution`: The ground truth answer (can contain LaTeX)

### Tested Datasets
- `hendrycks/MATH`: Competition mathematics problems
- Custom datasets with the same format

## Customization

### Custom Prompts
Override the `create_prompt` method in the `MathEvaluator` class:

```python
class CustomEvaluator(MathEvaluator):
    def create_prompt(self, question: str) -> str:
        return f"Q: {question}\nA: Let me solve this step by step."
```

### Custom Answer Extraction
Extend the `AnswerExtractor` class for different answer formats:

```python
class CustomAnswerExtractor(AnswerExtractor):
    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        # Custom extraction logic
        pass
```

## Performance Tips

1. **GPU Usage**: Use `--device cuda` for GPU acceleration
2. **Batch Processing**: The tool processes one question at a time for memory efficiency
3. **Memory Management**: Automatic garbage collection between evaluations
4. **Progress Tracking**: Real-time accuracy updates to monitor performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_length` or use CPU
2. **Model Loading Error**: Ensure model name/path is correct
3. **Dataset Loading Error**: Check dataset name and internet connection

### Debug Mode
Enable detailed logging by modifying the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This is an open-source tool. Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is released under the MIT License. See LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@misc{math-evaluator,
  title={Math Problem Evaluation Tool},
  author={Open Source Contributors},
  year={2024},
  url={https://github.com/your-repo/math-evaluator}
}
```