#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Math Problem Evaluation Tool
============================

An open-source tool for evaluating language models on mathematical reasoning tasks.
This tool supports multiple evaluation modes and can work with various model architectures.

Features:
- Extract and evaluate LaTeX-formatted answers
- Support for multiple sampling per question
- Detailed logging and reporting
- Extensible model wrapper system
"""

import os
import sys
import json
import time
import gc
import warnings
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import argparse
import hashlib
import re
from dataclasses import dataclass, asdict

import numpy as np
import torch
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("math_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results"""
    question_id: str
    question: str
    model_output: str
    predicted_answer: Optional[str]
    ground_truth: str
    is_correct: bool
    timestamp: str
    sample_id: int = 0

class AnswerExtractor:
    """Extract and normalize answers from model outputs"""

    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        """Extract the LAST \\boxed{...} or \\fbox{...} from model output."""
        # Handle nested braces by counting brace levels
        def extract_balanced_braces(text, start_pos):
            """Extract content within balanced braces starting at start_pos"""
            if start_pos >= len(text) or text[start_pos] != '{':
                return None

            brace_count = 1
            content_start = start_pos + 1
            pos = content_start

            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1

            if brace_count == 0:
                return text[content_start:pos-1]
            return None

        # Find all \boxed and \fbox patterns
        patterns = [r"\\boxed", r"\\fbox"]
        matches = []

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start = match.end()
                if start < len(text) and text[start] == '{':
                    content = extract_balanced_braces(text, start)
                    if content is not None:
                        matches.append((match.start(), content))

        # Return the last match
        if matches:
            return matches[-1][1].strip()
        return None

    @staticmethod
    def normalize(ans: str) -> Optional[str]:
        """Normalize answer format"""
        if ans is None:
            return None
        # Light normalization: remove whitespace, LaTeX spacing commands
        a = ans.strip()
        a = re.sub(r"\\\\,", "", a)        # remove LaTeX thin spaces
        a = re.sub(r"\s+", "", a)          # drop whitespaces
        return a

    @classmethod
    def evaluate_answer(cls, model_output: str, ground_truth: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Evaluate if model output matches ground truth

        Returns:
            Tuple of (is_correct, predicted_answer, normalized_ground_truth)
        """
        pred = cls.normalize(cls.extract_boxed_answer(model_output))

        # Handle cases where ground truth might not have \boxed{}
        gold_extracted = cls.extract_boxed_answer(ground_truth)
        if gold_extracted is None:
            # If no boxed answer in ground truth, use the ground truth as is
            gold = cls.normalize(ground_truth)
        else:
            gold = cls.normalize(gold_extracted)

        is_correct = (pred is not None) and (pred == gold)
        return is_correct, pred, gold

class MathEvaluator:
    """Main evaluation class"""

    def __init__(self,
                 model_name_or_path: str,
                 tokenizer_name: Optional[str] = None,
                 device: str = "auto",
                 max_length: int = 2048,
                 temperature: float = 0.1,
                 do_sample: bool = True):
        """
        Initialize the evaluator

        Args:
            model_name_or_path: Path to model or HuggingFace model name
            tokenizer_name: Tokenizer name (defaults to model_name_or_path)
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        """
        self.model_name = model_name_or_path
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample

        logger.info(f"Loading model: {model_name_or_path}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        tokenizer_name = tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if device == "auto" else None,
            trust_remote_code=True
        )

        if device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.answer_extractor = AnswerExtractor()

    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def create_prompt(self, question: str) -> str:
        """
        Create evaluation prompt for a math question
        Override this method for different prompt formats
        """
        prompt = f"""Solve the following math problem step by step. Show your work clearly and put your final answer in \\boxed{{}}.

Problem: {question}

Solution:"""
        return prompt

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response (remove input prompt)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def evaluate_dataset(self,
                        dataset_name: str = "hendrycks/MATH",
                        split: str = "test",
                        num_samples: int = None,
                        samples_per_question: int = 1,
                        output_dir: str = "./results") -> Dict[str, Any]:
        """
        Evaluate model on a math dataset

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            num_samples: Number of questions to evaluate (None for all)
            samples_per_question: Number of samples per question
            output_dir: Output directory for results

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Loading dataset: {dataset_name}")

        try:
            dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        logger.info(f"Evaluating {len(dataset)} questions with {samples_per_question} samples each")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        results = []
        correct_count = 0
        total_count = 0

        # Setup progress bar
        pbar = tqdm(total=len(dataset) * samples_per_question, desc="Evaluating")

        for idx, item in enumerate(dataset):
            question = item["problem"]
            ground_truth = item["solution"]

            question_results = []

            for sample_id in range(samples_per_question):
                try:
                    # Generate response
                    prompt = self.create_prompt(question)
                    response = self.generate_response(prompt)

                    # Evaluate answer
                    is_correct, pred_answer, norm_gt = self.answer_extractor.evaluate_answer(
                        response, ground_truth
                    )

                    # Create result record
                    result = EvaluationResult(
                        question_id=f"{idx}_{sample_id}",
                        question=question,
                        model_output=response,
                        predicted_answer=pred_answer,
                        ground_truth=norm_gt,
                        is_correct=is_correct,
                        timestamp=datetime.now().isoformat(),
                        sample_id=sample_id
                    )

                    results.append(result)
                    question_results.append(is_correct)
                    total_count += 1

                    if is_correct:
                        correct_count += 1

                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        "Accuracy": f"{correct_count/total_count:.3f}",
                        "Correct": correct_count,
                        "Total": total_count
                    })

                except Exception as e:
                    logger.error(f"Error processing question {idx}, sample {sample_id}: {e}")
                    pbar.update(1)
                    continue

        pbar.close()

        # Calculate final statistics
        accuracy = correct_count / total_count if total_count > 0 else 0

        eval_summary = {
            "model_name": self.model_name,
            "dataset": dataset_name,
            "split": split,
            "total_questions": len(dataset),
            "samples_per_question": samples_per_question,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": accuracy,
            "evaluation_time": datetime.now().isoformat()
        }

        # Save results
        self._save_results(results, eval_summary, output_dir)

        logger.info(f"Evaluation completed: {correct_count}/{total_count} = {accuracy:.3f}")
        return eval_summary

    def _save_results(self, results: List[EvaluationResult], summary: Dict[str, Any], output_dir: str):
        """Save evaluation results to files"""

        # Save detailed results as JSONL
        results_file = os.path.join(output_dir, "detailed_results.jsonl")
        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

        # Save summary
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_dir}")

def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Math Problem Evaluation Tool")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                       help="Tokenizer name (defaults to model_name)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")

    # Generation arguments
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling for generation")

    # Evaluation arguments
    parser.add_argument("--dataset", type=str, default="hendrycks/MATH",
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of questions to evaluate")
    parser.add_argument("--samples_per_question", type=int, default=1,
                       help="Number of samples per question")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MathEvaluator(
        model_name_or_path=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        do_sample=args.do_sample
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        samples_per_question=args.samples_per_question,
        output_dir=args.output_dir
    )

    print(f"\nEvaluation Results:")
    print(f"Model: {results['model_name']}")
    print(f"Dataset: {results['dataset']} ({results['split']})")
    print(f"Accuracy: {results['accuracy']:.3f} ({results['correct_samples']}/{results['total_samples']})")

if __name__ == "__main__":
    main()