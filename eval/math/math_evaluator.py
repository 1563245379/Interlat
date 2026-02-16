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

from core_training.hidden_model.custom_model import ModelWithInsertedHiddenState

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
                 do_sample: bool = True,
                 mode: str = "vanilla",
                 hidden_data: Optional[str] = None,
                 question_field: Optional[str] = None,
                 solution_field: Optional[str] = None,
                 plan_field: str = "plan",
                 hidden_field: str = "hidden_state",
                 task_id_field: str = "task_id",
                 max_hidden_length: int = 800,
                 insert_position: str = "auto",
                 disable_mix: bool = False):
        """
        Initialize the evaluator

        Args:
            model_name_or_path: Path to model or HuggingFace model name
            tokenizer_name: Tokenizer name (defaults to model_name_or_path)
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            mode: 'vanilla' (plain LM), 'hidden' (latent-aware), or 'plan' (plan text only)
            hidden_data: Dataset name/path containing hidden states + plans for hidden/plan modes
            question_field: Optional override for question field name
            solution_field: Optional override for solution field name
            plan_field: Field name storing plan text
            hidden_field: Field name storing hidden states
            task_id_field: Field name storing unique ids (for logging)
            max_hidden_length: Truncate hidden states to this length
            insert_position: 'auto' uses end of prompt tokens, or an integer index
            disable_mix: Force-disable hidden/plan mixing inside the wrapper
        """
        self.model_name = model_name_or_path
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        self.mode = mode
        self.hidden_data = hidden_data
        self.question_field = question_field
        self.solution_field = solution_field
        self.plan_field = plan_field
        self.hidden_field = hidden_field
        self.task_id_field = task_id_field
        self.max_hidden_length = max_hidden_length
        self.insert_position = insert_position
        self.disable_mix = disable_mix or (mode == "plan")

        logger.info(f"Loading model: {model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Eval mode: {self.mode}")

        # Load tokenizer & model (wrapper-aware)
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            tokenizer_name=tokenizer_name or model_name_or_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )

        self.model.eval()
        self.answer_extractor = AnswerExtractor()

    def _setup_device(self, device: str) -> str:
        """Setup computation device (resolves 'auto' -> 'cuda'|'cpu')."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _maybe_add_special_tokens(self, tokenizer):
        """Ensure <bop>/<eop> exist for plan/hidden modes."""
        if self.mode == "vanilla":
            return tokenizer

        to_add = []
        for tok in ["<bop>", "<eop>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                to_add.append(tok)

        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        return tokenizer

    def _load_model_and_tokenizer(self, model_name_or_path: str, tokenizer_name: str, torch_dtype: torch.dtype):
        """Load tokenizer and model; wrap with hidden-state module when available."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "right"
        tokenizer = self._maybe_add_special_tokens(tokenizer)

        attn_impl = "flash_attention_2" if self.device.startswith("cuda") else None
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map=None,
        )
        base_model.to(self.device)

        hidden_mha_path = os.path.join(model_name_or_path, "hidden_mha_state.pt")
        prep_config_path = os.path.join(model_name_or_path, "prepended_config.json")

        use_wrapper = self.mode in ("hidden", "plan") and os.path.exists(hidden_mha_path)
        if not use_wrapper:
            logger.info("Using base model (no hidden-state wrapper detected).")
            return base_model, tokenizer

        prep_conf = {
            "prepended_length": 1000,
            "hidden_size": getattr(base_model.config, "hidden_size", None),
            "plan_similarity_weight": 0.0,
            "random_contrast_weight": 0.0,
        }
        if os.path.exists(prep_config_path):
            try:
                with open(prep_config_path, "r") as f:
                    prep_conf.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read prepended_config.json: {e}")

        hidden_size = prep_conf.get("hidden_size") or getattr(base_model.config, "hidden_size", None)
        model = ModelWithInsertedHiddenState(
            base_model=base_model,
            prepended_length=int(prep_conf.get("prepended_length", 1000)),
            hidden_size=int(hidden_size),
            prepended_learnable=bool(prep_conf.get("prepended_learnable", False)),
            plan_similarity_weight=float(prep_conf.get("plan_similarity_weight", 0.0)),
            random_contrast_weight=float(prep_conf.get("random_contrast_weight", 0.0)),
            prepended_input_dim=prep_conf.get("hidden_size", hidden_size),
        )

        try:
            state = torch.load(hidden_mha_path, map_location="cpu")
            model.hidden_mha.load_state_dict(state["hidden_mha"])
            model.pre_ln.load_state_dict(state["pre_ln"])
            model.post_ln.load_state_dict(state["post_ln"])
            if "adaptive_proj" in state:
                model.adaptive_proj.load_state_dict(state["adaptive_proj"])
            if "scale" in state and hasattr(model, "scale"):
                with torch.no_grad():
                    model.scale.fill_(float(state["scale"]))
            if "output_scale" in state and hasattr(model, "output_scale"):
                with torch.no_grad():
                    model.output_scale.fill_(float(state["output_scale"]))
            logger.info("Loaded hidden-state wrapper weights from hidden_mha_state.pt")
        except Exception as e:
            logger.warning(f"Failed to load hidden-state weights, using base model: {e}")
            return base_model, tokenizer

        model.tokenizer = tokenizer
        model.to(self.device)
        return model, tokenizer

    def _resolve_fields(self, dataset) -> Tuple[str, str]:
        """Resolve question/solution field names with sensible fallbacks."""
        columns = set(dataset.column_names)
        q_field = self.question_field or ("task" if "task" in columns else "problem")
        s_field = self.solution_field or ("task_solution" if "task_solution" in columns else "solution")
        if q_field not in columns:
            raise KeyError(f"Question field '{q_field}' not found in dataset columns {columns}")
        if s_field not in columns:
            logger.warning(f"Solution field '{s_field}' missing; accuracy will be zero.")
        return q_field, s_field

    def _load_eval_dataset(self, dataset_name: str, split: str, num_samples: int = None):
        """Load evaluation dataset; switch to hidden_data when requested."""
        target = self.hidden_data if self.mode in ("hidden", "plan") and self.hidden_data else dataset_name
        logger.info(f"Loading dataset: {target} ({split})")
        dataset = load_dataset(target, split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def _compute_insert_pos(self, attention_mask: torch.Tensor) -> int:
        """Compute insertion index based on non-pad tokens or a fixed override."""
        if isinstance(self.insert_position, str) and self.insert_position != "auto":
            try:
                return int(self.insert_position)
            except ValueError:
                logger.warning(f"Invalid insert_position={self.insert_position}, falling back to auto")
        if not isinstance(attention_mask, torch.Tensor):
            return 0
        return int(attention_mask[0].sum().item())

    def _prepare_generation_kwargs(self, prompt: str, plan_text: Optional[str], hidden_state: Optional[Any]):
        """Prepare inputs/model kwargs for different modes."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.mode == "vanilla":
            return inputs, {}

        human_pos = self._compute_insert_pos(inputs.get("attention_mask"))
        plan_ids = self.tokenizer(plan_text or "", add_special_tokens=False).input_ids

        model_kwargs: Dict[str, Any] = {
            "plans": [plan_ids],
            "human_end_positions": torch.tensor([human_pos], device=self.device),
            "disable_mix": self.disable_mix,
        }

        if self.mode == "hidden" and hidden_state is not None:
            hs = torch.tensor(hidden_state)
            if hs.dim() == 3 and hs.size(0) == 1:
                hs = hs.squeeze(0)
            if self.max_hidden_length and hs.size(0) > self.max_hidden_length:
                hs = hs[:self.max_hidden_length]
            hs = hs.to(self.device, dtype=self.model.get_input_embeddings().weight.dtype)
            model_kwargs["prepended_hidden_states"] = [hs]
        else:
            model_kwargs["prepended_hidden_states"] = None

        return inputs, model_kwargs

    def create_prompt(self, question: str, plan_text: Optional[str] = None) -> str:
        """Create evaluation prompt; optionally include plan text."""
        if plan_text and self.mode in ("hidden", "plan"):
            prompt = (
                "Solve the following math problem step by step. "
                "A high-level plan is provided; follow it but verify each step. "
                "Provide the final answer in \\boxed{} .\n\n"
                f"Plan:\n{plan_text}\n\n"
                f"Problem: {question}\n\n"
                "Solution:"
            )
        else:
            prompt = (
                "Solve the following math problem step by step. "
                "Show your work clearly and put your final answer in \\boxed{} .\n\n"
                f"Problem: {question}\n\n"
                "Solution:"
            )
        return prompt

    def generate_response(self, prompt: str, plan_text: Optional[str] = None, hidden_state: Optional[Any] = None) -> str:
        """Generate model response with optional plan/hidden inputs."""
        inputs, model_kwargs = self._prepare_generation_kwargs(prompt, plan_text, hidden_state)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **model_kwargs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (remove input prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
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
        target_dataset = self.hidden_data if self.mode in ("hidden", "plan") and self.hidden_data else dataset_name
        try:
            dataset = self._load_eval_dataset(target_dataset, split, num_samples)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        q_field, s_field = self._resolve_fields(dataset)

        logger.info(f"Evaluating {len(dataset)} questions with {samples_per_question} samples each")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        results = []
        correct_count = 0
        total_count = 0

        # Setup progress bar
        pbar = tqdm(total=len(dataset) * samples_per_question, desc="Evaluating")

        for idx, item in enumerate(dataset):
            question = item[q_field]
            ground_truth = item.get(s_field, "") or ""
            plan_text = item.get(self.plan_field, "") if self.mode in ("hidden", "plan") else ""
            hidden_state = item.get(self.hidden_field, None) if self.mode == "hidden" else None
            question_id = str(item.get(self.task_id_field, idx))

            if self.mode == "hidden" and hidden_state is None:
                logger.warning(f"[Skip] No hidden_state for sample {question_id}, skipping.")
                pbar.update(samples_per_question)
                continue

            question_results = []

            for sample_id in range(samples_per_question):
                try:
                    # Generate response
                    prompt = self.create_prompt(question, plan_text)
                    response = self.generate_response(prompt, plan_text, hidden_state)

                    # Evaluate answer
                    is_correct, pred_answer, norm_gt = self.answer_extractor.evaluate_answer(
                        response, ground_truth
                    )

                    # Create result record
                    result = EvaluationResult(
                        question_id=f"{question_id}_{sample_id}",
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
            "dataset": target_dataset,
            "split": split,
            "total_questions": len(dataset),
            "samples_per_question": samples_per_question,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": accuracy,
            "evaluation_time": datetime.now().isoformat(),
            "mode": self.mode,
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

    # Plan / hidden-state options
    parser.add_argument("--mode", type=str, default="vanilla",
                        choices=["vanilla", "hidden", "plan"],
                        help="Evaluation mode: vanilla LM, hidden-state receiver, or plan-text input")
    parser.add_argument("--hidden_data", type=str, default=None,
                        help="Dataset containing plan/hidden_state fields (HF name or local path)")
    parser.add_argument("--question_field", type=str, default=None,
                        help="Override question field name")
    parser.add_argument("--solution_field", type=str, default=None,
                        help="Override solution field name")
    parser.add_argument("--plan_field", type=str, default="plan",
                        help="Field name for plan text")
    parser.add_argument("--hidden_field", type=str, default="hidden_state",
                        help="Field name for hidden states")
    parser.add_argument("--task_id_field", type=str, default="task_id",
                        help="Field name for sample id (used in logs)")
    parser.add_argument("--max_hidden_length", type=int, default=800,
                        help="Truncate hidden states to this length")
    parser.add_argument("--insert_position", type=str, default="auto",
                        help="Insert position for plan/hidden (auto=after prompt tokens or integer index)")
    parser.add_argument("--disable_mix", action="store_true",
                        help="Disable mixing hidden states with plan embeddings inside wrapper")

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
        do_sample=args.do_sample,
        mode=args.mode,
        hidden_data=args.hidden_data,
        question_field=args.question_field,
        solution_field=args.solution_field,
        plan_field=args.plan_field,
        hidden_field=args.hidden_field,
        task_id_field=args.task_id_field,
        max_hidden_length=args.max_hidden_length,
        insert_position=args.insert_position,
        disable_mix=args.disable_mix,
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