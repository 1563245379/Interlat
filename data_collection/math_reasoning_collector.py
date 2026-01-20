#!/usr/bin/env python3
import argparse
import logging
from typing import List, Dict, Any, Optional
from textwrap import dedent

from datasets import load_dataset, concatenate_datasets

# Import base framework
from base_data_collector import (
    BaseDataCollectionPipeline,
    CollectedSample
)


class MathReasoningCollector(BaseDataCollectionPipeline):
    """
    Data collector specialized for mathematical reasoning tasks

    This collector processes the MATH dataset and generates high-level solution plans
    without performing concrete calculations, focusing on reasoning patterns.
    """

    # MATH dataset subjects
    MATH_SUBJECTS = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    def __init__(self,
                 model_path: str,
                 output_dir: str = "math_reasoning_data",
                 temperature: float = 0.7,
                 max_new_tokens: int = 1500,
                 batch_size: int = 1,
                 dataset_split: str = "train",
                 subjects: Optional[List[str]] = None):
        """
        Initialize the mathematical reasoning data collector

        Args:
            model_path: Path to the language model
            output_dir: Directory to save collected data
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens to generate
            batch_size: Processing batch size
            dataset_split: Which split to use ('train', 'test')
            subjects: List of math subjects to include (None for all)
        """
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )

        self.dataset_split = dataset_split
        self.subjects = subjects or self.MATH_SUBJECTS
        self.base_task_offset = 0  # For unique task IDs across processes

        logging.info(f"Initialized MathReasoningCollector for subjects: {self.subjects}")

    def load_dataset_for_task(self) -> List[Dict]:
        """
        Load and prepare the MATH dataset

        Returns:
            List of dataset items with mathematical problems
        """
        logging.info(f"Loading MATH dataset ({self.dataset_split} split)...")

        # Load individual subject datasets and concatenate
        subject_datasets = []
        for subject in self.subjects:
            try:
                subject_dataset = load_dataset(
                    'EleutherAI/hendrycks_math',
                    subject,
                    split=self.dataset_split
                )
                subject_datasets.append(subject_dataset)
                logging.info(f"Loaded {len(subject_dataset)} samples from {subject}")
            except Exception as e:
                logging.error(f"Failed to load subject {subject}: {e}")
                continue

        if not subject_datasets:
            raise RuntimeError("No subjects could be loaded from MATH dataset")

        # Concatenate all subjects
        full_dataset = concatenate_datasets(subject_datasets)
        logging.info(f"Total dataset size: {len(full_dataset)} problems")

        # Convert to list for easier processing
        return [item for item in full_dataset]

    def distribute_dataset(self) -> List[Dict]:
        """
        Distribute dataset across processes with proper task ID offsetting

        Returns:
            Local dataset portion for this process
        """
        my_dataset = super().distribute_dataset()

        # Calculate base offset for unique task IDs
        if self.world_size > 1:
            total_samples = len(self.dataset)
            samples_per_process = len(my_dataset)
            self.base_task_offset = self.rank * samples_per_process
        else:
            self.base_task_offset = 0

        logging.info(f"Task ID offset for rank {self.rank}: {self.base_task_offset}")
        return my_dataset

    def create_math_planning_prompt(self, problem: str) -> str:
        """
        Create a specialized prompt for mathematical planning

        Args:
            problem: The mathematical problem statement

        Returns:
            Formatted prompt for plan generation
        """
        prompt_template = dedent("""
        You are a mathematical problem-solving planner.

        When you receive a math problem (Question), your task is to output a **high-level solution plan** (Plan) that guides another model to solve the problem in detail.

        ### IMPORTANT RULES:

        1. **You must provide a plan, not the final answer. Do NOT perform any concrete calculations or substitute values.**
        2. **Your plan should stay abstract and general. It must not depend on a single specific result or commit to one unique numerical outcome.**
           - Allowed: "Apply the quadratic formula to solve for x", "Consider using the Pythagorean theorem to relate the sides"
           - Forbidden: "The answer is 42", "Substitute x = 7 to get 3/5"
        3. **Do NOT copy or reference any existing solution steps from the dataset. Your plan must be independently generated and must not conflict with any official solution.**
        4. **Use the following output format exactly:**

        [Plan]
        Step 1: ...
        Step 2: ...
        Step 3: ...
        (Do NOT include the final answer or any numerical results)

        ---

        Question:
        {question}

        Now generate the Plan based on the rules above.
        """).strip()

        return prompt_template.format(question=problem)

    def create_prompt_for_task(self, task_item: Dict) -> str:
        """
        Create prompt for mathematical reasoning task

        Args:
            task_item: Dictionary containing problem data

        Returns:
            Formatted prompt string
        """
        problem = task_item['problem']
        return self.create_math_planning_prompt(problem)

    def extract_task_metadata(self, task_item: Dict, task_index: int) -> Dict[str, Any]:
        """
        Extract metadata from a math task item

        Args:
            task_item: Original dataset item
            task_index: Local index of the task

        Returns:
            Dictionary containing task metadata
        """
        return {
            "type": task_item.get('type', 'unknown'),
            "level": task_item.get('level', 'unknown'),
            "subject": task_item.get('subject', 'unknown'),
            "local_index": task_index,
            "global_index": self.base_task_offset + task_index
        }

    def create_generation_metadata(self, task_item: Dict) -> Dict[str, Any]:
        """
        Create metadata about the generation process

        Args:
            task_item: Original dataset item

        Returns:
            Dictionary containing generation metadata
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "model_path": self.model_manager.model_path,
            "dataset_split": self.dataset_split,
            "original_solution_available": 'solution' in task_item
        }

    def process_task_batch(self, batch_items: List[Dict], batch_idx: int) -> int:
        """
        Process a batch of mathematical reasoning tasks

        Args:
            batch_items: List of task items to process
            batch_idx: Index of current batch

        Returns:
            Number of successfully processed items
        """
        processed_count = 0

        for local_idx, task_item in enumerate(batch_items):
            try:
                # Extract task information
                problem = task_item['problem']
                solution = task_item.get('solution', '')

                # Create unique task ID
                global_task_idx = self.base_task_offset + batch_idx * self.batch_size + local_idx + 1
                task_id = f'MATH_{global_task_idx:06d}'

                # Create prompt and generate response
                prompt = self.create_prompt_for_task(task_item)

                logging.debug(f"Processing {task_id}: {problem[:100]}...")

                # Generate plan with hidden states
                generated_plan, hidden_states = self.model_manager.generate_with_hidden_states(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )

                # Log generation result
                logging.info(f"Generated plan for {task_id}: {generated_plan[:100]}...")

                # Create metadata
                task_metadata = self.extract_task_metadata(task_item, local_idx)
                generation_metadata = self.create_generation_metadata(task_item)

                # Create collected sample
                sample = CollectedSample(
                    task_id=task_id,
                    task_content=problem,
                    plan_output=generated_plan,
                    hidden_states=hidden_states,
                    task_metadata=task_metadata,
                    generation_metadata=generation_metadata
                )

                # Add to storage
                self.storage_manager.add_sample(sample)
                processed_count += 1

                # Log progress periodically
                if processed_count % 10 == 0:
                    logging.info(f"Rank {self.rank} processed {processed_count} tasks in batch {batch_idx}")

            except Exception as e:
                logging.error(f"Error processing task in batch {batch_idx}, item {local_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return processed_count


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Mathematical Reasoning Data Collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the language model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="math_reasoning_data",
        help="Directory to save collected data"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs='+',
        default=None,
        choices=MathReasoningCollector.MATH_SUBJECTS,
        help="Math subjects to include (default: all subjects)"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1500,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Processing batch size"
    )

    # Distributed training arguments (handled by the base class)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"math_reasoning_collection.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting Mathematical Reasoning Data Collection")
    logging.info(f"Arguments: {vars(args)}")

    try:
        # Initialize collector
        collector = MathReasoningCollector(
            model_path=args.model_path,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            dataset_split=args.dataset_split,
            subjects=args.subjects
        )

        # Run collection pipeline
        collector.initialize()
        collector.run_collection()
        collector.finalize()

        logging.info("Mathematical reasoning data collection completed successfully!")

    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()