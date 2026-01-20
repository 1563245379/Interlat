#!/usr/bin/env python3
import argparse
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import base framework
from base_data_collector import (
    BaseDataCollectionPipeline,
    CollectedSample
)


class ALFWorldTaskCollector(BaseDataCollectionPipeline):
    """
    Data collector specialized for ALFWorld household task planning

    This collector processes ALFWorld trajectory data and generates high-level
    task plans for household activities without referring to specific objects.
    """

    # Common ALFWorld task types
    TASK_TYPES = [
        'pick_and_place_simple',
        'pick_and_place_with_movable_recep',
        'pick_clean_then_place_in_recep',
        'pick_heat_then_place_in_recep',
        'pick_cool_then_place_in_recep',
        'look_at_obj_in_light'
    ]

    # Available actions in ALFWorld
    ALFWORLD_ACTIONS = [
        "go to {recep}",
        "take {obj} from {recep}",
        "put {obj} in/on {recep}",
        "open {recep}",
        "close {recep}",
        "toggle {obj} {recep}",
        "clean {obj} with {recep}",
        "heat {obj} with {recep}",
        "cool {obj} with {recep}"
    ]

    def __init__(self,
                 model_path: str,
                 dataset_path: str,
                 output_dir: str = "alfworld_task_data",
                 temperature: float = 1.2,
                 max_new_tokens: int = 1500,
                 batch_size: int = 1):
        """
        Initialize the ALFWorld task data collector

        Args:
            model_path: Path to the language model
            dataset_path: Path to ALFWorld dataset JSON file
            output_dir: Directory to save collected data
            temperature: Sampling temperature (higher for creative planning)
            max_new_tokens: Maximum tokens to generate
            batch_size: Processing batch size
        """
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )

        self.dataset_path = Path(dataset_path)
        self.base_task_offset = 0

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"ALFWorld dataset not found: {dataset_path}")

        logging.info(f"Initialized ALFWorldTaskCollector with dataset: {dataset_path}")

    def load_dataset_for_task(self) -> List[Dict]:
        """
        Load and prepare the ALFWorld dataset

        Returns:
            List of conversation trajectory items
        """
        logging.info(f"Loading ALFWorld dataset from: {self.dataset_path}")

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset_items = json.load(f)

            logging.info(f"Loaded {len(dataset_items)} ALFWorld trajectory items")
            return dataset_items

        except Exception as e:
            logging.error(f"Failed to load ALFWorld dataset: {e}")
            raise

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

    def create_alfworld_planning_prompt(self, task_description: str) -> str:
        """
        Create a specialized prompt for ALFWorld task planning

        Args:
            task_description: Description of the household task

        Returns:
            Formatted prompt for plan generation
        """
        action_list = "\n    ".join([f"{i+1}. {action}" for i, action in enumerate(self.ALFWORLD_ACTIONS)])

        prompt_template = f"""Please provide a general plan to solve this household task.

The task is: {task_description}

Available actions you can reference in your plan:
    {action_list}
    where {{obj}} and {{recep}} correspond to objects and receptacles.

Please generate a high-level, step-by-step plan that outlines the general approach to complete this task. Your plan should be abstract and not refer to specific object instances, but rather describe the types of actions needed.

Format your response as:
Step 1: ...
Step 2: ...
Step 3: ...
(continue as needed)"""

        return prompt_template

    def extract_task_from_conversation(self, conversation_item: Dict) -> str:
        """
        Extract the main task description from ALFWorld conversation format

        Args:
            conversation_item: ALFWorld conversation trajectory

        Returns:
            Cleaned task description
        """
        try:
            # ALFWorld format typically has conversations list
            conversations = conversation_item.get('conversations', [])

            # Look for the task description in early conversation turns
            for conv in conversations[:3]:  # Check first few turns
                if conv.get('from') == 'human' or conv.get('from') == 'user':
                    value = conv.get('value', '')
                    if isinstance(value, list) and len(value) > 0:
                        return value[0]
                    elif isinstance(value, str):
                        return value

            # Fallback: look for any task-like content
            for conv in conversations:
                value = conv.get('value', '')
                if isinstance(value, list):
                    for v in value:
                        if 'task' in v.lower() or len(v) > 50:
                            return v
                elif isinstance(value, str) and len(value) > 50:
                    return value

            return "Unknown household task"

        except Exception as e:
            logging.warning(f"Error extracting task description: {e}")
            return "Unknown household task"

    def create_prompt_for_task(self, task_item: Dict) -> str:
        """
        Create prompt for ALFWorld task planning

        Args:
            task_item: Dictionary containing conversation trajectory

        Returns:
            Formatted prompt string
        """
        task_description = self.extract_task_from_conversation(task_item)
        return self.create_alfworld_planning_prompt(task_description)

    def classify_task_type(self, task_description: str, task_item: Dict) -> str:
        """
        Attempt to classify the ALFWorld task type

        Args:
            task_description: Task description text
            task_item: Original trajectory item

        Returns:
            Classified task type
        """
        task_lower = task_description.lower()

        # Simple heuristic classification
        if 'clean' in task_lower:
            return 'pick_clean_then_place_in_recep'
        elif 'heat' in task_lower or 'microwave' in task_lower:
            return 'pick_heat_then_place_in_recep'
        elif 'cool' in task_lower or 'fridge' in task_lower:
            return 'pick_cool_then_place_in_recep'
        elif 'light' in task_lower or 'look at' in task_lower:
            return 'look_at_obj_in_light'
        elif any(word in task_lower for word in ['put', 'place', 'move']):
            if 'drawer' in task_lower or 'cabinet' in task_lower:
                return 'pick_and_place_with_movable_recep'
            else:
                return 'pick_and_place_simple'
        else:
            return 'unknown_task_type'

    def extract_task_metadata(self, task_item: Dict, task_index: int) -> Dict[str, Any]:
        """
        Extract metadata from an ALFWorld task item

        Args:
            task_item: Original trajectory item
            task_index: Local index of the task

        Returns:
            Dictionary containing task metadata
        """
        task_description = self.extract_task_from_conversation(task_item)
        task_type = self.classify_task_type(task_description, task_item)

        # Estimate complexity based on conversation length
        conversations = task_item.get('conversations', [])
        complexity = 'simple' if len(conversations) <= 5 else 'complex'

        return {
            "type": task_type,
            "level": complexity,
            "conversation_length": len(conversations),
            "local_index": task_index,
            "global_index": self.base_task_offset + task_index,
            "original_id": task_item.get('id', f'alfworld_{task_index}')
        }

    def create_generation_metadata(self, task_item: Dict) -> Dict[str, Any]:
        """
        Create metadata about the generation process

        Args:
            task_item: Original trajectory item

        Returns:
            Dictionary containing generation metadata
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "model_path": self.model_manager.model_path,
            "dataset_type": "alfworld",
            "dataset_path": str(self.dataset_path)
        }

    def process_task_batch(self, batch_items: List[Dict], batch_idx: int) -> int:
        """
        Process a batch of ALFWorld task planning tasks

        Args:
            batch_items: List of conversation trajectory items
            batch_idx: Index of current batch

        Returns:
            Number of successfully processed items
        """
        processed_count = 0

        for local_idx, task_item in enumerate(batch_items):
            try:
                # Extract task information
                task_description = self.extract_task_from_conversation(task_item)

                # Create unique task ID
                global_task_idx = self.base_task_offset + batch_idx * self.batch_size + local_idx + 1
                task_id = task_item.get('id', f'ALF_{global_task_idx:06d}')

                # Ensure task_id is string and unique
                if not isinstance(task_id, str):
                    task_id = f'ALF_{global_task_idx:06d}'

                # Create prompt and generate response
                prompt = self.create_prompt_for_task(task_item)

                logging.debug(f"Processing {task_id}: {task_description[:100]}...")

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
                    task_content=task_description,
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
        description="ALFWorld Task Data Collector",
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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to ALFWorld dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="alfworld_task_data",
        help="Directory to save collected data"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature for text generation (higher for creativity)"
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
            logging.FileHandler(f"alfworld_task_collection.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting ALFWorld Task Data Collection")
    logging.info(f"Arguments: {vars(args)}")

    try:
        # Initialize collector
        collector = ALFWorldTaskCollector(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )

        # Run collection pipeline
        collector.initialize()
        collector.run_collection()
        collector.finalize()

        logging.info("ALFWorld task data collection completed successfully!")

    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()