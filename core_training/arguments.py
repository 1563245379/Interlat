from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        # default="Qwen/Qwen2.5-7B"
        # default="Qwen/Qwen2.5-0.5B"
        # default="meta-llama/Meta-Llama-3.1-8B"
        default = "Interlat_preview/models/Qwen2.5-0.5B"
        )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    # Prepended hidden state arguments
    prepended_length: int = field(
        default=800,
        metadata={"help": "Length of prepended hidden states"}
    )
    prepended_hidden_state_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to prepended hidden state file (.pt or .npy)"}
    )
    prepended_learnable: bool = field(
        default=False,  # Changed to default False
        metadata={"help": "Whether prepended hidden state is learnable during training"}
    )
    prepend_position: str = field(
        default="first_human",
        metadata={"help": "Where to prepend: 'start' or 'first_human'"}
    )
    plan_similarity_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for plan similarity loss"}
    )
    random_contrast_weight: float = field(
        default=2.0,
        metadata={"help": "Weight for random contrast loss"}
    )
    PE_fuse: bool = field(
        default=False,
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default='Interlat_preview/datasets/alfworld_sft.json',
        metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    eval_ratio: float = field(
        default=0.01,
        metadata={"help": "When eval_data_path is None, take this ratio from train as eval set."}
    )
    hidden_data: str = field(
        # default='your_dataset'
        # default="recommend_gul_mdl/alfword_hidden_state"
        default='pailitao_v100/alfworld_qwen05B_hidden'
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Basics
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=3800)

    # DataLoader
    dataloader_num_workers: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    eval_steps: int = field(
        default=500,
        metadata={"help": "Run evaluation every X steps."}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X steps."}
    )
    logging_steps: int = field(
        default=20,
        metadata={"help": "Log every X steps."}
    )
    save_total_limit: int = field(
        default=20,
        metadata={"help": "Max number of checkpoints to keep."}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric name used to select best checkpoint."}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "True if a higher metric is better."}
    )

    # Early stopping (new)
    early_stopping_patience: int = field(
        default=100,
        metadata={"help": "Number of eval rounds with no improvement before stopping."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimal improvement to be considered as better."}
    )
