# Copilot Instructions for Interlat

## Project shape
- Interlat enables latent-only agent communication. Core code lives in `core_training/` (training), `compression_training/` (latent distillation), `data_collection/` (hidden-state extraction), and `eval/` (benchmarks: ALFWorld, MATH).
- Hidden-state models wrap a base HF causal LM with `hidden_model/custom_model.py` and `ModelWithInsertedHiddenState`; plan/hidden state handling is orchestrated in `core_training/train.py` with dataset prep in `core_training/data_processor.py`.

## Key workflows
- Data collection: use `scripts/collect_alfworld.sh` or `scripts/collect_math.sh`, or `data_collection/collect_data.py [alfworld|math]` to produce hidden-state datasets.
- Training (latent-enabled): `core_training/train.py` via `scripts/train_model.sh` or direct `python core_training/train.py --model_name_or_path ... --data_path ... --hidden_data_path ... --output_dir ...`. The script sets NCCL/Triton envs and delays CUDA init per-rank; do not move the model to GPU manually.
- Compression: `compression_training/compress.py` or `scripts/train_compression.sh` for teacher-student latent distillation (K controls latent length).
- Evaluation: ALFWorld via `eval/alfworld/eval_agent/main.py`; MATH via `eval/math/math_evaluator.py`. Math evaluator expects boxed answers and saves JSONL + summary.

## Conventions and patterns
- Precision auto-detects in `train.py` (`bf16` favored on Ampere+); training args override evaluation/save strategies to step-based; seeds are set after local_rank to avoid CUDA contention.
- Tokenizer padding: right-padding enforced in `train.py`; pad token defaults to eos when missing; special tokens are added when positional tracking is enabled.
- Loss callbacks: custom callbacks in `core_training/callbacks.py` (optimizer debug, MHA state save, gradient logging, loss recorder); leave them wired unless you know the implications.
- Hidden-state insertion: `prepended_length` and `prepend_position` control whether plan/hidden vectors are prepended; position-tracking tokens only added when `prepended_length > 0` and position mode is `first_human`.
- Model loading: uses `AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)` plus a safety `safe_to_bfloat16` pass; avoid reconverting dtypes to prevent aliasing.

## Integration notes
- External deps: HuggingFace Transformers/Tokenizers, PyTorch (NCCL heavy), datasets, tqdm. Set `HF_HOME`/`TRANSFORMERS_CACHE`/`HF_DATASETS_CACHE` for speed.
- Distributed quirks: `train.py` patches PyTorch 2.4 NCCL eager connect; staggered CUDA init per-rank. Honor these patches when adding distributed entrypoints.
- Evaluation IO: math evaluator expects dataset fields `problem` and `solution`; results stored under `output_dir` as `detailed_results.jsonl` and `summary.json`.

## When adding new code
- Reuse argument dataclasses in `core_training/arguments.py`; keep new CLI flags consistent with existing ones.
- Prefer extending data collators or processors in `core_training/data_processor.py` instead of ad-hoc preprocessing in scripts.
- For new callbacks/metrics, follow patterns in `core_training/callbacks.py` and register in the Trainer callback list in `train.py`.
- Keep NCCL/precision/device handling centralized in `train.py`; avoid early CUDA context creation and manual `.to(device)` before Trainer setup.
