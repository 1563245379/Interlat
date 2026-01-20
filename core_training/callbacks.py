import os
import random
import re
import math
import torch
import torch.nn as nn
from transformers import TrainerCallback
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt  # Only needed by LossRecorderCallback

IGNORE = -100


class OptimizerDebugCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("🔍 [Optimizer Debug] on_train_begin called")
        print(f"🔍 [Debug] kwargs keys: {list(kwargs.keys())}")

        trainer = kwargs.get("trainer")
        print(f"🔍 [Debug] trainer exists: {trainer is not None}")

        if trainer:
            print(f"🔍 [Debug] trainer.model exists: {hasattr(trainer, 'model') and trainer.model is not None}")
            print(f"🔍 [Debug] trainer.optimizer exists: {hasattr(trainer, 'optimizer')}")
            print(f"🔍 [Debug] trainer.optimizer value: {getattr(trainer, 'optimizer', 'N/A')}")

        # 🔧 Force recreation of optimizer (at training start)
        model = trainer.model if trainer else None
        print(f"🔍 [Debug] got model: {model is not None}")

        if model:
            print("🔧 [Force Fix] Recreating optimizer at training start...")

            # Collect all trainable parameters
            trainable_params = []
            print("🔧 [Force Fix] Start collecting trainable parameters...")

            try:
                param_count = 0
                for name, param in model.named_parameters():
                    param_count += 1
                    if param.requires_grad:
                        trainable_params.append(param)
                        if len(trainable_params) <= 5:  # Only print the first 5
                            print(f"    Found trainable param: {name}")

                print(f"🔧 [Force Fix] Scanned {param_count} params, found {len(trainable_params)} trainable params")
            except Exception as e:
                print(f"❗️ [Error] Failed when collecting params: {e}")

            if trainable_params and trainer:
                print(f"🔧 [Force Fix] About to create optimizer, param count: {len(trainable_params)}")
                try:
                    # Force recreate optimizer
                    import torch.optim as optim
                    trainer.optimizer = optim.AdamW(
                        trainable_params,
                        lr=args.learning_rate,
                        weight_decay=getattr(args, 'weight_decay', 0.01),
                    )
                    print(f"✅ [Force Fix] Optimizer recreated with {len(trainable_params)} params")

                    # Verify optimizer
                    total_opt_params = sum(len(group['params']) for group in trainer.optimizer.param_groups)
                    print(f"✅ Verified: optimizer now contains {total_opt_params} params")
                except Exception as e:
                    print(f"❗️ [Error] Failed when creating optimizer: {e}")
            else:
                print(
                    f"❗️ [Force Fix] Cannot create optimizer - trainable_params: {len(trainable_params)}, trainer: {trainer is not None}")
        else:
            print("❗️ [Force Fix] Cannot access model")

        if trainer and trainer.optimizer:
            print("🔍 [Optimizer Debug] Checking optimizer params...")
            total_opt_params = sum(len(group['params']) for group in trainer.optimizer.param_groups)
            print(f"Number of params in optimizer: {total_opt_params}")

            # Check which params are in optimizer
            opt_param_ids = set()
            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    opt_param_ids.add(id(p))

            # Check whether model params are in optimizer
            model = trainer.model
            missing_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and id(param) not in opt_param_ids:
                    missing_params.append(name)

            if missing_params:
                print(f"❗️ These params are not included in optimizer: {missing_params[:10]}...")  # Show only first 10
            else:
                print("✅ All trainable params are included in optimizer")
        else:
            print("❗️ Optimizer not created yet or not accessible")

    def on_step_begin(self, args, state, control, **kwargs):
        # Only check once at the first step
        if state.global_step == 0:
            print("🔍 [Optimizer Debug] Checking at the beginning of step 0...")
            trainer = kwargs.get("trainer")
            if trainer and trainer.optimizer:
                total_opt_params = sum(len(group['params']) for group in trainer.optimizer.param_groups)
                print(f"Number of params in optimizer at step 0: {total_opt_params}")

                # 🔧 Key fix: check and recreate optimizer
                if total_opt_params == 0:
                    print("❗️ Optimizer has no params! Recreating optimizer...")

                    # Get all trainable params
                    model = trainer.model
                    trainable_params = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            trainable_params.append(param)

                    print(f"Found {len(trainable_params)} trainable params")

                    if trainable_params:
                        # Recreate optimizer
                        import torch.optim as optim
                        trainer.optimizer = optim.AdamW(
                            trainable_params,
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay,
                        )
                        print(f"✅ Optimizer recreated with {len(trainable_params)} params")

                        # If needed, recreate LR scheduler too
                        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                            from transformers.optimization import get_scheduler
                            trainer.lr_scheduler = get_scheduler(
                                name=args.lr_scheduler_type,
                                optimizer=trainer.optimizer,
                                num_warmup_steps=args.get_warmup_steps(args.max_steps),
                                num_training_steps=args.max_steps,
                            )
                            print("✅ LR scheduler recreated")
                    else:
                        print("❌ No trainable params found!")
            else:
                print("❗️ Optimizer still not created at step 0")


class ParameterChangeCallback(TrainerCallback):
    def __init__(self, model_reference):
        """
        model_reference: the actual model being trained (may be wrapped by DDP/FSDP; needs unwrapping)
        """
        self.prev_params = {}
        self.model_ref = model_reference

    def _get_unwrapped_model(self):
        """Get the unwrapped model"""
        model = self.model_ref
        while hasattr(model, 'module'):
            model = model.module
        return model

    def _collect_target_params(self, model):
        """Collect parameters of target layers (custom layers + first layer of base_model)"""
        target_modules = {}

        # Custom layers
        custom_modules = {
            'hidden_mha': model.hidden_mha,
            # 'output_projection': model.output_projection,
            'input_projector': getattr(model, 'input_projector', None),
            'pre_ln': model.pre_ln,
            'post_ln': model.post_ln,
            'adaptive_proj': model.adaptive_proj,
        }
        for name, module in custom_modules.items():
            if module is not None:
                for pname, p in module.named_parameters():
                    full_name = f"{name}.{pname}"
                    target_modules[full_name] = p.data.clone().detach()

        # First layer of base model (assume LLaMA/Qwen style)
        if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'layers'):
            first_layer = model.base_model.model.layers[0]
            for pname, p in first_layer.named_parameters():
                full_name = f"base_model.layers.0.{pname}"
                target_modules[full_name] = p.data.clone().detach()
        else:
            print("⚠️ Warning: unable to locate base_model first layer (layers[0])")

        return target_modules

    def on_step_end(self, args, state, control, **kwargs):
        # Print every 50 steps (including step 0)
        if state.global_step % 50 != 0:
            return control

        if state.global_step == 0:
            # Step 0: cache initial params only
            model = self._get_unwrapped_model()
            self.prev_params = self._collect_target_params(model)
            print("✅ [ParamChange] Initial parameters cached.")
            return control

        # From step 50 onward (and every multiple of 50), compare changes
        model = self._get_unwrapped_model()
        current_params = self._collect_target_params(model)

        print(f"\n📊 [Step {state.global_step}] Parameter Change Summary:")
        for name in current_params:
            if name not in self.prev_params:
                print(f"  ❓ New param: {name}")
                continue
            delta = (current_params[name] - self.prev_params[name]).float()
            l2_norm = delta.norm().item()
            max_abs = delta.abs().max().item()
            mean_abs = delta.abs().mean().item()
            print(f"  {name:<40} | L2: {l2_norm:.2e} | Max: {max_abs:.2e} | Mean: {mean_abs:.2e}")

        # Update cached params for next comparison
        self.prev_params = current_params
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        # Print every 50 steps (including step 0)
        if state.global_step % 50 != 0:
            return control

        model = kwargs['model']
        # get all model parameters
        print(f"{'=' * 100}")
        print(f"{'Parameter Name':<40} | {'Shape':<15} | {'Mean':<12} | {'Std':<12}")
        print("-" * 80)

        for name, param in model.named_parameters():
            if param.numel() == 0:
                mean, std = 0.0, 0.0
            else:
                param_data = param.detach().cpu()
                mean = param_data.mean().item()
                std = param_data.std().item()

            shape_str = str(list(param.shape))
            print(f"{name:<40} | {shape_str:<15} | {mean:<12.4e} | {std:<12.4e}")

            # Advanced debug: check numerical anomalies
            has_nan = torch.isnan(param_data).any().item()
            has_inf = torch.isinf(param_data).any().item()
            min_val = param_data.min().item()
            max_val = param_data.max().item()
            print(f"    → min={min_val:.4e}, max={max_val:.4e}, NaN={has_nan}, Inf={has_inf}")

        print("=" * 80)
        return control


class SaveMHAStateCallback(TrainerCallback):
    """
    Responsible for saving extra model states (e.g., MHA weights).
    Rank 0 check is included.
    """

    def on_save(self, args, state, control, **kwargs):
        try:
            model = kwargs.get("model", None)
            if model is None:
                return control

            # Keep the original Rank 0 check (equivalent to _is_rank0(args))
            is_rank0 = (not torch.distributed.is_available()
                        or not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0)
            if not is_rank0:
                return control

            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Key: call your overridden save_pretrained (will also write hidden_mha_state.pt)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir)

        except Exception as e:
            # Only print warning on Rank 0
            if is_rank0:
                print(f"[SaveMHAStateCallback] WARN: failed to save extra states: {e}")
        return control


class EvalReportCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, data_collator, num_samples: int = 3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        device = next(model.parameters()).device

        # 1) Print aggregated loss components (recorded by the model forward in eval mode)
        comps = getattr(model, "last_loss_components", None)
        if comps:
            print("\n===== Eval Loss Breakdown =====")
            for k, v in comps.items():
                print(f"{k}: {v:.6f}")

                # 2) Append to CSV (saved under output_dir)
                try:
                    out_dir = getattr(args, "output_dir", ".")
                    os.makedirs(out_dir, exist_ok=True)
                    csv_path = os.path.join(out_dir, "mi_probe_log.csv")
                    header_needed = (not os.path.exists(csv_path))
                    with open(csv_path, "a") as f:
                        if header_needed:
                            f.write(
                                "step,probe_js_bits,probe_delta_nll_bits,probe_rate_bits_per_token,probe_ce_pos_nats,probe_ce_neg_nats\n")
                        step = getattr(state, "global_step", -1)
                        f.write(f"{step},{comps.get('probe_js_bits', float('nan'))},"
                                f"{comps.get('probe_delta_nll_bits', float('nan'))},"
                                f"{comps.get('probe_rate_bits_per_token', float('nan'))},"
                                f"{comps.get('probe_ce_pos_nats', float('nan'))},"
                                f"{comps.get('probe_ce_neg_nats', float('nan'))}\n")
                except Exception as e:
                    print(f"[EvalReportCallback] CSV write failed: {e}")

        # 2) Sample a few items: print gold/pred via logits + label_mask (no generate)
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return

        idxs = random.sample(range(len(self.eval_dataset)), min(self.num_samples, len(self.eval_dataset)))
        print("\n===== Eval Sample Teacher-Forcing Outputs =====")

        def safe_decode(tok_ids: torch.Tensor,
                        mask: torch.Tensor,  # True=keep, False=ignore
                        tokenizer,
                        skip_special_tokens=True) -> str:
            """
            Filter tok_ids using mask before decoding.
            """
            if tok_ids.dim() == 2:  # [B,L] → take the 0-th sample
                tok_ids = tok_ids[0]
                mask = mask[0]

            valid_ids = tok_ids[mask]
            return tokenizer.decode(valid_ids.tolist(),
                                    skip_special_tokens=skip_special_tokens,
                                    clean_up_tokenization_spaces=True)

        # Distributed-safe rank0 check
        is_rank0 = (
                not torch.distributed.is_available()  # torch.distributed not available
                or not torch.distributed.is_initialized()  # not initialized
                or torch.distributed.get_rank() == 0  # initialized and rank0
        )

        for idx in idxs:
            ex = self.eval_dataset[idx]
            batch = self.data_collator([ex])
            # Move only tensors to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            model.eval()
            with torch.no_grad():
                # Key change: use the same processing pipeline as training
                # Reset ratio_list for consistency
                original_ratio_list = model.ratio_list.copy() if hasattr(model, 'ratio_list') else []
                model.ratio_list = []

                # Get raw data
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                plans = batch["plans"]
                human_end_positions = batch["human_end_positions"]
                prepended_hidden_states = batch["prepended_hidden_states"]

                # Call curriculum processing to get aligned data (same as training)
                processed_outputs, attention_mask_processed, labels_processed = model._forward_with_hidden_states_curriculum(
                    input_ids, plans, attention_mask, None, labels,
                    human_end_positions, prepended_hidden_states,
                    None, None, None, None, None, disable_mix=True
                )

                # Restore original ratio_list
                model.ratio_list = original_ratio_list

            # Decode using processed data (consistent with training logic)
            sample_idx = 0  # take first sample

            # Mask: keep non-IGNORE positions
            label_mask = labels_processed[sample_idx].ne(IGNORE)  # use processed labels

            # ① Gold text
            gold_text = safe_decode(labels_processed[sample_idx], label_mask, self.tokenizer)

            # ② Pred text (keep only label_mask=True positions)
            pred_ids = torch.argmax(processed_outputs.logits[sample_idx], dim=-1)
            pred_text = safe_decode(pred_ids, label_mask, self.tokenizer)

            # ③ Input (use original input)
            src_text = safe_decode(batch["input_ids"][sample_idx],
                                   mask=batch["input_ids"][sample_idx].ne(self.tokenizer.pad_token_id) if hasattr(
                                       self.tokenizer,
                                       'pad_token_id') and self.tokenizer.pad_token_id is not None else torch.ones_like(
                                       batch["input_ids"][sample_idx], dtype=torch.bool),
                                   tokenizer=self.tokenizer,
                                   skip_special_tokens=False)

            # Per-sample loss (total loss)
            sample_loss = float(processed_outputs.loss.item())

            if is_rank0:
                print(f"\n--- Eval Sample #{idx} ---")
                print(f"Loss(total): {sample_loss:.6f}")
                print(f"[INPUT]\n{src_text[:2000]}{'...' if len(src_text) > 2000 else ''}")
                print(f"[GOLD]\n{gold_text[:2000]}{'...' if len(gold_text) > 2000 else ''}")
                print(f"[PRED]\n{pred_text[:2000]}{'...' if len(pred_text) > 2000 else ''}")

                # Debug info
                print(f"[DEBUG] Original input_ids shape: {batch['input_ids'].shape}")
                print(f"[DEBUG] Original labels shape: {batch['labels'].shape}")
                print(f"[DEBUG] Processed labels shape: {labels_processed.shape}")
                print(f"[DEBUG] Processed logits shape: {processed_outputs.logits.shape}")
                print(f"[DEBUG] Label mask shape: {label_mask.shape}")
                print(f"[DEBUG] Label mask sum: {label_mask.sum().item()}")
                print(f"[DEBUG] Gold text length: {len(gold_text)}")
                print(f"[DEBUG] Pred text length: {len(pred_text)}")


class GradientLoggingCallback(TrainerCallback):
    def __init__(self, log_path="gradient_log.txt", pattern=r"(output_projection|attn|mha)"):
        self.log_path = log_path
        self.pattern = re.compile(pattern)

    def _get_model(self, kwargs):
        # Compatible with different versions: prefer model; otherwise get from trainer
        model = kwargs.get("model")
        if model is None and "trainer" in kwargs:
            model = kwargs["trainer"].model
        return model

    def _iter_named_params(self, model):
        # Compatible with Deepspeed/FSDP wrapping
        mod = getattr(model, "module", model)
        for n, p in mod.named_parameters():
            yield n, p

    def _opt_check(self, trainer, model):
        # Check whether any target params are missing from optimizer
        name_by_id = {}
        for n, p in self._iter_named_params(model):
            name_by_id[id(p)] = n

        opt_names = set()
        for g in trainer.optimizer.param_groups:
            for p in g["params"]:
                n = name_by_id.get(id(p))
                if n is not None:
                    opt_names.add(n)

        miss = [n for n, p in self._iter_named_params(model)
                if self.pattern.search(n) and n not in opt_names and p.requires_grad]
        print("[OPT CHECK] missing params in optimizer:", miss)
        for i, g in enumerate(trainer.optimizer.param_groups):
            print(f'[OPT GROUP {i}] lr={g["lr"]} num_params={len(g["params"])}')

    def _safe_stats(self, t):
        try:
            v = t.detach()
            return dict(
                norm=float(v.norm().item()),
                mean=float(v.abs().mean().item()),
                max=float(v.abs().max().item()),
            )
        except Exception:
            return dict(norm=math.nan, mean=math.nan, max=math.nan)

    def _log_grads(self, model, header):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{header}\n")
            print(header)
            for name, p in self._iter_named_params(model):
                if not self.pattern.search(name):
                    continue
                if p.grad is None:
                    line = f"{name}: no gradient"
                else:
                    s = self._safe_stats(p.grad)
                    line = f"{name}: grad_norm={s['norm']:.6f}, mean={s['mean']:.6f}, max={s['max']:.6f}"
                f.write(line + "\n")
                print(line)

    def _log_weights(self, model, header):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{header}\n")
            print(header)
            for name, p in self._iter_named_params(model):
                if not self.pattern.search(name):
                    continue
                s = self._safe_stats(p.data)
                line = f"{name}: weight_norm={s['norm']:.6f}, mean={s['mean']:.6f}, max={s['max']:.6f}"
                f.write(line + "\n")
                print(line)

    def on_train_begin(self, args, state, control, **kwargs):
        model = self._get_model(kwargs)
        trainer = kwargs.get("trainer", None)
        print(f"Training starting with model: {type(model).__name__}")
        if trainer is not None and trainer.optimizer is not None:
            self._opt_check(trainer, model)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Gradient/Weight Log Start ===\n")

    def on_backward_end(self, args, state, control, **kwargs):
        # 🔍 First confirm this callback is being called
        print(f"🔍 [BACKWARD_END] Step {state.global_step} - Callback called")

        if state.global_step % 10:  # adjust frequency as needed
            return
        model = kwargs.get("model", kwargs["trainer"].model)
        mod = getattr(model, "module", model)

        print(f"🔍 [BACKWARD_END CHECK] Step {state.global_step} - Backward just finished, checking gradients...")

        # 🔧 [Key Fix] Store parameter IDs here (first time)
        if state.global_step == 10 and not hasattr(mod, '_gradient_test_param_ids'):
            print("🔧 [Fix] Storing parameter IDs directly in callback...")

            param_ids = {}
            for name, param in mod.named_parameters():
                if any(k in name for k in ["hidden_mha", "input_projector", "adaptive_proj", "pre_ln", "post_ln"]):
                    param_ids[name] = id(param)
                    print(f"[Callback Stored] {name}: {id(param)}")

            mod._gradient_test_param_ids = param_ids
            print(f"✅ [Fix] Stored {len(param_ids)} parameter IDs in callback")

        for n, p in mod.named_parameters():
            if any(k in n for k in ["hidden_mha", "input_projector", "adaptive_proj", "pre_ln", "post_ln"]):
                # 🔍 Check whether param ID matches the one stored earlier
                param_id_info = ""

                # 🔍 Search param ID record across multiple locations
                gradient_test_param_ids = None
                search_locations = [mod]

                # Add possible search locations
                if hasattr(mod, 'base_model'):
                    search_locations.append(mod.base_model)

                # Add possibly wrapped root module
                root_mod = mod
                while hasattr(root_mod, 'module'):
                    root_mod = root_mod.module
                    search_locations.append(root_mod)

                # Find param ID record
                for loc in search_locations:
                    if hasattr(loc, '_gradient_test_param_ids'):
                        gradient_test_param_ids = loc._gradient_test_param_ids
                        break

                if gradient_test_param_ids and n in gradient_test_param_ids:
                    original_id = gradient_test_param_ids[n]
                    current_id = id(p)
                    if original_id == current_id:
                        param_id_info = " [ID MATCH]"
                    else:
                        param_id_info = f" [ID MISMATCH: {original_id} -> {current_id}]"
                elif gradient_test_param_ids:
                    param_id_info = " [Name not in record]"
                else:
                    param_id_info = " [No ID record]"

                if p.grad is not None:
                    print(
                        f"✅ [BACKWARD_END][{state.global_step}] {n}: grad_norm={p.grad.norm().item():.3e}{param_id_info}")
                else:
                    print(f"❌ [BACKWARD_END][{state.global_step}] {n}: grad=None{param_id_info}")

    def on_step_end(self, args, state, control, **kwargs):
        # Note: this callback runs after optimizer step; gradients are already cleared
        # Mainly used to confirm training is progressing normally
        if state.global_step % 10 != 0:
            return

        model = kwargs.get("model")
        if model is None:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return control
            model = trainer.model

        mod = getattr(model, "module", model)  # Compatible with DDP

        print(f"✅ [STEP_END CONFIRM] Step {state.global_step} - Optimizer step completed")

        # Simple training status check
        total_params = sum(1 for p in mod.parameters() if p.requires_grad)
        print(f"✅ [TRAINING STATUS] Model has {total_params} trainable parameters")

        # Check whether weights updated (simple stats on first MHA weight)
        if hasattr(mod, 'hidden_mha') and hasattr(mod.hidden_mha, 'in_proj_weight'):
            weight = mod.hidden_mha.in_proj_weight
            weight_mean = weight.data.mean().item()
            weight_std = weight.data.std().item()
            print(f"✅ [WEIGHT STATS] MHA weight mean={weight_mean:.6f}, std={weight_std:.6f}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training completed. Gradient log saved to {self.log_path}")


class LossRecorderCallback(TrainerCallback):
    def __init__(self, log_path="loss_log.txt", plot_path="loss_curve.png"):
        self.losses = []
        self.log_path = log_path
        self.plot_path = plot_path

    def on_train_begin(self, args, state, control, **kwargs):
        self.losses.clear()

    def on_evaluate(self, args, state, control, **kwargs):
        # Optional: also record once per evaluation
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            loss = logs["loss"]
            print(f"\nStep {state.global_step} - Loss: {loss:.4f}")
            self.losses.append(loss)
            with open(self.log_path, 'a') as f:
                f.write(f"{state.global_step},{loss}\n")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            # Plot loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, label='Training Loss')
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.plot_path)
            plt.close()
            print(f"Loss curve saved to {self.plot_path}")
        except Exception as e:
            print(f"Could not save loss curve to {self.plot_path}: {e}")
            # Try saving to current directory
            try:
                fallback_path = "./loss_curve.png"
                plt.savefig(fallback_path)
                plt.close()
                print(f"Loss curve saved to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
                plt.close()  # Ensure the figure is closed
