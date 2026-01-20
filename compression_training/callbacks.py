# callbacks.py
from __future__ import annotations

import os
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import TrainerCallback


__all__ = [
    "PreCreateCkptDirCallback",
    "ParamChangeTrackerCallback",
    "EarlyStoppingStatusCallback",
    "PrintMetricsCallback",
]


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_global_rank0() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _is_local_process_zero(state) -> bool:
    """
    HuggingFace TrainerState has `is_local_process_zero` in many versions.
    Fallback to global rank 0 if missing.
    """
    if hasattr(state, "is_local_process_zero"):
        return bool(state.is_local_process_zero)
    return _is_global_rank0()


class PreCreateCkptDirCallback(TrainerCallback):
    """
    Pre-create checkpoint directories at save steps to avoid race conditions
    in distributed training when multiple processes attempt to save.
    """

    def on_step_end(self, args, state, control, **kwargs):
        save_strategy = getattr(args, "save_strategy", None)
        save_steps = int(getattr(args, "save_steps", 0) or 0)

        # `save_strategy` might be Enum (IntervalStrategy) or str.
        if str(save_strategy) != "steps" or save_steps <= 0:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or (step % save_steps != 0):
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

        if _is_dist_initialized():
            if _is_global_rank0():
                os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()
        else:
            os.makedirs(ckpt_dir, exist_ok=True)


class ParamChangeTrackerCallback(TrainerCallback):
    """
    Track parameter changes and per-step gradient norms for selected parameters.

    It prints:
      - Δprev: parameter delta norm compared to previous log snapshot
      - Δinit: parameter delta norm compared to initial snapshot
      - ‖θ‖: current parameter norm
      - ‖∇θ‖_step: gradient norm captured at step end

    Notes:
      - The snapshot is stored on CPU in fp32 for stable statistics.
      - Printing is limited to local process zero to avoid log spam.
    """

    def __init__(
        self,
        model: nn.Module,
        track_patterns: Optional[Sequence[str]] = None,
        topn: int = 10,
        skip_first_log: bool = True,
        verbose_on_bind: bool = True,
    ):
        super().__init__()
        self.model = model
        self.track_patterns = list(track_patterns) if track_patterns is not None else [
            "h2e",
            "student_lm.model.layers.",
            "student_lm.lm_head",
            "lat_bos",
        ]
        self.topn = int(topn)
        self.skip_first_log = bool(skip_first_log)
        self.verbose_on_bind = bool(verbose_on_bind)

        self._bound: bool = False
        self._tracked: List[Tuple[str, nn.Parameter]] = []
        self._init_snap: Dict[str, torch.Tensor] = {}
        self._prev_snap: Dict[str, torch.Tensor] = {}
        self._step_grad_norm: Dict[str, float] = {}
        self._did_first_log: bool = False

    def _select_tracked(self) -> List[Tuple[str, nn.Parameter]]:
        tracked: List[Tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(pat in name for pat in self.track_patterns):
                tracked.append((name, p))
        return tracked

    def _take_snapshot(self) -> None:
        self._init_snap = {n: p.detach().float().cpu().clone() for n, p in self._tracked}
        self._prev_snap = {n: p.detach().float().cpu().clone() for n, p in self._tracked}

    def on_train_begin(self, args, state, control, **kwargs):
        self._tracked = self._select_tracked()
        self._take_snapshot()
        self._step_grad_norm = {}
        self._bound = True

        if not _is_local_process_zero(state):
            return

        if not self._tracked:
            print("[ParamChangeTracker] WARNING: No parameters matched track_patterns =", self.track_patterns)
            return

        if self.verbose_on_bind:
            print("[ParamChangeTracker] Tracking parameters:")
            for n, _ in self._tracked:
                print("  -", n)

    def on_step_end(self, args, state, control, **kwargs):
        if not self._bound:
            return

        # Capture grad norms right after backward (before optimizer step may clear grads).
        self._step_grad_norm = {}
        for name, p in self._tracked:
            if p.grad is None:
                continue
            self._step_grad_norm[name] = float(p.grad.detach().float().norm().item())

    def _compute_stats(self) -> List[Tuple[str, float, float, float, float]]:
        stats: List[Tuple[str, float, float, float, float]] = []
        for name, p in self._tracked:
            cur = p.detach().float().cpu()
            theta_norm = float(cur.norm().item())
            d_prev = float((cur - self._prev_snap[name]).norm().item())
            d_init = float((cur - self._init_snap[name]).norm().item())
            g_norm = float(self._step_grad_norm.get(name, 0.0))

            stats.append((name, theta_norm, d_prev, d_init, g_norm))
            self._prev_snap[name] = cur.clone()

        # Sort by Δprev descending (largest movement first)
        stats.sort(key=lambda x: x[2], reverse=True)
        return stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._bound:
            return
        if self.skip_first_log and not self._did_first_log:
            self._did_first_log = True
            return
        if not _is_local_process_zero(state):
            return
        if not self._tracked:
            return

        stats = self._compute_stats()
        step = int(getattr(state, "global_step", 0) or 0)

        print(f"[param-delta] step={step} (top {self.topn})")
        print(f"{'name':60s}  {'Δprev':>10s}  {'Δinit':>10s}  {'‖θ‖':>10s}  {'‖∇θ‖_step':>12s}")
        for name, theta_norm, d_prev, d_init, g_norm in stats[: self.topn]:
            print(f"{name:60s}  {d_prev:10.4e}  {d_init:10.4e}  {theta_norm:10.4e}  {g_norm:12.4e}")


class EarlyStoppingStatusCallback(TrainerCallback):
    """
    Print early-stopping status on each evaluation.

    This callback does NOT stop training by itself.
    It is for transparent logging alongside HuggingFace's EarlyStoppingCallback.

    Args:
      metric_for_best: metric name, e.g., "loss" or "accuracy". If missing "eval_", it will be prefixed.
      greater_is_better: whether larger metric indicates improvement.
      patience: allowed number of non-improving evals.
      threshold: minimum change to be considered an improvement.
      show_last: number of recent eval records to show.
    """

    def __init__(
        self,
        metric_for_best: str,
        greater_is_better: bool,
        patience: int,
        threshold: float,
        show_last: int = 5,
    ):
        self.metric_for_best = str(metric_for_best)
        self.greater_is_better = bool(greater_is_better)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.show_last = int(show_last)

        self.best: Optional[float] = None
        self.bad_count: int = 0
        self.history: Deque[Tuple[int, float, bool]] = deque(maxlen=max(5, self.show_last))

    def _metric_key(self, metrics: Dict[str, float]) -> str:
        key = self.metric_for_best
        if not key.startswith("eval_"):
            key = f"eval_{key}"
        if key not in metrics:
            key = "eval_loss"
        return key

    def _is_improved(self, cur: float) -> bool:
        if self.best is None:
            return True
        if self.greater_is_better:
            return cur > (self.best + self.threshold)
        return cur < (self.best - self.threshold)

    def on_train_begin(self, args, state, control, **kwargs):
        if not _is_local_process_zero(state):
            return
        arrow = "↑" if self.greater_is_better else "↓"
        print(
            f"[early-stop] watching '{self.metric_for_best}' (mapped to eval_*), "
            f"target {arrow}, patience={self.patience}, threshold={self.threshold}"
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not _is_local_process_zero(state):
            return

        if not isinstance(metrics, dict):
            return

        key = self._metric_key(metrics)
        if key not in metrics:
            print(f"[early-stop] metric '{key}' not found. available: {list(metrics.keys())}")
            return

        cur = float(metrics[key])
        step = int(getattr(state, "global_step", 0) or 0)
        improved = self._is_improved(cur)

        if improved:
            self.best = cur
            self.bad_count = 0
        else:
            self.bad_count += 1

        remain = max(0, self.patience - self.bad_count)
        arrow = "↑" if self.greater_is_better else "↓"
        tag = "✅ improved" if improved else "— no improve"

        self.history.append((step, cur, improved))

        print(
            f"[early-stop] step={step} {key}={cur:.6f} | best={self.best:.6f} (target {arrow}) | "
            f"{tag} | patience used={self.bad_count}/{self.patience} → remaining={remain}"
        )

        rows = list(self.history)[-self.show_last :]
        print(f"{'step':>8}  {key:>16}  {'improved':>9}")
        for s, v, imp in rows:
            print(f"{s:8d}  {v:16.6f}  {str(imp):>9}")


class PrintMetricsCallback(TrainerCallback):
    """
    Print custom metrics stored in `model.last_metrics` (a dict).

    Expected:
      model.last_metrics = {"ce_theta": ..., "kl": ..., "align": ..., "total": ...}
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        if hasattr(model, "module"):  # DDP wrapper
            model = model.module

        metrics = getattr(model, "last_metrics", None)
        if not metrics:
            return

        logging_steps = int(getattr(args, "logging_steps", 1) or 1)
        step = int(getattr(state, "global_step", 0) or 0)

        if step % logging_steps != 0:
            return
        if not _is_local_process_zero(state):
            return

        print(
            f"[step {step}] "
            f"ce_theta={float(metrics.get('ce_theta', -1)):.6f} "
            f"kl={float(metrics.get('kl', -1)):.6f} "
            f"align={float(metrics.get('align', -1)):.6f} "
            f"total={float(metrics.get('total', -1)):.6f}"
        )
# callbacks.py
from __future__ import annotations

import os
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import TrainerCallback

__all__ = [
    "PreCreateCkptDirCallback",
    "ParamChangeTrackerCallback",
    "EarlyStoppingStatusCallback",
    "PrintMetricsCallback",
]


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_global_rank0() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _is_local_process_zero(state) -> bool:
    # Many HF versions have state.is_local_process_zero
    if hasattr(state, "is_local_process_zero"):
        return bool(state.is_local_process_zero)
    return _is_global_rank0()


class PreCreateCkptDirCallback(TrainerCallback):
    """
    Pre-create checkpoint directories at save steps to avoid race conditions in distributed training.
    """

    def on_step_end(self, args, state, control, **kwargs):
        save_strategy = getattr(args, "save_strategy", None)
        save_steps = int(getattr(args, "save_steps", 0) or 0)

        # save_strategy can be Enum or str; cast to str for compatibility.
        if str(save_strategy) != "steps" or save_steps <= 0:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or (step % save_steps != 0):
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

        if _is_dist_initialized():
            if _is_global_rank0():
                os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()
        else:
            os.makedirs(ckpt_dir, exist_ok=True)


class ParamChangeTrackerCallback(TrainerCallback):
    """
    Track parameter changes and per-step gradient norms for selected parameters.

    Prints:
      - Δprev: parameter delta norm compared to previous snapshot
      - Δinit: parameter delta norm compared to initial snapshot
      - ‖θ‖: current parameter norm
      - ‖∇θ‖_step: gradient norm captured at step end

    Notes:
      - Snapshots stored on CPU in fp32 for stable statistics.
      - Printing only on local process zero to avoid log spam.
    """

    def __init__(
        self,
        model: nn.Module,
        track_patterns: Optional[Sequence[str]] = None,
        topn: int = 10,
        skip_first_log: bool = True,
        verbose_on_bind: bool = True,
    ):
        super().__init__()
        self.model = model
        self.track_patterns = list(track_patterns) if track_patterns is not None else [
            "h2e",
            "student_lm.model.layers.",
            "student_lm.lm_head",
            "lat_bos",
        ]
        self.topn = int(topn)
        self.skip_first_log = bool(skip_first_log)
        self.verbose_on_bind = bool(verbose_on_bind)

        self._bound: bool = False
        self._tracked: List[Tuple[str, nn.Parameter]] = []
        self._init_snap: Dict[str, torch.Tensor] = {}
        self._prev_snap: Dict[str, torch.Tensor] = {}
        self._step_grad_norm: Dict[str, float] = {}
        self._did_first_log: bool = False

    def _select_tracked(self) -> List[Tuple[str, nn.Parameter]]:
        tracked: List[Tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(pat in name for pat in self.track_patterns):
                tracked.append((name, p))
        return tracked

    def _take_snapshot(self) -> None:
        self._init_snap = {n: p.detach().float().cpu().clone() for n, p in self._tracked}
        self._prev_snap = {n: p.detach().float().cpu().clone() for n, p in self._tracked}

    def on_train_begin(self, args, state, control, **kwargs):
        self._tracked = self._select_tracked()
        self._take_snapshot()
        self._step_grad_norm = {}
        self._bound = True

        if not _is_local_process_zero(state):
            return

        if not self._tracked:
            print("[ParamChangeTracker] WARNING: No parameters matched track_patterns =", self.track_patterns)
            return

        if self.verbose_on_bind:
            print("[ParamChangeTracker] Tracking parameters:")
            for n, _ in self._tracked:
                print("  -", n)

    def on_step_end(self, args, state, control, **kwargs):
        if not self._bound:
            return

        # Capture grad norms after backward. Some optimizers may clear grads later.
        self._step_grad_norm = {}
        for name, p in self._tracked:
            if p.grad is None:
                continue
            self._step_grad_norm[name] = float(p.grad.detach().float().norm().item())

    def _compute_stats(self) -> List[Tuple[str, float, float, float, float]]:
        stats: List[Tuple[str, float, float, float, float]] = []
        for name, p in self._tracked:
            cur = p.detach().float().cpu()
            theta_norm = float(cur.norm().item())
            d_prev = float((cur - self._prev_snap[name]).norm().item())
            d_init = float((cur - self._init_snap[name]).norm().item())
            g_norm = float(self._step_grad_norm.get(name, 0.0))

            stats.append((name, theta_norm, d_prev, d_init, g_norm))
            self._prev_snap[name] = cur.clone()

        stats.sort(key=lambda x: x[2], reverse=True)
        return stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._bound:
            return
        if self.skip_first_log and not self._did_first_log:
            self._did_first_log = True
            return
        if not _is_local_process_zero(state):
            return
        if not self._tracked:
            return

        stats = self._compute_stats()
        step = int(getattr(state, "global_step", 0) or 0)

        print(f"[param-delta] step={step} (top {self.topn})")
        print(f"{'name':60s}  {'Δprev':>10s}  {'Δinit':>10s}  {'‖θ‖':>10s}  {'‖∇θ‖_step':>12s}")
        for name, theta_norm, d_prev, d_init, g_norm in stats[: self.topn]:
            print(f"{name:60s}  {d_prev:10.4e}  {d_init:10.4e}  {theta_norm:10.4e}  {g_norm:12.4e}")


class EarlyStoppingStatusCallback(TrainerCallback):
    """
    Display detailed early-stopping status on each evaluation.

    This callback does NOT stop training by itself.
    It is intended to be used alongside transformers.EarlyStoppingCallback.
    """

    def __init__(
        self,
        metric_for_best: str,
        greater_is_better: bool,
        patience: int,
        threshold: float,
        show_last: int = 5,
    ):
        self.metric_for_best = str(metric_for_best)
        self.greater_is_better = bool(greater_is_better)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.show_last = int(show_last)

        self.best: Optional[float] = None
        self.bad_count: int = 0
        self.history: Deque[Tuple[int, float, bool]] = deque(maxlen=max(5, self.show_last))

    def _metric_key(self, metrics: Dict[str, float]) -> str:
        key = self.metric_for_best
        if not key.startswith("eval_"):
            key = f"eval_{key}"
        if key not in metrics:
            key = "eval_loss"
        return key

    def _is_improved(self, cur: float) -> bool:
        if self.best is None:
            return True
        if self.greater_is_better:
            return cur > (self.best + self.threshold)
        return cur < (self.best - self.threshold)

    def on_train_begin(self, args, state, control, **kwargs):
        if not _is_local_process_zero(state):
            return
        arrow = "↑" if self.greater_is_better else "↓"
        print(
            f"[early-stop] watching '{self.metric_for_best}' (mapped to eval_*), "
            f"target {arrow}, patience={self.patience}, threshold={self.threshold}"
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not _is_local_process_zero(state):
            return
        if not isinstance(metrics, dict):
            return

        key = self._metric_key(metrics)
        if key not in metrics:
            print(f"[early-stop] metric '{key}' not found. available: {list(metrics.keys())}")
            return

        cur = float(metrics[key])
        step = int(getattr(state, "global_step", 0) or 0)
        improved = self._is_improved(cur)

        if improved:
            self.best = cur
            self.bad_count = 0
        else:
            self.bad_count += 1

        remain = max(0, self.patience - self.bad_count)
        arrow = "↑" if self.greater_is_better else "↓"
        tag = "✅ improved" if improved else "— no improve"

        self.history.append((step, cur, improved))

        print(
            f"[early-stop] step={step} {key}={cur:.6f} | best={self.best:.6f} (target {arrow}) | "
            f"{tag} | patience used={self.bad_count}/{self.patience} → remaining={remain}"
        )

        rows = list(self.history)[-self.show_last :]
        print(f"{'step':>8}  {key:>16}  {'improved':>9}")
        for s, v, imp in rows:
            print(f"{s:8d}  {v:16.6f}  {str(imp):>9}")


class PrintMetricsCallback(TrainerCallback):
    """
    Print custom metrics stored in `model.last_metrics` (a dict).
    Expected keys: ce_theta, kl, align, total
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        if hasattr(model, "module"):  # DDP wrapper
            model = model.module

        metrics = getattr(model, "last_metrics", None)
        if not metrics:
            return

        logging_steps = int(getattr(args, "logging_steps", 1) or 1)
        step = int(getattr(state, "global_step", 0) or 0)

        if step % logging_steps != 0:
            return
        if not _is_local_process_zero(state):
            return

        print(
            f"[step {step}] "
            f"ce_theta={float(metrics.get('ce_theta', -1)):.6f} "
            f"kl={float(metrics.get('kl', -1)):.6f} "
            f"align={float(metrics.get('align', -1)):.6f} "
            f"total={float(metrics.get('total', -1)):.6f}"
        )
