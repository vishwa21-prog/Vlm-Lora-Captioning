"""Shared helpers: config loading, seeding, timing, GPU memory tracking,
trainable-parameter counting. Used by every training/eval script so cost
metrics are computed the same way everywhere.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load the project YAML config into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
class Timer:
    """Simple wall-clock timer, usable as a context manager.

    with Timer() as t:
        train()
    print(t.elapsed_seconds)
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_seconds: float = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start


# --------------------------------------------------------------------------
# GPU memory tracking
# --------------------------------------------------------------------------
class GPUMemoryTracker:
    """Tracks peak CUDA memory allocated during a `with` block.

    Reports 0 on machines without CUDA (e.g. CPU-only smoke tests) rather
    than raising, so the same code path works everywhere.
    """

    def __enter__(self) -> "GPUMemoryTracker":
        self.peak_mb: float = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)


# --------------------------------------------------------------------------
# Parameter counting (the key "training cost" number for LoRA vs full FT)
# --------------------------------------------------------------------------
def count_trainable_params(model: torch.nn.Module) -> Dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(pct, 4),
    }


def print_trainable_params(model: torch.nn.Module, label: str = "") -> Dict[str, float]:
    stats = count_trainable_params(model)
    print(
        f"[{label}] trainable params: {stats['trainable_params']:,} / "
        f"{stats['total_params']:,} ({stats['trainable_pct']:.4f}%)"
    )
    return stats


# --------------------------------------------------------------------------
# Checkpoint disk size (part of "training cost" comparison)
# --------------------------------------------------------------------------
def dir_size_mb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return round(total / (1024 ** 2), 2)


# --------------------------------------------------------------------------
# Optional W&B init that no-ops cleanly if disabled or not installed
# --------------------------------------------------------------------------
@dataclass
class WandbRun:
    enabled: bool = False
    run: Optional[Any] = field(default=None)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(data, step=step)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def init_wandb(cfg: Dict[str, Any], run_name: str, job_type: str) -> WandbRun:
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("use_wandb", False):
        return WandbRun(enabled=False)
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without experiment logging.")
        return WandbRun(enabled=False)

    run = wandb.init(
        project=log_cfg.get("wandb_project", "smolvlm-lora-domain-captioning"),
        entity=log_cfg.get("wandb_entity"),
        name=run_name,
        job_type=job_type,
        config=cfg,
        reinit=True,
    )
    return WandbRun(enabled=True, run=run)
