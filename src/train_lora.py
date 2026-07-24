"""LoRA fine-tuning of SmolVLM (via Hugging Face PEFT) on the configured
domain captioning dataset. Base weights stay frozen; only LoRA adapter
matrices are trained.

Usage:
    python -m src.train_lora --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import Trainer, TrainingArguments

from src.data import CaptionCollator, load_caption_dataset
from src.model_utils import apply_lora, enable_gradient_checkpointing, load_base_model_and_processor
from src.utils import (
    GPUMemoryTracker,
    Timer,
    dir_size_mb,
    init_wandb,
    load_config,
    print_trainable_params,
    set_seed,
)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    dataset = load_caption_dataset(cfg)
    print(f"train/val/test sizes: "
          f"{len(dataset['train'])}/{len(dataset['validation'])}/{len(dataset['test'])}")

    dtype = torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float32
    model, processor = load_base_model_and_processor(cfg, dtype=dtype)

    # Freeze everything, then let PEFT punch small trainable LoRA holes
    # into the target modules named in config['lora']['target_modules'].
    for p in model.parameters():
        p.requires_grad = False

    if cfg["training"].get("gradient_checkpointing", True):
        enable_gradient_checkpointing(model)

    model = apply_lora(model, cfg)
    stats = print_trainable_params(model, label="lora-finetune")

    collator = CaptionCollator(
        processor=processor,
        image_column=cfg["data"]["image_column"],
        caption_column=cfg["data"]["caption_column"],
        max_target_length=cfg["data"]["max_target_length"],
        train=True,
    )

    out_dir = cfg["training"]["output_dir_lora"]
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate_lora"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        max_steps=cfg["training"]["max_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=2,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        fp16=cfg["training"].get("fp16", False),
        bf16=cfg["training"].get("bf16", True),
        report_to=["wandb"] if cfg["logging"].get("use_wandb", False) else [],
        run_name=cfg["logging"].get("run_name_lora", "smolvlm-lora-finetune"),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    wandb_run = init_wandb(cfg, args.run_name, job_type="train_lora")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    with Timer() as timer, GPUMemoryTracker() as mem:
        trainer.train()

    # Saves only the (small) adapter weights, not the full base model.
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    cost_report = {
        "mode": "lora_finetune",
        "training_seconds": round(timer.elapsed_seconds, 2),
        "peak_gpu_memory_mb": round(mem.peak_mb, 2),
        "checkpoint_size_mb": dir_size_mb(out_dir),
        **stats,
    }
    report_path = os.path.join(out_dir, "training_cost.json")
    with open(report_path, "w") as f:
        json.dump(cost_report, f, indent=2)
    print(f"Training cost report written to {report_path}")
    print(json.dumps(cost_report, indent=2))

    wandb_run.log(cost_report)
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
