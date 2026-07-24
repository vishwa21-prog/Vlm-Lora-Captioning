"""Evaluates and compares base / fully fine-tuned / LoRA-fine-tuned SmolVLM
checkpoints on the held-out test split: caption quality (BLEU-4, CIDEr) and
training cost (trainable params, wall-clock time, peak GPU memory,
checkpoint disk size — the latter three pulled from each checkpoint's
training_cost.json, written by train_full.py / train_lora.py).

Usage:
    python -m src.evaluate --config config/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import sacrebleu
import torch
from tqdm import tqdm

from src.data import load_caption_dataset
from src.inference import generate_caption, load_model_for_inference
from src.utils import init_wandb, load_config


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU-4 via sacrebleu (one reference per hypothesis)."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(bleu.score, 2)


def compute_cider(hypotheses: List[str], references: List[str]) -> float:
    """CIDEr via pycocoevalcap. Falls back to reporting None if the package
    (or its numpy/six deps) isn't available, rather than crashing the whole
    evaluation run.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("pycocoevalcap not installed — skipping CIDEr. `pip install pycocoevalcap`.")
        return None

    gts = {i: [" ".join(_simple_tokenize(references[i]))] for i in range(len(references))}
    res = {i: [" ".join(_simple_tokenize(hypotheses[i]))] for i in range(len(hypotheses))}

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return round(float(score), 4)


def caption_dataset(model, processor, dataset, cfg, device: str) -> Dict[str, List[str]]:
    caption_col = cfg["data"]["caption_column"]
    image_col = cfg["data"]["image_column"]

    hyps, refs = [], []
    for example in tqdm(dataset, desc="Generating captions"):
        image = example[image_col]
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
        caption = generate_caption(model, processor, image, cfg, device=device)
        hyps.append(caption)
        refs.append(str(example[caption_col]).strip())
    return {"hypotheses": hyps, "references": refs}


def load_cost_report(checkpoint_dir: str) -> Dict:
    path = os.path.join(checkpoint_dir, "training_cost.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_caption_dataset(cfg)
    test_set = dataset["test"]
    print(f"Evaluating on {len(test_set)} test examples")

    checkpoints = {
        "base": None,
        "full": cfg["training"]["output_dir_full"],
        "lora": cfg["training"]["output_dir_lora"],
    }
    to_compare = cfg["evaluation"].get("compare_checkpoints", ["base", "full", "lora"])

    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    wandb_run = init_wandb(cfg, run_name="evaluate-comparison", job_type="evaluate")

    rows = []
    all_captions = {}
    for name in to_compare:
        checkpoint_dir = checkpoints.get(name)
        if checkpoint_dir and not os.path.isdir(checkpoint_dir):
            print(f"Skipping '{name}': checkpoint dir {checkpoint_dir} not found "
                  f"(train it first with train_full.py / train_lora.py).")
            continue

        print(f"\n=== Evaluating: {name} ===")
        model, processor = load_model_for_inference(cfg, checkpoint_dir, device)
        captions = caption_dataset(model, processor, test_set, cfg, device)
        all_captions[name] = captions

        bleu = compute_bleu(captions["hypotheses"], captions["references"])
        cider = compute_cider(captions["hypotheses"], captions["references"])
        cost = load_cost_report(checkpoint_dir) if checkpoint_dir else {}

        row = {
            "checkpoint": name,
            "bleu4": bleu,
            "cider": cider,
            "trainable_params": cost.get("trainable_params"),
            "trainable_pct": cost.get("trainable_pct"),
            "training_seconds": cost.get("training_seconds"),
            "peak_gpu_memory_mb": cost.get("peak_gpu_memory_mb"),
            "checkpoint_size_mb": cost.get("checkpoint_size_mb"),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write comparison table.
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison table written to {csv_path}")

    # Save raw generated captions for qualitative inspection.
    captions_path = os.path.join(results_dir, "sample_captions.json")
    with open(captions_path, "w") as f:
        json.dump(all_captions, f, indent=2)
    print(f"Sample captions written to {captions_path}")

    if wandb_run.enabled:
        import wandb

        table = wandb.Table(columns=list(rows[0].keys()) if rows else [])
        for r in rows:
            table.add_data(*[r[k] for k in r])
        wandb_run.log({"comparison_table": table})
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
