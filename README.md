# Fine-Tuning a Vision-Language Model for Domain Captioning

Fine-tunes **SmolVLM** (a small pretrained Vision-Language Model from Hugging Face)
on a narrow-domain image-captioning dataset, comparing three settings:

1. **Base model** — zero-shot / off-the-shelf SmolVLM, no training.
2. **Fully fine-tuned** — every parameter updated.
3. **LoRA fine-tuned** — low-rank adapters injected via Hugging Face **PEFT**,
   base weights frozen.

Comparison is done on caption quality (**BLEU**, **CIDEr**) and training cost
(trainable params, wall-clock time, peak GPU memory), all logged to
**Weights & Biases**.

---

## 1. Project layout

```
vlm-lora-captioning/
├── config/
│   └── config.yaml          # single source of truth for all hyperparameters
├── src/
│   ├── data.py               # dataset loading / preprocessing / collator
│   ├── model_utils.py        # model + processor loading, LoRA wrapping
│   ├── train_full.py         # full fine-tuning script
│   ├── train_lora.py         # LoRA fine-tuning script (PEFT)
│   ├── evaluate.py           # BLEU/CIDEr + cost comparison across checkpoints
│   ├── inference.py          # generate captions from any checkpoint
│   └── utils.py              # seeding, cost/timing tracking, logging helpers
├── scripts/
│   └── run_all.sh            # runs the full pipeline end-to-end
├── outputs/
│   ├── checkpoints/          # saved model weights land here
│   └── results/              # metrics.json / comparison_table.csv land here
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login        # only needed for gated models / pushing to hub
wandb login                  # only needed if config.yaml -> logging.use_wandb: true
```

Tested against: `transformers>=4.46`, `peft>=0.13`, `torch>=2.2`.
SmolVLM (`HuggingFaceTB/SmolVLM-Instruct` / `HuggingFaceTB/SmolVLM-256M-Instruct`)
became available in Transformers starting with the 4.46 line — if `from_pretrained`
fails with a `ValueError: Unrecognized configuration class`, upgrade `transformers`.

## 3. Dataset

By default this project is wired to `ybelkada/football-dataset` (a tiny public
image-caption dataset good for a smoke test — a few hundred image/caption pairs,
domain = football photos). **Swap in your real narrow-domain dataset** by
editing `config.yaml -> data.dataset_name`. Any Hugging Face `datasets` entry
(or local `imagefolder` with a `metadata.csv` of `file_name,caption` columns)
works as long as it exposes an image column and a text/caption column — set
`data.image_column` / `data.caption_column` accordingly.

To use a local folder instead of the Hub:
```yaml
data:
  dataset_name: "imagefolder"
  data_dir: "./data/my_domain_captions"   # folder with images/ + metadata.csv
```

## 4. Run everything

```bash
bash scripts/run_all.sh
```

This runs, in order: full fine-tune → LoRA fine-tune → evaluation of all three
checkpoints (base, full, LoRA) → writes `outputs/results/comparison_table.csv`.

Or run steps individually:

```bash
# Full fine-tuning
python -m src.train_full --config config/config.yaml

# LoRA fine-tuning
python -m src.train_lora --config config/config.yaml

# Evaluate base vs full vs LoRA
python -m src.evaluate --config config/config.yaml

# Caption a single image with any checkpoint
python -m src.inference --config config/config.yaml \
    --checkpoint outputs/checkpoints/lora \
    --image path/to/image.jpg
```

## 5. What gets compared, and how

| Axis | Metric | Where |
|---|---|---|
| Caption quality | BLEU-4 (sacrebleu) | `src/evaluate.py` |
| Caption quality | CIDEr (pycocoevalcap) | `src/evaluate.py` |
| Training cost | # trainable params, % of total | `src/utils.py::count_trainable_params` |
| Training cost | wall-clock training time | `src/utils.py::Timer` |
| Training cost | peak GPU memory (MB) | `src/utils.py::GPUMemoryTracker` |
| Training cost | checkpoint disk size (MB) | `src/evaluate.py` |

Results are written to `outputs/results/comparison_table.csv` and, if enabled,
logged as a W&B Table plus per-epoch loss curves.

## 6. LoRA configuration

LoRA is applied to the language-model attention projections (`q_proj`,
`k_proj`, `v_proj`, `o_proj`) by default — edit `config.yaml -> lora` to
target vision-tower layers too, or change rank/alpha/dropout:

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: "none"
```

## 7. Notes on hardware

- SmolVLM-256M-Instruct fits full fine-tuning on a single 16GB GPU with
  batch size 2–4 + gradient accumulation. LoRA needs meaningfully less memory
  and trains a small fraction of the parameters (`count_trainable_params`
  prints this at the start of `train_lora.py`).
- For CPU-only smoke testing, set `training.max_steps: 5` and
  `data.max_samples: 16` in `config.yaml` to sanity-check the pipeline
  before a real run.

## 8. Repro / determinism

`src/utils.py::set_seed` seeds `random`, `numpy`, and `torch` (CPU + CUDA).
Set `training.seed` in the config to change it.

## 9. License

MIT — see `LICENSE`.
