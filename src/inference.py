"""Generate a caption for a single image from any checkpoint (base / full /
LoRA).

Usage:
    python -m src.inference --config config/config.yaml \
        --checkpoint outputs/checkpoints/lora --image path/to/image.jpg

    # base model, no --checkpoint needed:
    python -m src.inference --config config/config.yaml --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from src.data import CAPTION_INSTRUCTION
from src.model_utils import load_base_model_and_processor, load_lora_checkpoint
from src.utils import load_config


def generate_caption(
    model, processor, image: Image.Image, cfg, device: str = "cpu"
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_INSTRUCTION},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt").to(device)

    gen_cfg = cfg["generation"]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            num_beams=gen_cfg["num_beams"],
            do_sample=gen_cfg["do_sample"],
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return caption


def load_model_for_inference(cfg, checkpoint: str | None, device: str):
    dtype = torch.bfloat16 if (device == "cuda" and cfg["training"].get("bf16", True)) else torch.float32
    model, processor = load_base_model_and_processor(cfg, dtype=dtype)

    if checkpoint and os.path.isdir(checkpoint):
        adapter_config = os.path.join(checkpoint, "adapter_config.json")
        if os.path.isfile(adapter_config):
            model = load_lora_checkpoint(model, checkpoint)
        else:
            # A full-fine-tune checkpoint directory: reload full weights.
            try:
                from transformers import AutoModelForImageTextToText as _AutoModelCls
            except ImportError:
                from transformers import AutoModelForVision2Seq as _AutoModelCls

            model = _AutoModelCls.from_pretrained(checkpoint, torch_dtype=dtype)

    model.to(device)
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a saved LoRA or full-fine-tune checkpoint. "
                              "Omit to use the base (untrained) model.")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model_for_inference(cfg, args.checkpoint, device)
    image = Image.open(args.image)

    caption = generate_caption(model, processor, image, cfg, device=device)
    print(caption)


if __name__ == "__main__":
    main()
