"""Dataset loading + preprocessing for domain image-captioning with SmolVLM.

SmolVLM (like other Idefics3-family models) expects chat-formatted inputs:
a user turn containing an <image> placeholder + instruction, and an
assistant turn containing the target caption. We build that turn structure
with the processor's chat template, then mask the prompt tokens out of the
loss so the model is only trained to produce the caption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, load_dataset
from PIL import Image
import torch

CAPTION_INSTRUCTION = "Describe this image in one detailed sentence."


def load_caption_dataset(cfg: Dict[str, Any]) -> DatasetDict:
    """Loads and splits the dataset described in cfg['data'] into
    train/validation/test DatasetDict, regardless of whether it comes
    from the Hub or a local imagefolder.
    """
    data_cfg = cfg["data"]

    if data_cfg["dataset_name"] == "imagefolder":
        if not data_cfg.get("data_dir"):
            raise ValueError("data.data_dir must be set when dataset_name == 'imagefolder'")
        raw = load_dataset("imagefolder", data_dir=data_cfg["data_dir"])
    else:
        raw = load_dataset(data_cfg["dataset_name"])

    # Normalize to a single split we can re-split ourselves, so the
    # train/val/test fractions in the config are respected consistently
    # across whatever the source dataset happened to ship with.
    if isinstance(raw, DatasetDict):
        base_split = data_cfg.get("train_split", "train")
        full = raw[base_split]
    else:
        full = raw

    if data_cfg.get("max_samples"):
        full = full.select(range(min(data_cfg["max_samples"], len(full))))

    test_frac = data_cfg.get("test_fraction", 0.1)
    val_frac = data_cfg.get("val_fraction", 0.1)
    seed = data_cfg.get("seed", 42)

    split1 = full.train_test_split(test_size=test_frac, seed=seed)
    train_val, test = split1["train"], split1["test"]

    remaining_val_frac = val_frac / (1 - test_frac)
    split2 = train_val.train_test_split(test_size=remaining_val_frac, seed=seed)
    train, val = split2["train"], split2["test"]

    return DatasetDict(train=train, validation=val, test=test)


def _to_rgb(image: Any) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")


@dataclass
class CaptionCollator:
    """Builds SmolVLM chat-formatted, tokenized, loss-masked batches.

    processor: an AutoProcessor loaded for the SmolVLM checkpoint.
    image_column / caption_column: dataset column names.
    max_target_length: truncation length for the caption/target text.
    train: if True, labels are included (prompt tokens masked with -100).
           if False (inference/eval), only the prompt is built.
    """

    processor: Any
    image_column: str = "image"
    caption_column: str = "text"
    max_target_length: int = 64
    train: bool = True

    def _build_prompt_messages(self) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CAPTION_INSTRUCTION},
                ],
            }
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [_to_rgb(ex[self.image_column]) for ex in batch]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._build_prompt_messages(), add_generation_prompt=True
            )
            for _ in batch
        ]

        if not self.train:
            inputs = self.processor(
                text=prompt_texts, images=images, return_tensors="pt", padding=True
            )
            return inputs

        captions = [str(ex[self.caption_column]).strip() for ex in batch]
        full_texts = [p + c + self.processor.tokenizer.eos_token for p, c in zip(prompt_texts, captions)]

        inputs = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True,
            truncation=True, max_length=None,
        )

        # Mask everything up to (and including) the prompt so loss is only
        # computed on the caption tokens.
        labels = inputs["input_ids"].clone()
        for i, prompt_text in enumerate(prompt_texts):
            prompt_len = len(
                self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            )
            labels[i, :prompt_len] = -100
        # Mask padding tokens too.
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[inputs["input_ids"] == pad_id] = -100

        inputs["labels"] = labels
        return inputs
