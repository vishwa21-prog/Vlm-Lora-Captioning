"""Loading SmolVLM + its processor, and wrapping it with a PEFT/LoRA adapter."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model

try:
    # transformers >= 4.49 renamed this from AutoModelForVision2Seq.
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor


def load_base_model_and_processor(
    cfg: Dict[str, Any], dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.nn.Module, Any]:
    """Loads SmolVLM + processor from the model id in the config."""
    model_id = cfg["model"]["base_model_id"]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        _attn_implementation="eager",  # safest default across GPU/CPU
    )
    return model, processor


def apply_lora(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    """Wraps `model` with a LoRA adapter per config['lora']."""
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_lora_checkpoint(base_model: torch.nn.Module, checkpoint_dir: str) -> torch.nn.Module:
    """Loads a saved LoRA adapter on top of a freshly loaded base model."""
    return PeftModel.from_pretrained(base_model, checkpoint_dir)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        # Needed alongside gradient checkpointing when using PEFT, otherwise
        # the frozen base model's inputs won't require grad and backward
        # through the adapters silently produces no gradients.
        model.enable_input_require_grads()
