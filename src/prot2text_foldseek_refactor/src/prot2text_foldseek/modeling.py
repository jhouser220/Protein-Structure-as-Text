"""Build Prot2Text-V2 from the released HF checkpoint and attach a fresh LoRA adapter."""
from __future__ import annotations

import json
import logging
import os
import sys

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import EsmConfig, EsmModel, LlamaConfig, LlamaForCausalLM

from .config import LoraSettings

logger = logging.getLogger(__name__)


def import_model_classes(repo_root: str):
    """Import the architecture classes from the Prot2Text-V2 repo clone."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from models import Esm2LlamaInstructForCausalLM, ModalityAdapter, ModalityAdapterConfig
    return ModalityAdapter, ModalityAdapterConfig, Esm2LlamaInstructForCausalLM


def _load_state_dict(local_dir: str) -> dict:
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        shards = [f for f in os.listdir(local_dir) if f.endswith(".safetensors")]
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(os.path.join(local_dir, shard)))
    return state_dict


def build_base_from_hf(hf_model_id: str, repo_root: str, dtype=torch.bfloat16):
    """Instantiate the architecture and load the released Prot2Text-V2 weights."""
    ModalityAdapter, ModalityAdapterConfig, Esm2LlamaInstructForCausalLM = import_model_classes(repo_root)

    logger.info("downloading weights for: %s", hf_model_id)
    local_dir = snapshot_download(hf_model_id, allow_patterns=["*.json", "*.safetensors"])
    with open(os.path.join(local_dir, "config.json")) as fh:
        hf_config = json.load(fh)

    base = Esm2LlamaInstructForCausalLM(
        esm_encoder=EsmModel(EsmConfig(**hf_config["esm_config"]), add_pooling_layer=False),
        adapter=ModalityAdapter(ModalityAdapterConfig(**hf_config["adapter_config"])),
        llama_decoder=LlamaForCausalLM(LlamaConfig(**hf_config["llama_config"])),
        placeholder_id=hf_config.get("placeholder_id", 128003),
    )

    state_dict = _load_state_dict(local_dir)
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    logger.info("loaded %d tensors missing=%d unexpected=%d", len(state_dict), len(missing), len(unexpected))
    if missing:
        logger.warning("e.g. missing: %s", missing[:5])
    if unexpected:
        logger.warning("e.g. unexpected: %s", unexpected[:5])
    assert not missing, "state dict did not fully populate the local architecture - check config keys"

    sd = base.state_dict()
    assert any(k.startswith("adapter.fc1") for k in sd), "no adapter.fc1 in released weights"
    w = sd["adapter.fc1.weight"].float()
    logger.info("adapter.fc1 %s |w| mean=%.4f std=%.4f", tuple(w.shape), w.abs().mean(), w.std())
    return base.to(dtype)


def attach_lora(base, lora: LoraSettings):
    """Wrap `base` with a fresh LoRA adapter on the Llama decoder projections."""
    logger.info("initializing fresh LoRA (r=%d)", lora.rank)
    model = get_peft_model(base, LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        init_lora_weights=True,                 # A random, B zero: standard LoRA init
        target_modules=list(lora.targets),
        # modules_to_save=None keeps the stage-1 projector frozen
        modules_to_save=None if lora.fix_modality_adapter else ["adapter.fc1", "adapter.fc2"],
    ))
    model.print_trainable_parameters()
    n_lora = sum("lora_A" in n for n, _ in model.named_modules())
    logger.info("LoRA on %d decoder modules", n_lora)
    return model


def enable_gradient_checkpointing(model) -> None:
    """Enable non-reentrant checkpointing on both sub-models, unwrapping the PEFT layer."""
    model.train()
    base = model.base_model.model if hasattr(model, "base_model") else model
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    base.llama_decoder.gradient_checkpointing_enable(**kwargs)
    base.esm_encoder.gradient_checkpointing_enable(**kwargs)
    base.llama_decoder.config.use_cache = False
    logger.info("gradient checkpointing enabled (non-reentrant)")


def build_model(hf_model_id: str, repo_root: str, lora: LoraSettings, device: str = "cuda",
                dtype=torch.bfloat16):
    """Base model + LoRA, on `device`, with gradient checkpointing if configured."""
    model = attach_lora(build_base_from_hf(hf_model_id, repo_root, dtype), lora).to(device)
    if lora.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if str(device).startswith("cuda"):
        logger.info("allocated: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    return model
