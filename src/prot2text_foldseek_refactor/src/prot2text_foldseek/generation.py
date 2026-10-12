"""Prediction generation under several prompt conditions."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import GenerationConfig
from .dataset import SplitFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    """One prompt condition. `tag` names the output file generation_{tag}.json."""
    tag: str
    include_retrieval: bool
    taxonomy_dropout: float = 0.0


# File names here are what the analysis notebook loads.
CONDITIONS = {
    "with": Condition("test_retrieval", include_retrieval=True),
    "without": Condition("test_noretrieval", include_retrieval=False),
    # no taxonomy (always "unknown") and no retrieval: sequence-only baseline
    "seq_only": Condition("seq_only", include_retrieval=False, taxonomy_dropout=1.0),
}


def merge_for_inference(model):
    """Merge LoRA into the base weights for faster, deterministic inference."""
    merged = model.merge_and_unload()
    merged.eval()
    if hasattr(merged, "llama_decoder"):
        merged.llama_decoder.config.use_cache = True
    return merged


@torch.no_grad()
def generate_split(model, factory: SplitFactory, llama_tokenizer, split: str, condition: Condition,
                   cfg: GenerationConfig, out_dir: str, trim=None, device="cuda") -> dict:
    """Generate descriptions for one split under one condition.

    All dropout is disabled except the condition's own, and drop_last=False so no protein
    is silently discarded. Saves {accession: {true, pred}} to out_dir/generation_{tag}.json.
    """
    dataset, collater = factory.make(
        split, "inference", include_retrieval=condition.include_retrieval,
        taxonomy_dropout=condition.taxonomy_dropout, retrieval_dropout=0.0, trim=trim)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collater,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    model.eval()
    results = {}
    for batch in tqdm(loader, desc=f"generate [{condition.tag}]"):
        out = model.generate(
            inputs=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            protein_input_ids=batch["protein_input_ids"].to(device),
            protein_attention_mask=batch["protein_attention_mask"].to(device),
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
            return_dict_in_generate=False,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
        )
        preds = llama_tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)
        refs = llama_tokenizer.batch_decode(batch["description_input_ids"], skip_special_tokens=True)
        for acc, ref, pred in zip(batch["accession"], refs, preds):
            results[acc] = {"true": ref.strip(), "pred": pred.strip()}

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"generation_{condition.tag}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("saved %s (%d proteins)", path, len(results))
    return results
