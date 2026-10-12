"""Training and evaluation loops (LoRA fine-tuning with gradient accumulation)."""
from __future__ import annotations

import json
import logging
import math
import os
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainConfig

logger = logging.getLogger(__name__)


def build_dataloaders(train_ds, train_col, eval_ds, eval_col, cfg: TrainConfig):
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_col,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=eval_col,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    logger.info("train micro-batches/epoch: %d | optimizer steps/epoch: %d | eval micro-batches: %d",
                len(train_loader), steps, len(eval_loader))
    return train_loader, eval_loader


def forward_loss(model, batch, device="cuda"):
    """Cross-entropy loss over description tokens."""
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
        protein_input_ids=batch["protein_input_ids"].to(device),
        protein_attention_mask=batch["protein_attention_mask"].to(device),
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )[0]


def train_epoch(model, loader, optimizer, epoch, cfg: TrainConfig, device="cuda") -> float:
    """One epoch with gradient accumulation; returns mean loss over micro-batches.

    The loss is divided by gradient_accumulation_steps before backward() so accumulated
    gradients match the scale of one large batch. Clipping happens after accumulation.
    A final partial accumulation group is stepped rather than discarded.
    """
    model.train()
    accum = cfg.gradient_accumulation_steps
    params = [p for p in model.parameters() if p.requires_grad]
    total_loss, n_batches, total_gn, n_steps = 0.0, 0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"train {epoch}/{cfg.num_epochs}")
    for i, batch in enumerate(progress):
        loss = forward_loss(model, batch, device) / accum
        loss.backward()
        total_loss += loss.item() * accum
        n_batches += 1
        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            max_norm = float("inf") if cfg.gradient_clipping is None else cfg.gradient_clipping
            total_gn += float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))
            n_steps += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        progress.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

    mean_loss = total_loss / max(n_batches, 1)
    if math.isnan(mean_loss):
        raise ValueError("NaN in training loss")
    logger.info("[epoch %d] train_loss=%.4f lr=%.2e gradnorm=%.3f", epoch, mean_loss,
                optimizer.param_groups[0]["lr"], total_gn / max(n_steps, 1))
    return mean_loss


@torch.no_grad()
def eval_epoch(model, loader, epoch, cfg: TrainConfig, device="cuda") -> float:
    """Mean loss over the evaluation set without updating weights."""
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc=f"eval {epoch}/{cfg.num_epochs}"):
        total += forward_loss(model, batch, device).item()
        n += 1
    mean_loss = total / max(n, 1)
    logger.info("[epoch %d] eval_loss=%.4f", epoch, mean_loss)
    return mean_loss


def fit(model, train_loader, eval_loader, cfg: TrainConfig, out_dir: str, device="cuda") -> list[dict]:
    """Train for cfg.num_epochs, saving the adapter + optimizer state each epoch and
    writing history.json. Returns the per-epoch history."""
    os.makedirs(out_dir, exist_ok=True)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)

    history = []
    start = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, cfg, device)
        eval_loss = eval_epoch(model, eval_loader, epoch, cfg, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss,
                        "elapsed_s": time.time() - start})

        ckpt = os.path.join(out_dir, f"adapter_checkpoint_{epoch}")
        model.save_pretrained(ckpt)
        torch.save({"optimizer_state_dict": optimizer.state_dict()},
                   os.path.join(out_dir, f"optimizer_scheduler_checkpoint_{epoch}.pt"))
        logger.info("saved %s (elapsed %.2f h)", ckpt, (time.time() - start) / 3600)

    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("total training time: %.2f h", (time.time() - start) / 3600)
    return history
