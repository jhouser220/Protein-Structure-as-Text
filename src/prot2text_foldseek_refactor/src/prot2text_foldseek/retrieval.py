"""Foldseek index/search, hit parsing, and downsampling of easy training hits."""
from __future__ import annotations

import glob
import json
import logging
import os
import random
import shutil
import subprocess
import time
from typing import Sequence

import numpy as np

from .config import FoldseekConfig, RetrievalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- foldseek CLI
def run_command(cmd: Sequence[str], check: bool = True) -> str:
    """Run a command, log the tail of its combined output, and return it."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(list(cmd), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if output:
        logger.info(output[-4000:])
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (status {proc.returncode}): {' '.join(cmd)}")
    return output


def build_index(pdb_dir: str, index_db: str, cfg: FoldseekConfig) -> None:
    """Build the Foldseek structural index from the training PDBs (no-op if present)."""
    if os.path.exists(index_db + ".dbtype"):
        logger.info("index already built: %s", index_db)
        return
    start = time.time()
    run_command([cfg.binary, "createdb", pdb_dir, index_db])
    logger.info("createdb took %.0fs", time.time() - start)


def search_split(split: str, query_dir: str, index_db: str, out_path: str, tmp_dir: str,
                 cfg: FoldseekConfig) -> str:
    """Search one split's structures against the training index (skips if output exists)."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.info("exists, skipping: %s", os.path.basename(out_path))
        return out_path
    start = time.time()
    run_command([
        cfg.binary, "easy-search", query_dir, index_db, out_path, tmp_dir,
        "--format-output", cfg.output_format,
        "-s", str(cfg.sensitivity),
        "--max-seqs", str(cfg.max_seqs),
        "-e", str(cfg.evalue),
        "--threads", str(cfg.threads),
    ])
    logger.info("[%s] search took %.0fs", split, time.time() - start)
    return out_path


# ------------------------------------------------------ Drive backup / restore
def backup_exists(backup_dir: str, splits) -> bool:
    return (os.path.isdir(backup_dir)
            and any(f.startswith("trainDB") for f in os.listdir(backup_dir))
            and all(os.path.exists(os.path.join(backup_dir, f"hits_{s}.m8")) for s in splits))


def backup_artifacts(index_db: str, hit_paths: dict[str, str], backup_dir: str) -> None:
    """Copy the index files and hit tables to persistent storage so PDBs can be discarded."""
    os.makedirs(backup_dir, exist_ok=True)
    for path in glob.glob(index_db + "*"):
        if os.path.isfile(path):
            shutil.copy(path, backup_dir)
    for path in hit_paths.values():
        shutil.copy(path, backup_dir)


def restore_artifacts(backup_dir: str, scratch_dir: str, splits) -> tuple[str, dict[str, str]]:
    """Copy the index and hit tables back to local disk. Returns (index_db, {split: m8_path})."""
    os.makedirs(scratch_dir, exist_ok=True)
    for fname in os.listdir(backup_dir):
        if fname.startswith("trainDB"):
            shutil.copy(os.path.join(backup_dir, fname), os.path.join(scratch_dir, fname))
    index_db = os.path.join(scratch_dir, "trainDB")
    assert os.path.exists(index_db + ".dbtype"), "index files did not copy correctly"

    m8 = {}
    for split in splits:
        dst = os.path.join(scratch_dir, f"hits_{split}.m8")
        shutil.copy(os.path.join(backup_dir, f"hits_{split}.m8"), dst)
        m8[split] = dst
        logger.info("restored %s: %.1f MB", split, os.path.getsize(dst) / 1e6)
    return index_db, m8


# ------------------------------------------------------------- lookups / parsing
def build_train_desc(train_df) -> dict[str, str]:
    """accession -> function text for the training split (the retrieval payload)."""
    return {a: d for a, d in zip(train_df["AlphaFoldDB"], train_df["function"])
            if isinstance(a, str) and isinstance(d, str)}


def build_taxon_lookup(train_df) -> dict[str, str]:
    """training accession -> organism name."""
    return {str(a): str(t) for a, t in zip(train_df["AlphaFoldDB"], train_df["taxon"])
            if isinstance(a, str) and isinstance(t, str)}


def parse_hits(path: str, train_desc: dict[str, str], columns: list[str], keep_top: int) -> dict[str, list[dict]]:
    """Parse a Foldseek .m8 file into {query_accession: [hit, ...]}, best first.

    Drops self-hits and targets without a description; keeps at most `keep_top` per query.
    """
    hits: dict[str, list[dict]] = {}
    dropped_self = 0
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(columns):
                continue  # malformed line
            rec = dict(zip(columns, parts))
            query, target = rec["query"], rec["target"]
            if query == target:
                dropped_self += 1
                continue
            if target not in train_desc:
                continue
            bucket = hits.setdefault(query, [])
            if len(bucket) >= keep_top:
                continue
            bucket.append({
                "target": target,
                "fident": float(rec["fident"]),   # fractional sequence identity
                "qcov": float(rec["qcov"]),       # fraction of query covered
                "evalue": float(rec["evalue"]),
                "bits": float(rec["bits"]),
                "desc": train_desc[target],
            })
    logger.info("%s: %d queries with >=1 usable hit (self-hits dropped: %d)",
                os.path.basename(path), len(hits), dropped_self)
    return hits


def assert_no_self_hits(hits: dict[str, dict[str, list[dict]]]) -> None:
    for split, per_query in hits.items():
        for query, hit_list in per_query.items():
            assert all(h["target"] != query for h in hit_list), f"self-hit survived for {query} ({split})"


def summarize_hits(hits: dict[str, list[dict]], n_proteins: int, label: str) -> str:
    """One-line hit-rate and best-hit identity percentile summary for a split."""
    best = np.array([hs[0]["fident"] for hs in hits.values() if hs])
    rate = len(best) / max(n_proteins, 1)
    if not best.size:
        return f"{label:5s} hit-rate={rate:5.1%}  (no hits)"
    p = np.percentile(best, [10, 25, 50, 75, 90])
    return (f"{label:5s} hit-rate={rate:5.1%}  best-hit fident p10/25/50/75/90 = "
            + " / ".join(f"{v:.2f}" for v in p))


# ------------------------------------------------------------------ downsampling
def select_for_prompt(hits: list[dict], split: str, cfg: RetrievalConfig, rng: random.Random) -> list[dict]:
    """Choose hits for the prompt. Easy train hits (high identity) are dropped with
    probability 1 - easy_keep_prob so the model can't just copy a near-identical
    neighbour. Eval/test hits pass through unchanged."""
    selected = []
    for hit in hits:
        if split == "train" and hit["fident"] > cfg.easy_ident_threshold:
            if rng.random() > cfg.easy_keep_prob:
                continue
        selected.append(hit)
        if len(selected) >= cfg.max_hits_in_prompt:
            break
    return selected


def build_prompt_hits(hits: dict[str, dict[str, list[dict]]], splits, cfg: RetrievalConfig,
                      seed: int) -> dict[str, dict[str, list[dict]]]:
    """{split: {query_accession: [hit, ...]}} after downsampling."""
    rng = random.Random(seed)
    return {s: {q: select_for_prompt(hs, s, cfg, rng) for q, hs in hits[s].items()} for s in splits}


def save_prompt_hits(prompt_hits: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(prompt_hits, fh)
    logger.info("saved %s", path)


def load_prompt_hits(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
