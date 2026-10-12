"""Loading run artifacts and assembling the per-protein analysis table."""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field

import pandas as pd

from .config import Paths
from .metrics import identity_bin

logger = logging.getLogger(__name__)

LOCAL_CONDITIONS = ("seq_only", "without", "with")   # prediction conditions generated locally
GENERATION_FILES = {
    "seq_only": "generation_seq_only.json",
    "without": "generation_test_noretrieval.json",
    "with": "generation_test_retrieval.json",
}


def _load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


@dataclass
class RunArtifacts:
    generations: dict            # {condition: {acc: {"true", "pred"}}}
    prompt_hits: dict            # {split: {acc: [hit, ...]}}
    test_df: pd.DataFrame
    train_taxon: dict            # train accession -> organism
    agg_metrics: dict            # metrics.json (with/without retrieval)
    agg_metrics_seq: dict        # metrics_seq_only.json
    stratified: pd.DataFrame     # stratified_metrics.csv from the training run
    history: list = field(default_factory=list)


def load_run_artifacts(paths: Paths) -> RunArtifacts:
    """Load everything the training notebook wrote, plus the test/train CSVs."""
    generations = {}
    for cond, fname in GENERATION_FILES.items():
        data = _load_json(os.path.join(paths.out_dir, fname), {})
        logger.info("%s: %d proteins <- %s", cond, len(data), fname)
        generations[cond] = data

    train_df = pd.read_csv(os.path.join(paths.csv_dir, "train.csv"), usecols=["AlphaFoldDB", "taxon"])
    train_taxon = {str(a): str(t) for a, t in zip(train_df["AlphaFoldDB"], train_df["taxon"])
                   if isinstance(a, str) and isinstance(t, str)}

    return RunArtifacts(
        generations=generations,
        prompt_hits=_load_json(paths.prompt_hits_path),
        test_df=pd.read_csv(os.path.join(paths.csv_dir, "test.csv")),
        train_taxon=train_taxon,
        agg_metrics=_load_json(os.path.join(paths.out_dir, "metrics.json"), {}),
        agg_metrics_seq=_load_json(os.path.join(paths.out_dir, "metrics_seq_only.json"), {}),
        stratified=pd.read_csv(os.path.join(paths.out_dir, "stratified_metrics.csv")),
        history=_load_json(os.path.join(paths.out_dir, "history.json"), []),
    )


def load_external_baseline(predict_path: str, metrics_path: str):
    """Load the externally generated baseline: ({acc: pred}, matched accession set, metrics dict)."""
    with open(predict_path, "rb") as fh:
        predictions = pickle.load(fh)
    with open(metrics_path) as fh:
        payload = json.load(fh)
    return predictions, set(payload["matched_names"]), payload["baseline"]


def build_master_frame(art: RunArtifacts, baseline_predictions: dict) -> pd.DataFrame:
    """One row per test protein present in every local condition: taxon, reference, best-hit
    metadata, identity bin, and one pred_<condition> column per condition (plus baseline)."""
    conditions = {k: v for k, v in art.generations.items() if v}
    shared = set.intersection(*[set(v) for v in conditions.values()])
    logger.info("%d accessions in all local conditions; %d overlap with external baseline",
                len(shared), len(shared & set(baseline_predictions)))

    meta = art.test_df.set_index("AlphaFoldDB")
    reference_source = conditions.get("with") or next(iter(conditions.values()))

    rows = []
    for acc in shared:
        hits = art.prompt_hits["test"].get(acc, [])
        best = hits[0] if hits else None
        row = {
            "accession": acc,
            "taxon": meta.loc[acc, "taxon"] if acc in meta.index else "",
            "reference": reference_source[acc]["true"],
            "best_hit_target": best["target"] if best else None,
            "best_hit_fident": best["fident"] if best else None,
            "best_hit_qcov": best["qcov"] if best else None,
            "best_hit_evalue": best["evalue"] if best else None,
            "best_hit_desc": best["desc"] if best else None,
            # hits don't store organism, so look it up from the training CSV
            "best_hit_org": art.train_taxon.get(best["target"]) if best else None,
            "pred_baseline": baseline_predictions.get(acc),
        }
        for cond, gen in conditions.items():
            row[f"pred_{cond}"] = gen[acc]["pred"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("accession")
    df["fident_bin"] = df["best_hit_fident"].apply(identity_bin)
    return df


def copy_rate(df: pd.DataFrame, pred_columns: list[str]) -> pd.DataFrame:
    """How closely each prediction mirrors the best retrieved description (ROUGE-L F1 between
    prediction and best_hit_desc), averaged per identity bin. High values on high-identity bins
    suggest the model is copying retrieval rather than predicting."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    has_hit = df["best_hit_desc"].notna()
    out = pd.DataFrame(index=df.index[has_hit])
    out["fident_bin"] = df.loc[has_hit, "fident_bin"]
    for col in pred_columns:
        preds = df.loc[has_hit, col]
        out[f"copy_{col}"] = [
            scorer.score(str(hit), str(p))["rougeL"].fmeasure if isinstance(p, str) else float("nan")
            for hit, p in zip(df.loc[has_hit, "best_hit_desc"], preds)
        ]
    return out
