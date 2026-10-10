"""Text-generation metrics (adapted from the Prot2Text-V2 paper) and identity-bin stratification.

Shared by the training notebook and the analysis notebook.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import pandas as pd

from .config import MetricsConfig

# (low, high, label): test proteins are binned by best-hit sequence identity.
IDENT_BINS = [(0.00, 0.20, "<20%"), (0.20, 0.30, "20-30%"),
              (0.30, 0.50, "30-50%"), (0.50, 1.01, ">50%")]
NO_HIT = "no hit"
BIN_ORDER = [NO_HIT] + [name for _, _, name in IDENT_BINS]


# ------------------------------------------------------------------ binning
def identity_bin(fident) -> str:
    """Map a best-hit identity (or None/NaN) to its bin label."""
    if fident is None or (isinstance(fident, float) and math.isnan(fident)):
        return NO_HIT
    for lo, hi, name in IDENT_BINS:
        if lo <= fident < hi:
            return name
    return IDENT_BINS[-1][2]


def best_fident(split_hits: dict, accession: str):
    """Best-hit identity for an accession from {accession: [hit, ...]}, or None."""
    hits = split_hits.get(accession, [])
    return hits[0]["fident"] if hits else None


# ------------------------------------------------------------ cached loaders
@lru_cache(maxsize=None)
def _metric(name: str):
    import evaluate
    return evaluate.load(name)


@lru_cache(maxsize=None)
def _tokenizer(kind: str, name: str):
    from transformers import BertTokenizer, RobertaTokenizer
    return (RobertaTokenizer if kind == "roberta" else BertTokenizer).from_pretrained(name)


# ------------------------------------------------------------------- metrics
def compute_exact_match(predictions, references) -> float:
    """Fraction of predictions equal to the reference after lowercasing and
    stripping non-word characters."""
    def normalize(t):
        return re.sub(r"[^\w]", "", t.lower())
    return sum(normalize(p) == normalize(r) for p, r in zip(predictions, references)) / len(predictions)


def compute_bleu(predictions, references, max_order=4):
    return _metric("bleu").compute(predictions=predictions, references=references, max_order=max_order)


def compute_rouge(predictions, references):
    """rouge1 (unigram), rouge2 (bigram), rougeL (LCS)."""
    return _metric("rouge").compute(predictions=predictions, references=references)


def truncate_for_bert(texts, tokenizer, max_length):
    """Round-trip through the tokenizer so texts fit the BERTScore model's limit."""
    ids = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length,
                    return_tensors="pt")["input_ids"]
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def _mean_prf(scores: dict) -> dict:
    return {k: sum(scores[k]) / len(scores[k]) for k in ("precision", "recall", "f1")}


def compute_bert_score(predictions, references, cfg: MetricsConfig = MetricsConfig()) -> dict:
    """BERTScore P/R/F1 with RoBERTa-large and BioBERT-large."""
    bert = _metric("bertscore")
    n = cfg.bert_truncate_length

    rt = _tokenizer("roberta", cfg.roberta_model)
    roberta = bert.compute(predictions=truncate_for_bert(predictions, rt, n),
                           references=truncate_for_bert(references, rt, n), lang="en")
    bt = _tokenizer("bert", cfg.biobert_model)
    biobert = bert.compute(predictions=truncate_for_bert(predictions, bt, n),
                           references=truncate_for_bert(references, bt, n),
                           model_type=cfg.biobert_model, num_layers=cfg.biobert_num_layers)
    return {"roberta-large": _mean_prf(roberta), "biobert-large": _mean_prf(biobert)}


def compute_metrics(predictions, references, bert_score=True, verbose=True,
                    cfg: MetricsConfig = MetricsConfig()) -> dict:
    """Exact match, BLEU-2/4, ROUGE and (optionally, since it is slow) BERTScore."""
    out = {
        "n": len(predictions),
        "exact_match": compute_exact_match(predictions, references),
        "bleu2": compute_bleu(predictions, references, max_order=2),
        "bleu4": compute_bleu(predictions, references, max_order=4),
        "rouge": compute_rouge(predictions, references),
    }
    if bert_score:
        out["bert"] = compute_bert_score(predictions, references, cfg)
    if verbose:
        print(f"n            : {out['n']}")
        print(f"exact match  : {out['exact_match']:.4f}")
        print(f"BLEU-2       : {out['bleu2']['bleu']:.4f}")
        print(f"BLEU-4       : {out['bleu4']['bleu']:.4f}")
        r = out["rouge"]
        print(f"ROUGE-1/2/L  : {r['rouge1']:.4f} / {r['rouge2']:.4f} / {r['rougeL']:.4f}")
        for name, v in out.get("bert", {}).items():
            print(f"BERTScore {name:14s} P/R/F1: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f}")
    return out


def score_results(results: dict, **kwargs):
    """compute_metrics on a generation dict {acc: {"true", "pred"}}. Returns (accs, metrics)."""
    accs = list(results)
    return accs, compute_metrics([results[a]["pred"] for a in accs],
                                 [results[a]["true"] for a in accs], **kwargs)


# --------------------------------------------------------------- summaries
def flatten_metrics(m: dict) -> dict:
    """Pull the headline scalars from a compute_metrics() dict (NaN where absent)."""
    nan = float("nan")
    if not m:
        return {}
    return {
        "exact_match": m.get("exact_match", nan),
        "bleu2": m.get("bleu2", {}).get("bleu", nan),
        "bleu4": m.get("bleu4", {}).get("bleu", nan),
        "rouge1": m.get("rouge", {}).get("rouge1", nan),
        "rougeL": m.get("rouge", {}).get("rougeL", nan),
        "bert_rob_f1": m.get("bert", {}).get("roberta-large", {}).get("f1", nan),
        "bert_bio_f1": m.get("bert", {}).get("biobert-large", {}).get("f1", nan),
    }


def comparison_table(metrics_by_condition: dict[str, dict]) -> pd.DataFrame:
    """Rows = conditions, columns = headline metrics."""
    table = pd.DataFrame({name: flatten_metrics(m) for name, m in metrics_by_condition.items()}).T
    table.index.name = "condition"
    return table


def stratified_metrics(references: dict[str, str], predictions: dict[str, dict[str, str]],
                       bin_of: dict[str, str], treatment: str, control: str,
                       min_bin_size: int = 20, bert_score: bool = True,
                       cfg: MetricsConfig = MetricsConfig()) -> pd.DataFrame:
    """Compare two conditions within each identity bin.

    references:  {accession: reference text}
    predictions: {condition: {accession: predicted text}}
    bin_of:      {accession: bin label}
    Only accessions present in both conditions are scored. Bins smaller than
    `min_bin_size` are reported as a count only. Delta columns (d_*) are
    treatment minus control.
    """
    shared = set(references) & set(predictions[treatment]) & set(predictions[control])
    rows = []
    for name in BIN_ORDER:
        accs = [a for a in references if a in shared and bin_of.get(a) == name]
        row = {"bin": name, "n": len(accs)}
        if len(accs) < min_bin_size:
            rows.append(row)
            continue
        refs = [references[a] for a in accs]
        for tag in (treatment, control):
            m = compute_metrics([predictions[tag][a] for a in accs], refs,
                                bert_score=bert_score, verbose=False, cfg=cfg)
            row[f"bleu4_{tag}"] = m["bleu4"]["bleu"]
            row[f"rougeL_{tag}"] = m["rouge"]["rougeL"]
            if bert_score:
                row[f"bert_rob_{tag}"] = m["bert"]["roberta-large"]["f1"]
                row[f"bert_bio_{tag}"] = m["bert"]["biobert-large"]["f1"]
        for key in ("bleu4", "rougeL") + (("bert_rob", "bert_bio") if bert_score else ()):
            row[f"d_{key}"] = row[f"{key}_{treatment}"] - row[f"{key}_{control}"]
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------- per-protein scores
def per_protein_scores(predictions, references, cfg: MetricsConfig = MetricsConfig()) -> dict[str, list]:
    """Sentence-level ROUGE-L, BLEU-4, and RoBERTa/BioBERT BERTScore F1 per pair."""
    import sacrebleu
    from bert_score import score as bert_score_fn
    from rouge_score import rouge_scorer

    n = cfg.bert_truncate_length
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    bleu4 = [sacrebleu.sentence_bleu(p, [r]).score / 100 for p, r in zip(predictions, references)]

    rt = _tokenizer("roberta", cfg.roberta_model)
    _, _, f1_rob = bert_score_fn(truncate_for_bert(predictions, rt, n),
                                 truncate_for_bert(references, rt, n), lang="en", verbose=True)
    bt = _tokenizer("bert", cfg.biobert_model)
    _, _, f1_bio = bert_score_fn(truncate_for_bert(predictions, bt, n),
                                 truncate_for_bert(references, bt, n),
                                 model_type=cfg.biobert_model, num_layers=cfg.biobert_num_layers,
                                 verbose=True)
    return {"rougeL": rouge_l, "bleu4": bleu4, "bert_rob": f1_rob.tolist(), "bert_bio": f1_bio.tolist()}
