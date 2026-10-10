"""Dataset and collater that build retrieval-augmented Llama chat prompts.

Based on Prot2TextLightDataset / Prot2TextLightCollater from the Prot2Text-V2 paper.
"""
from __future__ import annotations

import random
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import DataConfig, RetrievalConfig

SYSTEM_MESSAGE = (
    "You are a scientific assistant specialized in protein function "
    "predictions. Given the sequence embeddings and other information "
    "of a protein, describe its function clearly and concisely in "
    "professional language. "
)
PLACEHOLDER_TOKEN = "<|reserved_special_token_1|>"
PLACEHOLDER_ID = 128003   # must match Esm2LlamaInstructConfig.placeholder_id
PAD_TOKEN = "<|reserved_special_token_0|>"


def load_tokenizers(esm_path: str, llama_path: str):
    """Load the ESM2 and Llama tokenizers and verify the placeholder token id."""
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, pad_token=PAD_TOKEN)
    placeholder_id = llama_tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    assert placeholder_id == PLACEHOLDER_ID, (
        f"placeholder id {placeholder_id} != expected {PLACEHOLDER_ID}")
    return esm_tokenizer, llama_tokenizer


class Prot2TextFoldseekDataset(Dataset):
    """Row-wise CSV access with attached Foldseek hits."""

    def __init__(self, df, hits):
        self.data = df.reset_index(drop=True)
        self.hits = hits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        acc = row["AlphaFoldDB"]
        return {
            "accession": acc,
            "taxon": row["taxon"],
            "sequence": row["sequence"],
            "function": row["function"],
            "hits": self.hits.get(acc, []),
        }


class Prot2TextFoldseekCollater:
    """Builds the chat prompt with taxonomy + Foldseek retrieval (no protein name).

    mode is "train" (returns labels) or "inference" (prompt only).
    include_retrieval=False gives the name-free baseline (taxonomy + sequence only).
    random_crop defaults to True in train and False at inference.
    """

    def __init__(self, sequence_tokenizer, description_tokenizer, *, mode, taxonomy_dropout,
                 retrieval_dropout, include_retrieval, max_hits, max_hit_chars,
                 max_sequence_length, max_description_length, taxon_lookup=None,
                 random_crop=None, seed=0):
        assert mode in ("train", "inference"), f"bad mode: {mode!r}"
        self.st = sequence_tokenizer
        self.dt = description_tokenizer
        self.mode = mode
        self.taxonomy_dropout = taxonomy_dropout
        self.retrieval_dropout = retrieval_dropout
        self.include_retrieval = include_retrieval
        self.max_hits = max_hits
        self.max_hit_chars = max_hit_chars
        self.max_sequence_length = max_sequence_length
        self.max_description_length = max_description_length
        self.taxon_lookup = taxon_lookup or {}
        self.random_crop = (mode == "train") if random_crop is None else random_crop
        self.rng = random.Random(seed)

    def format_hits(self, hits):
        """Serialise retrieval hits into one prompt string.

        Returns None (omit the segment: ablation or dropout), "none found" (search ran
        but found nothing), or the formatted hit list.
        """
        if not self.include_retrieval:
            return None
        if self.mode == "train" and self.rng.random() < self.retrieval_dropout:
            return None
        if not hits:
            return "none found"
        parts = []
        for i, h in enumerate(hits[: self.max_hits], 1):
            full = str(h["desc"])
            d = full[: self.max_hit_chars]
            if len(full) > self.max_hit_chars:
                d = d.rsplit(" ", 1)[0].rstrip(".,;: ") + "..."
            d = " ".join(d.split())
            org = self.taxon_lookup.get(h["target"], "unknown organism")
            parts.append(f"({i}) {h['fident'] * 100:.0f}% id, {h['qcov']:.2f} cov, "
                         f"E={h['evalue']:.0e}, org={org}: {d}")
        return " ".join(parts)

    def build_user_message(self, taxon, hits, n_placeholders):
        """`Taxon: ... ; Similar structures: ... ; Sequence embeddings: <ph>...`"""
        segments = [f"Taxon: {taxon}"]
        block = self.format_hits(hits)
        if block is not None:
            segments.append(f"Similar structures: {block}")
        segments.append("Sequence embeddings: " + PLACEHOLDER_TOKEN * n_placeholders)
        return " ; ".join(segments)

    def __call__(self, batch):
        accessions = [b["accession"] for b in batch]
        descriptions = [b["function"] if isinstance(b["function"], str) else "" for b in batch]

        taxons = [
            b["taxon"] if isinstance(b["taxon"], str) and self.rng.random() > self.taxonomy_dropout
            else "unknown"
            for b in batch
        ]

        seqs = []
        for b in batch:
            s = b["sequence"] if isinstance(b["sequence"], str) else ""
            if len(s) > self.max_sequence_length:
                start = (self.rng.randint(0, len(s) - self.max_sequence_length)
                         if self.random_crop else 0)
                s = s[start: start + self.max_sequence_length]
            seqs.append(s)

        self.st.padding_side = "right"
        tok_seq = self.st(seqs, truncation=True, padding="longest",
                          max_length=self.max_sequence_length + 2, return_tensors="pt")
        seq_ids, seq_mask = tok_seq["input_ids"], tok_seq["attention_mask"]
        seq_lens = seq_mask.sum(dim=1).tolist()   # placeholders must match exactly

        user_messages = [self.build_user_message(t, b["hits"], n)
                         for t, b, n in zip(taxons, batch, seq_lens)]
        convos = [[{"role": "system", "content": SYSTEM_MESSAGE},
                   {"role": "user", "content": um}] for um in user_messages]

        self.dt.padding_side = "left"
        tok_prompt = self.dt.apply_chat_template(
            convos, add_generation_prompt=True, tokenize=True,
            padding="longest", return_tensors="pt", return_dict=True)
        prompt_ids, prompt_mask = tok_prompt["input_ids"], tok_prompt["attention_mask"]

        self.dt.padding_side = "right"
        tok_desc = self.dt([d + self.dt.eos_token for d in descriptions],
                           add_special_tokens=False, truncation=True, padding="longest",
                           max_length=self.max_description_length, return_tensors="pt")
        desc_ids, desc_mask = tok_desc["input_ids"], tok_desc["attention_mask"]

        labels = desc_ids.clone()
        labels[desc_mask == 0] = -100

        out = {
            "accession": accessions,
            "protein_input_ids": seq_ids,
            "protein_attention_mask": seq_mask,
            "description_input_ids": desc_ids,
            "description_attention_mask": desc_mask,
        }
        if self.mode == "train":
            out.update({
                "input_ids": torch.cat([prompt_ids, desc_ids], dim=1),
                "attention_mask": torch.cat([prompt_mask, desc_mask], dim=1),
                "labels": torch.cat([torch.full_like(prompt_ids, -100), labels], dim=1),
            })
        else:
            out.update({"input_ids": prompt_ids, "attention_mask": prompt_mask})
        return out


class SplitFactory:
    """Creates matched (dataset, collater) pairs, holding all the shared state
    (CSVs, hits, tokenizers, lookups, configs) so callers only pass what varies."""

    def __init__(self, raw, prompt_hits, esm_tokenizer, llama_tokenizer, taxon_lookup,
                 data_cfg: DataConfig, retrieval_cfg: RetrievalConfig, seed: int):
        self.raw = raw
        self.prompt_hits = prompt_hits
        self.esm_tokenizer = esm_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.taxon_lookup = taxon_lookup
        self.data_cfg = data_cfg
        self.retrieval_cfg = retrieval_cfg
        self.seed = seed

    def make(self, split, mode, *, include_retrieval=True, taxonomy_dropout=None,
             retrieval_dropout=None, random_crop=None, trim=None):
        d = self.data_cfg
        df = self.raw[split]
        if trim and trim < len(df):
            df = df.sample(n=trim, random_state=self.seed)
        dataset = Prot2TextFoldseekDataset(df, self.prompt_hits[split])
        collater = Prot2TextFoldseekCollater(
            self.esm_tokenizer, self.llama_tokenizer,
            mode=mode,
            taxonomy_dropout=d.taxonomy_dropout if taxonomy_dropout is None else taxonomy_dropout,
            retrieval_dropout=d.retrieval_dropout if retrieval_dropout is None else retrieval_dropout,
            include_retrieval=include_retrieval,
            max_hits=self.retrieval_cfg.max_hits_in_prompt,
            max_hit_chars=self.retrieval_cfg.max_hit_chars,
            max_sequence_length=d.max_sequence_length,
            max_description_length=d.max_description_length,
            taxon_lookup=self.taxon_lookup,
            random_crop=random_crop,
            seed=self.seed,
        )
        return dataset, collater


def inspect_batch(batch, llama_tokenizer, index=0):
    """Decode one prompt with the placeholder run collapsed, and check that the
    placeholder count equals the encoder's valid positions for every sample."""
    ids = batch["input_ids"][index]
    n_ph = int((ids == PLACEHOLDER_ID).sum())
    text = llama_tokenizer.decode(ids, skip_special_tokens=False)
    text = re.sub(f"(?:{re.escape(PLACEHOLDER_TOKEN)}){{2,}}", f"{PLACEHOLDER_TOKEN}...x{n_ph}", text)

    n_placeholder = (batch["input_ids"] == PLACEHOLDER_ID).sum(dim=1)
    n_encoder = batch["protein_attention_mask"].sum(dim=1)
    assert torch.equal(n_placeholder, n_encoder), (
        f"placeholder/encoder mismatch: {n_placeholder} vs {n_encoder}")

    shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
    return text, shapes
