"""Central configuration.

Every tunable value lives here as a dataclass field, so functions elsewhere take
a config object instead of reading notebook-level globals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DRIVE_ROOT = "/content/drive/MyDrive/Colab_Notebooks/266_Final_Project"


@dataclass
class Paths:
    drive_root: str = DRIVE_ROOT
    scratch_dir: str = "/data/foldseek_work"   # fast local disk (PDBs, tmp, working index)
    run_name: str = "drop01run"

    @property
    def repo_root(self) -> str:
        return os.path.join(self.drive_root, "Prot2Text-V2-main")

    @property
    def src_dir(self) -> str:
        return os.path.join(self.drive_root, "src")

    @property
    def csv_dir(self) -> str:
        return os.path.join(self.drive_root, "Prot2TextDataset")

    @property
    def pdb_root(self) -> str:
        return os.path.join(self.scratch_dir, "alphafold_pdbs")

    @property
    def index_db(self) -> str:
        return os.path.join(self.scratch_dir, "trainDB")

    @property
    def index_backup_dir(self) -> str:
        """Drive copy of the Foldseek index and hit tables."""
        return os.path.join(self.drive_root, "foldseek_index")

    @property
    def prompt_hits_path(self) -> str:
        return os.path.join(self.drive_root, "foldseek_work", "prompt_hits.json")

    @property
    def failed_downloads_path(self) -> str:
        return os.path.join(self.scratch_dir, "failed_downloads.json")

    @property
    def out_dir(self) -> str:
        """Run outputs: adapter checkpoints, generations, metrics."""
        return os.path.join(self.drive_root, "foldseek_lora", self.run_name)

    def hits_m8(self, split: str) -> str:
        return os.path.join(self.scratch_dir, f"hits_{split}.m8")


@dataclass(frozen=True)
class ModelNames:
    esm_path: str = "facebook/esm2_t36_3B_UR50D"
    llama_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_model_id: str = "xiao-fei/Prot2Text-V2-11B-Instruct-hf"


@dataclass
class DownloadConfig:
    file_template: str = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
    api_template: str = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    workers: int = 16
    retries: int = 2          # attempts beyond the first, network errors only
    timeout: int = 15         # seconds per request
    backoff: float = 0.5      # seconds between retries
    progress_every: int = 2000
    user_agent: str = "prot2text-foldseek/1.0"


@dataclass
class FoldseekConfig:
    binary: str = "foldseek"
    sensitivity: float = 9.5   # 1-10; higher searches harder for remote homologs
    max_seqs: int = 300        # hits retained per query
    evalue: float = 10.0
    threads: int = 44
    output_format: str = "query,target,fident,alnlen,evalue,bits,qcov,tcov"

    @property
    def columns(self) -> list[str]:
        return self.output_format.split(",")


@dataclass
class RetrievalConfig:
    keep_top_hits: int = 10          # hits kept per query when parsing
    easy_ident_threshold: float = 0.60   # train hits above this identity are "easy"
    easy_keep_prob: float = 0.25     # probability of keeping an easy train hit
    max_hits_in_prompt: int = 2
    max_hit_chars: int = 200         # truncation of each retrieved description


@dataclass
class DataConfig:
    taxonomy_dropout: float = 0.1
    retrieval_dropout: float = 0.1
    max_sequence_length: int = 1021  # amino acids fed to ESM2
    max_description_length: int = 512  # tokens
    trim_train: int | None = 100_000
    trim_eval: int | None = 500
    trim_test: int | None = None


@dataclass
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    fix_modality_adapter: bool = True   # keep the stage-1 projector frozen
    gradient_checkpointing: bool = True
    targets: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )


@dataclass
class TrainConfig:
    num_epochs: int = 1
    batch_size: int = 4               # proteins per micro-batch
    gradient_accumulation_steps: int = 16
    gradient_clipping: float | None = 1.0
    learning_rate: float = 1e-4
    num_workers: int = 4


@dataclass
class GenerationConfig:
    batch_size: int = 4
    max_new_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = False
    eos_token_id: int = 128009   # <|eot_id|>
    pad_token_id: int = 128002   # <|reserved_special_token_0|>
    num_workers: int = 2


@dataclass(frozen=True)
class MetricsConfig:
    bert_truncate_length: int = 495
    roberta_model: str = "FacebookAI/roberta-large"
    biobert_model: str = "dmis-lab/biobert-large-cased-v1.1"
    biobert_num_layers: int = 24


@dataclass
class Config:
    seed: int = 0
    splits: tuple[str, ...] = ("train", "eval", "test")
    paths: Paths = field(default_factory=Paths)
    models: ModelNames = field(default_factory=ModelNames)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    foldseek: FoldseekConfig = field(default_factory=FoldseekConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
