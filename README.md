# Prot2Text-V2 + Foldseek Retrieval

Fine-tunes [Prot2Text-V2](https://huggingface.co/xiao-fei/Prot2Text-V2-11B-Instruct-hf) with LoRA to predict protein function descriptions, augmenting each prompt with the function text of structurally similar training proteins retrieved by [Foldseek](https://github.com/steineggerlab/foldseek).

The project has two notebooks and a small package that holds all the logic:

| Notebook | Purpose |
|---|---|
| `train.ipynb` | Download structures, run Foldseek retrieval, train, generate, and score |
| `analysis.ipynb` | Load the training outputs and interpret predictions per protein and per identity bin |

## How it works

1. **Structures.** AlphaFold PDB files are downloaded for every protein in the `train`, `eval` and `test` splits.
2. **Retrieval.** A Foldseek index is built from the training structures. Every split is searched against it, and each query keeps its top hits (self-hits are removed).
3. **Downsampling.** Training hits with sequence identity above a threshold ("easy" hits) are mostly dropped, so the model cannot learn to copy a near-identical neighbour. Eval and test hits are left untouched.
4. **Prompting.** Each protein gets a Llama chat prompt of the form `Taxon: ... ; Similar structures: ... ; Sequence embeddings: <placeholders>`. The ESM2 encoder's output replaces the placeholders.
5. **Training.** LoRA is applied to the Llama decoder's attention and MLP projections. The modality adapter stays frozen by default.
6. **Generation.** Test-set predictions are produced under three conditions:
   - `with`: taxonomy and retrieved structures in the prompt.
   - `without`: taxonomy only.
   - `seq_only`: neither (taxonomy is always "unknown").
7. **Evaluation.** Exact match, BLEU-2/4, ROUGE and BERTScore (RoBERTa-large and BioBERT-large), overall and stratified by best-hit identity.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── train.ipynb
├── analysis.ipynb
└── src/
    └── prot2text_foldseek/
        ├── config.py       # dataclasses for all paths and hyperparameters
        ├── structures.py   # AlphaFold downloads
        ├── retrieval.py    # Foldseek index/search, hit parsing, downsampling, Drive backup/restore
        ├── dataset.py      # Dataset, collater, SplitFactory, prompt inspection
        ├── modeling.py     # load the HF checkpoint, attach LoRA, gradient checkpointing
        ├── training.py     # train/eval loops and fit()
        ├── generation.py   # generate_split and the prompt conditions
        ├── metrics.py      # metrics, identity bins, stratified and per-protein scoring
        └── analysis.py     # artifact loading and the per-protein master table
```

## File descriptions

### Top-level files

| File | Description |
|---|---|
| `README.md` | This document. |
| `requirements.txt` | Pinned and minimum versions of the Python dependencies. Installed by the first cell of `train.ipynb`. |
| `train.ipynb` | End-to-end training notebook. Sections: setup (Drive, dependencies, Foldseek binary), data loading, retrieval (download, index, search, backup or restore), hit parsing and downsampling, datasets and collaters with a prompt sanity check, model construction, training, generation under the three conditions, and evaluation (overall and stratified by identity bin). Contains no logic of its own; it calls the package. |
| `analysis.ipynb` | Post-hoc analysis notebook. Loads the training outputs and the external baseline, builds the per-protein table, compares aggregate and stratified metrics, shows example proteins by identity bin, runs the copying analysis, computes per-protein scores, and exports a CSV. |

### Package: `src/prot2text_foldseek/`

| File | Description | Main contents |
|---|---|---|
| `__init__.py` | Marks the directory as a package. Contains only a docstring. | n/a |
| `config.py` | Single source of truth for paths and hyperparameters. Nothing else in the package hardcodes a tunable value. | `Paths` (with derived-path properties such as `index_db` and `out_dir`), `ModelNames`, `DownloadConfig`, `FoldseekConfig`, `RetrievalConfig`, `DataConfig`, `LoraSettings`, `TrainConfig`, `GenerationConfig`, `MetricsConfig`, and the top-level `Config` that bundles them all. |
| `structures.py` | Downloads AlphaFold structures with a thread pool. Tries the direct v4 file URL first and falls back to the prediction API. Skips files already on disk. | `download_one`, `download_split_structures`, `download_all` (also records failed accessions to JSON). |
| `retrieval.py` | Everything between the PDB files and the hits placed in prompts. Wraps the Foldseek CLI, parses its output, and applies the easy-hit downsampling. Also backs up and restores the index on Drive. | `run_command`, `build_index`, `search_split`, `backup_artifacts` / `restore_artifacts` / `backup_exists`, `build_train_desc`, `build_taxon_lookup`, `parse_hits`, `assert_no_self_hits`, `summarize_hits`, `select_for_prompt`, `build_prompt_hits`, `save_prompt_hits` / `load_prompt_hits`. |
| `dataset.py` | Builds the model inputs. The collater assembles the Llama chat prompt (taxonomy, retrieved structures, placeholder tokens) and tokenizes the ESM2 sequence. `SplitFactory` holds the shared state (CSVs, hits, tokenizers) so callers only pass what varies per split or condition. | `load_tokenizers`, `Prot2TextFoldseekDataset`, `Prot2TextFoldseekCollater`, `SplitFactory`, `inspect_batch` (decodes a prompt and checks that placeholder counts match the encoder's valid positions), and the prompt constants (`SYSTEM_MESSAGE`, `PLACEHOLDER_TOKEN`, `PLACEHOLDER_ID`). |
| `modeling.py` | Builds the model. Loads the released Prot2Text-V2 weights into the architecture from the upstream repo, attaches a fresh LoRA adapter to the Llama decoder, and enables gradient checkpointing. | `import_model_classes`, `build_base_from_hf`, `attach_lora`, `enable_gradient_checkpointing`, `build_model`. |
| `training.py` | Training and evaluation loops for the LoRA fine-tune, with gradient accumulation and clipping. `fit` also saves the adapter and optimizer state after each epoch and writes `history.json`. | `build_dataloaders`, `forward_loss`, `train_epoch`, `eval_epoch`, `fit`. |
| `generation.py` | Test-set prediction under different prompt conditions. The output file names produced here are the ones `analysis.py` reads. | `Condition`, `CONDITIONS` (`with`, `without`, `seq_only`), `merge_for_inference` (merges LoRA into the base weights), `generate_split`. |
| `metrics.py` | Text-generation metrics and identity-bin logic, shared by both notebooks. Metric models and tokenizers are loaded once and cached. | `IDENT_BINS`, `identity_bin`, `best_fident`, `compute_exact_match`, `compute_bleu`, `compute_rouge`, `compute_bert_score`, `compute_metrics`, `score_results`, `flatten_metrics`, `comparison_table`, `stratified_metrics`, `per_protein_scores`. |
| `analysis.py` | Loads what the training run wrote and assembles the per-protein table used by `analysis.ipynb`. | `RunArtifacts`, `load_run_artifacts`, `load_external_baseline`, `build_master_frame`, `copy_rate`, plus `GENERATION_FILES` (the condition-to-filename map). |

### How the pieces depend on each other

Every module imports `config.py`. Beyond that, the only cross-module imports are `generation.py` -> `dataset.py` (for `SplitFactory`) and `analysis.py` -> `metrics.py` (for `identity_bin`). The notebooks wire the rest together by passing data between calls.

`train.ipynb` uses `structures`, `retrieval`, `dataset`, `modeling`, `training`, `generation` and `metrics`. `analysis.ipynb` uses `analysis` and `metrics`.

## Requirements

- Google Colab (or similar) with a GPU that fits the 11B model in bfloat16 with LoRA and gradient checkpointing.
- Google Drive for persistent storage.
- A clone of the Prot2Text-V2 code repository at `Prot2Text-V2-main`. Only its `models` package is used (`Esm2LlamaInstructForCausalLM`, `ModalityAdapter`, `ModalityAdapterConfig`).
- Access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face (the tokenizer is loaded from there).
- The Prot2Text dataset CSVs (`train.csv`, `eval.csv`, `test.csv`) with columns `AlphaFoldDB, Full Name, taxon, sequence, function`.
- Python packages: see `requirements.txt`. The Foldseek binary is downloaded by the first cells of `train.ipynb`.

## Setup

Arrange Drive like this (the root and subfolder names are defined in `config.py` and can be changed there):

```
MyDrive/Colab_Notebooks/266_Final_Project/
├── src/                      # copy this repo's src/ and requirements.txt here
├── Prot2Text-V2-main/        # upstream repo clone
└── Prot2TextDataset/         # train.csv, eval.csv, test.csv
```

If your root differs, change `DRIVE_ROOT` in `config.py` and the `sys.path.insert(...)` line in each notebook.

## Usage

### 1. Train and generate

Open `train.ipynb` in Colab and run the cells top to bottom. The notebook only configures and calls the package.

- **First run:** downloads structures, builds the Foldseek index, searches all splits, and backs the index and hit tables up to `foldseek_index/` on Drive.
- **Later runs:** if that backup exists it is restored instead, and structures are not re-downloaded.

To change settings, edit `cfg` in the notebook before the later cells, for example:

```python
cfg = Config()
cfg.train.num_epochs = 2
cfg.lora.rank = 32
cfg.data.trim_train = 20_000   # smaller run for debugging
cfg.paths.run_name = "my_experiment"
```

### 2. Analyze

Open `analysis.ipynb` and run it after training has finished. It also needs the external baseline files (`Full_run/test_baseline_predictions.pkl` and the baseline metrics JSON); their paths are set in the second code cell.

## Configuration

All tunable values live in `src/prot2text_foldseek/config.py`:

| Group | Controls |
|---|---|
| `Paths` | Drive root, scratch directory, run name, and derived paths |
| `DownloadConfig` | Concurrency, retries, timeouts, AlphaFold URL templates |
| `FoldseekConfig` | Sensitivity, max hits, e-value, threads, output format |
| `RetrievalConfig` | Hits kept, easy-hit identity threshold and keep probability, hits per prompt, description truncation |
| `DataConfig` | Taxonomy and retrieval dropout, max sequence and description length, split trimming |
| `LoraSettings` | Rank, alpha, dropout, which modules are targeted, frozen adapter, gradient checkpointing |
| `TrainConfig` | Epochs, batch size, gradient accumulation, clipping, learning rate |
| `GenerationConfig` | Batch size, max new tokens, beam search/sampling, special-token ids |
| `MetricsConfig` | BERTScore models and truncation length |

## Outputs

Written to `foldseek_lora/<run_name>/` on Drive:

| File | Contents |
|---|---|
| `adapter_checkpoint_<epoch>/` | LoRA adapter weights |
| `optimizer_scheduler_checkpoint_<epoch>.pt` | Optimizer state |
| `history.json` | Per-epoch train and eval loss and elapsed time |
| `generation_test_retrieval.json`, `generation_test_noretrieval.json`, `generation_seq_only.json` | `{accession: {"true", "pred"}}` for each condition |
| `metrics.json`, `metrics_seq_only.json` | Aggregate metrics |
| `stratified_metrics.csv` | With vs. without retrieval, per identity bin |
| `all_predictions_analysis.csv` | Per-protein table written by the analysis notebook |

`prompt_hits.json` (the downsampled hits actually used in prompts) is saved to `foldseek_work/` on Drive.

## Notes and caveats

- **Eval loss** is computed without taxonomy/retrieval dropout and without random cropping, so it is deterministic.
- **Training steps:** a final partial gradient-accumulation group is stepped rather than discarded.
- **Foldseek IDs:** the pipeline assumes Foldseek query and target IDs match the accession keys in the CSVs. Check a few hits after the first search if you change the PDB file naming.
- **Copying analysis:** the `copy_rate` section of the analysis notebook (ROUGE-L between each prediction and its best retrieved description) is a reconstruction of an earlier cell that referenced undefined variables. Confirm it measures what you intend.
- **Not covered:** the original upstream scripts (`train_instruct.py`, `train_contrast.py`) support multi-GPU DDP, a learning-rate scheduler, and checkpoint resume. This project is single-GPU and has no scheduler or resume option.

## Acknowledgements

Built on [Prot2Text-V2](https://huggingface.co/xiao-fei/Prot2Text-V2-11B-Instruct-hf), [Foldseek](https://github.com/steineggerlab/foldseek), [AlphaFold DB](https://alphafold.ebi.ac.uk/), ESM2, Llama 3.1, and Hugging Face `transformers` and `peft`. The evaluation functions are adapted from the Prot2Text-V2 paper's code.
