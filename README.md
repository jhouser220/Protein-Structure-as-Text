Prot2Text-V2 + Foldseek Retrieval

Fine-tunes Prot2Text-V2 with LoRA to predict protein function descriptions, augmenting each prompt with the function text of structurally similar training proteins retrieved by Foldseek.

The project has two notebooks and a small package that holds all the logic:

Notebook	Purpose
train.ipynb	Download structures, run Foldseek retrieval, train, generate, and score
analysis.ipynb	Load the training outputs and interpret predictions per protein and per identity bin
How it works
Structures. AlphaFold PDB files are downloaded for every protein in the train, eval and test splits.
Retrieval. A Foldseek index is built from the training structures. Every split is searched against it, and each query keeps its top hits (self-hits are removed).
Downsampling. Training hits with sequence identity above a threshold ("easy" hits) are mostly dropped, so the model cannot learn to copy a near-identical neighbour. Eval and test hits are left untouched.
Prompting. Each protein gets a Llama chat prompt of the form Taxon: ... ; Similar structures: ... ; Sequence embeddings: <placeholders>. The ESM2 encoder's output replaces the placeholders.
Training. LoRA is applied to the Llama decoder's attention and MLP projections. The modality adapter stays frozen by default.
Generation. Test-set predictions are produced under three conditions:
with: taxonomy and retrieved structures in the prompt.
without: taxonomy only.
seq_only: neither (taxonomy is always "unknown").
Evaluation. Exact match, BLEU-2/4, ROUGE and BERTScore (RoBERTa-large and BioBERT-large), overall and stratified by best-hit identity.

Requirements
Google Colab (or similar) with a GPU that fits the 11B model in bfloat16 with LoRA and gradient checkpointing.
Google Drive for persistent storage.
A clone of the Prot2Text-V2 code repository at Prot2Text-V2-main. Only its models package is used (Esm2LlamaInstructForCausalLM, ModalityAdapter, ModalityAdapterConfig).
Access to meta-llama/Llama-3.1-8B-Instruct on Hugging Face (the tokenizer is loaded from there).
The Prot2Text dataset CSVs (train.csv, eval.csv, test.csv) with columns AlphaFoldDB, Full Name, taxon, sequence, function.
Python packages: see requirements.txt. The Foldseek binary is downloaded by the first cells of train.ipynb.
Setup

Arrange Drive like this (the root and subfolder names are defined in config.py and can be changed there):

MyDrive/Colab_Notebooks/266_Final_Project/
├── src/                      # copy this repo's src/ and requirements.txt here
├── Prot2Text-V2-main/        # upstream repo clone
└── Prot2TextDataset/         # train.csv, eval.csv, test.csv

If your root differs, change DRIVE_ROOT in config.py and the sys.path.insert(...) line in each notebook.

Usage
1. Train and generate

Open train.ipynb in Colab and run the cells top to bottom. The notebook only configures and calls the package.

First run: downloads structures, builds the Foldseek index, searches all splits, and backs the index and hit tables up to foldseek_index/ on Drive.
Later runs: if that backup exists it is restored instead, and structures are not re-downloaded.

To change settings, edit cfg in the notebook before the later cells, for example:

python
cfg = Config()
cfg.train.num_epochs = 2
cfg.lora.rank = 32
cfg.data.trim_train = 20_000   # smaller run for debugging
cfg.paths.run_name = "my_experiment"
2. Analyze

Open analysis.ipynb and run it after training has finished. It also needs the external baseline files (Full_run/test_baseline_predictions.pkl and the baseline metrics JSON); their paths are set in the second code cell.

Configuration

All tunable values live in src/prot2text_foldseek/config.py:

Group	Controls
Paths	Drive root, scratch directory, run name, and derived paths
DownloadConfig	Concurrency, retries, timeouts, AlphaFold URL templates
FoldseekConfig	Sensitivity, max hits, e-value, threads, output format
RetrievalConfig	Hits kept, easy-hit identity threshold and keep probability, hits per prompt, description truncation
DataConfig	Taxonomy and retrieval dropout, max sequence and description length, split trimming
LoraSettings	Rank, alpha, dropout, which modules are targeted, frozen adapter, gradient checkpointing
TrainConfig	Epochs, batch size, gradient accumulation, clipping, learning rate
GenerationConfig	Batch size, max new tokens, beam search/sampling, special-token ids
MetricsConfig	BERTScore models and truncation length
Outputs

Written to foldseek_lora/<run_name>/ on Drive:

File	Contents
adapter_checkpoint_<epoch>/	LoRA adapter weights
optimizer_scheduler_checkpoint_<epoch>.pt	Optimizer state
history.json	Per-epoch train and eval loss and elapsed time
generation_test_retrieval.json, generation_test_noretrieval.json, generation_seq_only.json	{accession: {"true", "pred"}} for each condition
metrics.json, metrics_seq_only.json	Aggregate metrics
stratified_metrics.csv	With vs. without retrieval, per identity bin
all_predictions_analysis.csv	Per-protein table written by the analysis notebook

prompt_hits.json (the downsampled hits actually used in prompts) is saved to foldseek_work/ on Drive.

Notes and caveats
Eval loss is computed without taxonomy/retrieval dropout and without random cropping, so it is deterministic.
Training steps: a final partial gradient-accumulation group is stepped rather than discarded.
Foldseek IDs: the pipeline assumes Foldseek query and target IDs match the accession keys in the CSVs. Check a few hits after the first search if you change the PDB file naming.
Copying analysis: the copy_rate section of the analysis notebook (ROUGE-L between each prediction and its best retrieved description) is a reconstruction of an earlier cell that referenced undefined variables. Confirm it measures what you intend.
Not covered: the original upstream scripts (train_instruct.py, train_contrast.py) support multi-GPU DDP, a learning-rate scheduler, and checkpoint resume. This project is single-GPU and has no scheduler or resume option.
Acknowledgements

Built on Prot2Text-V2, Foldseek, AlphaFold DB, ESM2, Llama 3.1, and Hugging Face transformers and peft. The evaluation functions are adapted from the Prot2Text-V2 paper's code.
