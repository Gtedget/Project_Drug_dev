# Drug Generation From SMILES

This repository contains four molecule-generation workflows built on the same dataset and tokenizer:

1. A baseline GPT-style SMILES language model.
2. A conditional encoder-decoder baseline.
3. A standalone hybrid graph-conditioned generator that uses graph attention as an alternative to the plain transformer path.
4. A discrete denoising diffusion baseline for SMILES generation.

The project is organized so both approaches can be trained, generated from, and compared inside the same environment.

## What The Project Does

The codebase supports an end-to-end workflow for de novo molecule generation:

- load and clean SMILES data from CSV
- train or reuse a tokenizer
- train a baseline decoder-only transformer on SMILES strings
- train a standalone conditional graph model that uses target context, binding-site text, ligand graph features, and scalar molecular descriptors
- generate new molecules from either model
- filter, validate, and score generated molecules with RDKit
- compare the model families with shared metrics and timing

## Model Families

### 1. GPT SMILES Generator

Main files:

- `src/train.py`
- `src/generate.py`

This is the baseline language-model approach. It treats SMILES as a sequence modeling problem and trains a GPT-style decoder on canonicalized molecular strings.

Use this path when you want:

- the simplest baseline
- fast experimentation on unconditional generation
- a clean comparison point for newer conditional models

### 2. Standalone Hybrid GAT Generator

Main files:

- `src/conditional_hybrid_graph_generator.py`
- `src/train_gat_conditional_from_raw_smiles.py`
- `src/generate_gat_conditional_from_raw_smiles.py`

This is the stronger standalone graph path. It does not depend on the GPT training script and is meant to be a separate alternative.

It conditions generation on:

- target text, typically `target_name`
- optional binding-site text, typically `binding_site`
- normalized scalar conditioning such as `alogp`
- a vector of molecular descriptors such as HBA, HBD, PSA, molecular weight, QED, and ring counts
- a seed ligand graph derived from SMILES

Internally, this model combines:

- a target plus pocket text graph encoder
- a ligand graph encoder
- descriptor and logP encoders
- a transformer decoder over SMILES tokens

This is the path to use when you want a graph-based alternative that can exploit the richer metadata already present in `data/raw_smiles.csv`.

### 3. Conditional Encoder-Decoder Baseline

Main files:

- `src/conditional_generator.py`
- `src/train_conditional_from_raw_smiles.py`
- `src/generate_conditional_from_raw_smiles.py`

This model conditions a transformer decoder on target text and logP without the ligand-graph and pocket encoders used by the hybrid model.

### 4. Discrete Diffusion SMILES Generator

Main files:

- `src/diffusion_smiles_generator.py`
- `src/train_diffusion_from_raw_smiles.py`
- `src/generate_diffusion_from_raw_smiles.py`

This baseline uses a discrete denoising process over tokenized SMILES and iteratively refines a noisy sequence into a molecule candidate.

## Current Data Assumptions

The default dataset is:

- `data/raw_smiles.csv`

Minimum required column:

- `smiles`

Optional but useful columns already supported by the current scripts:

- `target_name`
- `binding_site`
- `alogp`
- `hba`
- `hbd`
- `psa`
- `rtb`
- `full_mwt`
- `aromatic_rings`
- `heavy_atoms`
- `qed_weighted`

Important limitation:

The current repository does not contain full protein sequences, protein structures, or 3D pocket coordinates. Because of that, the graph-conditioned model uses text-level target and pocket context rather than true residue-contact or atomistic protein-pocket graphs.

## Environment Setup

The project has been validated in the repository virtual environment on Windows.

### Recommended Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Important Dependency Notes

- The validated stack in this workspace is `numpy 2.4.x`, `pandas 3.0.x`, and `pyarrow 24.x`.
- Fresh Python processes in this Anaconda-backed `venv` rely on `src/runtime_bootstrap.py` to keep `torch`, `transformers`, `datasets`, and `huggingface_hub` from scanning broken base-install metadata.
- Use `rdkit` from `requirements.txt` for this environment.
- `accelerate>=1.1.0` is required for the Hugging Face `Trainer` used by the GPT workflow.
- The repo has been exercised in Python 3.12 inside the local `venv`.

## Repository Hygiene

This repository is intentionally kept lean so it can be linked cleanly into downstream projects such as a metabolite-prediction repository.

The checked-in repository should contain:

- source code in `src/`
- the main project documentation in `README.md`
- tokenizer assets in `tokenizer/`
- a small set of canonical input datasets in `data/`

Generated experiment outputs, local checkpoints, smoke-test files, logs, snapshots, cached Python files, and the local `venv/` are ignored via `.gitignore` and should remain local-only.

Exception:

- `model/gpt_activity_long10/`
- `model/conditional_gat_activity_long10.pt`

When present, these two 10-epoch checkpoint paths are kept as the reference GPT and GAT models.

## Repository Layout

Key files and directories:

- `data/raw_smiles.csv`: base molecule dataset
- `data/target_activity_training.csv`: target-aware activity dataset acquired from ChEMBL and UniProt
- `data/target_metadata_enriched.csv`: enriched target metadata table
- `data/target_activity_training_curated.csv`: curated training dataset used for target-aware experiments
- `data/target_activity_training_curation_report.csv`: curation summary for the curated target-aware dataset
- `model/`: local model outputs; the retention policy reserves `model/gpt_activity_long10/` and `model/conditional_gat_activity_long10.pt` as kept reference checkpoints when they exist
- `tokenizer/`: tokenizer artifacts used by both workflows
- `src/train.py`: GPT training entry point
- `src/generate.py`: GPT generation entry point
- `src/train_conditional_from_raw_smiles.py`: conditional encoder-decoder training entry point
- `src/generate_conditional_from_raw_smiles.py`: conditional encoder-decoder generation entry point
- `src/train_gat_conditional_from_raw_smiles.py`: hybrid GAT training entry point
- `src/generate_gat_conditional_from_raw_smiles.py`: hybrid GAT generation entry point
- `src/train_diffusion_from_raw_smiles.py`: diffusion training entry point
- `src/generate_diffusion_from_raw_smiles.py`: diffusion generation entry point
- `src/run_drug_generator.py`: unified launcher for both workflows
- `src/comparison_metrics.py`: shared evaluation metrics
- `src/compare_models.py`: matched multi-model comparison runner
- `src/run_vollab_gpu_pipeline.py`: install, data-prep, long-run training, generation, and evaluation pipeline for all four models on a GPU machine
- `COLAB_GPU_README.md`: step-by-step Google Colab GPU instructions for running each of the four models

## Long GPU Run

For a longer end-to-end GPU run across GPT, conditional, GAT, and diffusion, use the dedicated pipeline script:

```powershell
venv\Scripts\python.exe src/run_vollab_gpu_pipeline.py `
  --install `
  --torch_index_url https://download.pytorch.org/whl/cu124 `
  --run_name vollab_long `
  --epochs 20 `
  --max_length 96 `
  --num_return_sequences 200
```

What it does:

- installs Python dependencies, and optionally upgrades `torch` from a GPU wheel index
- builds `data/target_activity_training.csv` and `data/target_metadata_enriched.csv` if they are missing
- curates `data/target_activity_training_curated.csv` with a SELFIES column if it is missing
- trains `tokenizer/selfies_tokenizer.json` if it is missing
- resumes from existing checkpoints and generated CSVs by default
- trains only the missing models with longer-run defaults
- generates only the missing molecule CSVs for all four models
- scores the generated outputs and writes a comparison table plus a run manifest

To force a full rerun even when outputs already exist, add:

- `--force_train_existing`
- `--force_generate_existing`

Default long-run outputs are written to:

- `model/gpt_<run_name>/`
- `model/conditional_<run_name>.pt`
- `model/conditional_gat_<run_name>.pt`
- `model/diffusion_<run_name>.pt`
- `data/generated_*_<run_name>.csv`
- `data/model_comparison_<run_name>.csv`
- `data/run_manifest_<run_name>.json`

## Typical Workflow

### 1. Prepare Data

If you need to clean or preprocess raw SMILES, the repo includes helper scripts such as:

- `src/preprocess_smiles.py`
- `src/curate_training_data.py`
- `src/convert_smiles_to_selfies.py`
- `src/tokenizer_train.py`

If you are using the included ChEMBL-style CSV, you can usually start directly from `data/raw_smiles.csv`.

For a training-ready dataset with salt stripping, parent-fragment standardization, length filtering, and reduced target ambiguity, run:

```powershell
venv\Scripts\python.exe src/curate_training_data.py `
  --input_csv data/raw_smiles.csv `
  --output_csv data/curated_training_smiles.csv `
  --report_csv data/curation_report.csv `
  --max_smiles_length 128 `
  --max_targets_per_smiles 2
```

This produces:

- `data/curated_training_smiles.csv`: compact curated training file for the model scripts
- `data/curation_report.csv`: summary of rows removed by standardization, deduplication, length filtering, and target-ambiguity filtering

### 1b. Acquire Target Activity And Protein Annotation Data

To move beyond plain molecule generation and toward target-aware training, the repo now includes a data-acquisition script that:

- reads the target IDs already present in an input CSV
- pulls ChEMBL activity records for those targets
- enriches each target with UniProt sequence metadata and site annotations
- optionally adds a SELFIES column for the acquired molecules

Example:

```powershell
venv\Scripts\python.exe src/build_target_activity_dataset.py `
  --input_csv data/raw_smiles.csv `
  --output_csv data/target_activity_training.csv `
  --target_metadata_csv data/target_metadata_enriched.csv `
  --max_activities_per_target 100 `
  --selfies_column selfies
```

This produces:

- `data/target_activity_training.csv`: activity-supervised molecule dataset with assay metadata, protein sequence fields, and optional SELFIES
- `data/target_metadata_enriched.csv`: one-row-per-target metadata table with UniProt accession, sequence length, site counts, and PDB cross-references

This is the recommended starting point if you want to train a more app-ready model that learns target-specific binding behavior instead of only molecule syntax.

### 2. Train The GPT Baseline

```powershell
venv\Scripts\python.exe src/train.py `
  --data_csv data/raw_smiles.csv `
  --smiles_column smiles `
  --output_dir model/gpt_smiles `
  --text_path data/smiles.txt `
  --max_length 128 `
  --batch_size 64 `
  --num_train_epochs 20
```

Key configurable options:

- `--data_csv`
- `--smiles_column`
- `--output_dir`
- `--text_path`
- `--max_length`
- `--batch_size`
- `--num_train_epochs`
- `--learning_rate`
- `--n_embd`, `--n_layer`, `--n_head`

### 3. Generate With The GPT Baseline

```powershell
venv\Scripts\python.exe src/generate.py `
  --model_dir model/gpt_smiles `
  --max_length 128 `
  --temperature 0.8 `
  --top_k 50 `
  --top_p 0.95 `
  --num_return_sequences 100 `
  --output_csv data/generated_smiles.csv
```

### 4. Train The Hybrid GAT Model

```powershell
venv\Scripts\python.exe src/train_gat_conditional_from_raw_smiles.py `
  --data_csv data/raw_smiles.csv `
  --smiles_column smiles `
  --protein_column target_name `
  --binding_site_column binding_site `
  --logp_column alogp `
  --protein_encoder_type gat `
  --batch_size 32 `
  --num_epochs 5 `
  --max_smiles_length 128 `
  --output_path model/conditional_hybrid.pt
```

What gets saved in the checkpoint:

- model weights
- model configuration
- data-column configuration
- logP normalization statistics
- descriptor names
- descriptor mean and standard deviation for generation-time normalization

### 5. Generate With The Hybrid GAT Model

```powershell
venv\Scripts\python.exe src/generate_gat_conditional_from_raw_smiles.py `
  --checkpoint_path model/conditional_hybrid.pt `
  --data_csv data/raw_smiles.csv `
  --target_name "Macrophage colony-stimulating factor 1 receptor" `
  --binding_site "Beta tubulin paclitaxel binding site" `
  --alogp 3.5 `
  --seed_smiles "CCOc1ccc(NC(=O)Nc2ccccc2)cc1" `
  --num_return_sequences 100 `
  --max_length 128 `
  --temperature 0.8 `
  --output_csv data/generated_hybrid_smiles.csv
```

Generation output includes:

- generated `smiles`
- conditioning target `protein`
- `binding_site`
- `seed_smiles`
- `alogp`
- exported descriptor-conditioning columns such as `condition_hba`, `condition_psa`, and related fields used for downstream comparison

## Unified Launcher

If you prefer one command surface for both models, use:

- `src/run_drug_generator.py`

Examples:

```powershell
venv\Scripts\python.exe src/run_drug_generator.py train --model gpt --data_csv data/raw_smiles.csv
venv\Scripts\python.exe src/run_drug_generator.py generate --model gpt --model_dir model/gpt_smiles
venv\Scripts\python.exe src/run_drug_generator.py train --model gat --data_csv data/raw_smiles.csv --num_epochs 3
venv\Scripts\python.exe src/run_drug_generator.py generate --model gat --checkpoint_path model/conditional_hybrid.pt --num_return_sequences 50
```

## Comparing GPT, Conditional, Hybrid GAT, And Diffusion

Use the shared comparison harness:

```powershell
venv\Scripts\python.exe src/compare_models.py `
  --data_csv data/raw_smiles.csv `
  --epochs 1 `
  --batch_size 32 `
  --max_length 48 `
  --num_return_sequences 50 `
  --results_csv data/model_comparison_all.csv
```

The comparison runner trains all four models with matched settings, runs generation, and writes a side-by-side CSV report.

Current comparison metrics include:

- `valid_count`
- `validity_rate`
- `unique_count`
- `uniqueness_rate`
- `novel_count`
- `novelty_rate`
- `mean_qed`
- `mean_logp`
- `mean_mol_wt`
- `train_time_seconds`
- `inference_time_seconds`
- `requested_samples_per_second`
- `valid_samples_per_second`
- `logp_target_mae`
- `descriptor_target_mae`
- `descriptor_target_coverage`

## Evaluation And Postprocessing

Basic evaluation scripts are included:

```powershell
venv\Scripts\python.exe src/evaluate.py
venv\Scripts\python.exe src/evaluate_metrics.py
```

These scripts provide simple RDKit-based filtering and summary metrics for generated molecules. The comparison harness is the better option when you want model-vs-model benchmarking.

## Common Output Paths

- `model/gpt_smiles/`: trained GPT model and tokenizer
- `model/conditional_hybrid.pt`: saved hybrid GAT checkpoint
- `data/generated_smiles.csv`: GPT-generated molecules
- `data/generated_hybrid_smiles.csv`: hybrid GAT-generated molecules
- `data/generated_smiles_valid.csv`: valid GPT outputs after filtering
- `data/filtered_smiles.csv`: filtered molecule table from evaluation scripts
- `data/model_comparison_hybrid.csv`: side-by-side GPT vs hybrid-GAT comparison report

## Known Limitations

- The hybrid graph model is stronger than the earlier standalone GAT baseline, but generation quality still depends heavily on training time and data quality.
- The dataset has limited binding-site coverage, so pocket conditioning is sparse for some targets.
- Because the repo lacks true protein structure inputs, pocket graphs are built from text tokens rather than physical residue-contact graphs.
- On CPU, the hybrid GAT path is substantially slower than the GPT baseline, especially during full-dataset training.

## Additional Documentation

For older or supplementary notes, see:

- `COMPREHENSIVE_README.md`
- `CONDITIONAL_MODELS_README.md`

The main README should be treated as the current operational guide for the repository.
