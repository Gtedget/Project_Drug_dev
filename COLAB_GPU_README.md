# Run On Colab GPU

This guide shows how to run the repository on Google Colab with a GPU and execute the four model families one by one:

- GPT
- conditional encoder-decoder
- hybrid GAT
- diffusion

The commands below are written as Colab notebook cells. Run them in order.

## 1. Start A GPU Runtime

In Colab, open:

- `Runtime > Change runtime type > T4 GPU` or another CUDA GPU

Then verify the GPU:

```bash
!nvidia-smi
```

## 2. Get The Project Into Colab

Choose one of these approaches.

### Option A: Clone From Git

```bash
%cd /content
!git clone <YOUR_REPOSITORY_URL> Project_drug_fromsmiles
%cd /content/Project_drug_fromsmiles
```

### Option B: Copy From Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!cp -r /content/drive/MyDrive/Project_drug_fromsmiles ./Project_drug_fromsmiles
%cd /content/Project_drug_fromsmiles
```

## 3. Install Dependencies

Colab usually already has a GPU-enabled PyTorch, so the safest path is:

1. install the non-torch requirements
2. verify CUDA is available
3. only reinstall `torch` if CUDA is missing

```bash
!python -m pip install --upgrade pip
!grep -v '^torch$' requirements.txt > requirements_colab.txt
!python -m pip install -r requirements_colab.txt
```

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` is `False`, install a CUDA build and restart the runtime once:

```bash
!python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

## 4. Verify The Repo Layout

```bash
!ls
!ls data
!ls src
!ls tokenizer
```

## 5. Prepare Input Data And Tokenizer

If your copy of the repo already includes these files, you can skip to section 6:

- `data/target_activity_training.csv`
- `data/target_metadata_enriched.csv`
- `data/target_activity_training_curated.csv`
- `tokenizer/selfies_tokenizer.json`

### 5a. Build The Target-Aware Activity Dataset

Run this only if `data/target_activity_training.csv` is missing.

```bash
!python src/build_target_activity_dataset.py \
  --input_csv data/raw_smiles.csv \
  --output_csv data/target_activity_training.csv \
  --target_metadata_csv data/target_metadata_enriched.csv \
  --target_column target_chembl_id \
  --target_name_column target_name \
  --max_activities_per_target 100 \
  --selfies_column selfies
```

### 5b. Curate The Training Dataset

```bash
!python src/curate_training_data.py \
  --input_csv data/target_activity_training.csv \
  --output_csv data/target_activity_training_curated.csv \
  --report_csv data/target_activity_training_curation_report.csv \
  --smiles_column smiles \
  --target_column target_chembl_id \
  --max_smiles_length 96 \
  --max_targets_per_smiles 2 \
  --selfies_column selfies
```

### 5c. Train The SELFIES Tokenizer

```bash
!python src/tokenizer_train.py \
  --data_csv data/target_activity_training_curated.csv \
  --sequence_column selfies \
  --source_smiles_column smiles \
  --representation selfies \
  --text_path data/selfies_activity_colab.txt \
  --tokenizer_output tokenizer/selfies_tokenizer.json \
  --vocab_size 512
```

## 6. Pick A Conditioning Example

The conditional, GAT, and diffusion models need a target context for generation. This cell prints a usable example from the curated dataset.

```python
import pandas as pd

df = pd.read_csv("data/target_activity_training_curated.csv")
example_row = df[["target_chembl_id", "smiles"]].dropna().iloc[0]
print(example_row.to_dict())
```

Example values from the current curated dataset:

- `target_chembl_id = CHEMBL1844`
- `seed_smiles = COc1ccc2ncnc(N(C)c3ccccc3)c2c1`

Replace these if you want to target a different row.

## 7. Common Evaluation Cell

Use this helper after each generation run.

```python
import sys
sys.path.append("src")

from comparison_metrics import evaluate_generated_file, load_train_smiles

train_smiles = load_train_smiles(
    "data/target_activity_training_curated.csv",
    smiles_column="smiles",
)

def show_metrics(path, requested_samples=200):
    metrics = evaluate_generated_file(
        generated_csv=path,
        requested_samples=requested_samples,
        train_smiles=train_smiles,
        smiles_column="smiles",
    )
    print(metrics)
```

## 8. GPT Model

### 8a. Train GPT

```bash
!python src/train.py \
  --data_csv data/target_activity_training_curated.csv \
  --smiles_column smiles \
  --sequence_column selfies \
  --text_path data/selfies_activity_gpt_colab.txt \
  --output_dir model/gpt_colab_long \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json \
  --max_length 96 \
  --batch_size 32 \
  --num_train_epochs 20 \
  --learning_rate 3e-4 \
  --save_steps 500 \
  --logging_steps 50 \
  --n_embd 256 \
  --n_layer 6 \
  --n_head 8
```

### 8b. Generate With GPT

```bash
!python src/generate.py \
  --model_dir model/gpt_colab_long \
  --representation selfies \
  --max_length 96 \
  --temperature 0.8 \
  --num_return_sequences 200 \
  --output_csv data/generated_gpt_colab_long.csv
```

### 8c. Evaluate GPT

```python
show_metrics("data/generated_gpt_colab_long.csv", requested_samples=200)
```

## 9. Conditional Encoder-Decoder Model

### 9a. Train Conditional Encoder-Decoder

```bash
!python src/train_conditional_from_raw_smiles.py \
  --data_csv data/target_activity_training_curated.csv \
  --smiles_column selfies \
  --source_smiles_column smiles \
  --protein_column target_chembl_id \
  --output_path model/conditional_colab_long.pt \
  --max_length 96 \
  --batch_size 32 \
  --num_epochs 20 \
  --learning_rate 3e-4 \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json \
  --condition_mode target_lookup
```

### 9b. Generate With Conditional Encoder-Decoder

Use `target_chembl_id` as the generation condition.

```bash
!python src/generate_conditional_from_raw_smiles.py \
  --checkpoint_path model/conditional_colab_long.pt \
  --data_csv data/target_activity_training_curated.csv \
  --target_name CHEMBL1844 \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json \
  --num_return_sequences 200 \
  --max_length 96 \
  --temperature 0.8 \
  --output_csv data/generated_conditional_colab_long.csv
```

### 9c. Evaluate Conditional Encoder-Decoder

```python
show_metrics("data/generated_conditional_colab_long.csv", requested_samples=200)
```

## 10. Hybrid GAT Model

### 10a. Train Hybrid GAT

```bash
!python src/train_gat_conditional_from_raw_smiles.py \
  --data_csv data/target_activity_training_curated.csv \
  --smiles_column selfies \
  --structure_smiles_column smiles \
  --protein_column target_chembl_id \
  --protein_encoder_type gat \
  --batch_size 16 \
  --lr 3e-4 \
  --num_epochs 20 \
  --max_smiles_length 96 \
  --output_path model/conditional_gat_colab_long.pt \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json
```

### 10b. Generate With Hybrid GAT

```bash
!python src/generate_gat_conditional_from_raw_smiles.py \
  --checkpoint_path model/conditional_gat_colab_long.pt \
  --data_csv data/target_activity_training_curated.csv \
  --protein_value CHEMBL1844 \
  --seed_smiles COc1ccc2ncnc(N(C)c3ccccc3)c2c1 \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json \
  --num_return_sequences 200 \
  --max_length 96 \
  --temperature 0.8 \
  --output_csv data/generated_gat_colab_long.csv
```

### 10c. Evaluate Hybrid GAT

```python
show_metrics("data/generated_gat_colab_long.csv", requested_samples=200)
```

## 11. Diffusion Model

### 11a. Train Diffusion

```bash
!python src/train_diffusion_from_raw_smiles.py \
  --data_csv data/target_activity_training_curated.csv \
  --smiles_column selfies \
  --source_smiles_column smiles \
  --protein_column target_chembl_id \
  --output_path model/diffusion_colab_long.pt \
  --max_length 96 \
  --batch_size 32 \
  --num_epochs 20 \
  --learning_rate 3e-4 \
  --num_diffusion_steps 32 \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json
```

### 11b. Generate With Diffusion

```bash
!python src/generate_diffusion_from_raw_smiles.py \
  --checkpoint_path model/diffusion_colab_long.pt \
  --data_csv data/target_activity_training_curated.csv \
  --target_name CHEMBL1844 \
  --representation selfies \
  --tokenizer_path tokenizer/selfies_tokenizer.json \
  --num_return_sequences 200 \
  --max_length 96 \
  --temperature 0.8 \
  --output_csv data/generated_diffusion_colab_long.csv
```

### 11c. Evaluate Diffusion

```python
show_metrics("data/generated_diffusion_colab_long.csv", requested_samples=200)
```

## 12. Compare All Four Models Side By Side

```python
show_metrics("data/generated_gpt_colab_long.csv", requested_samples=200)
show_metrics("data/generated_conditional_colab_long.csv", requested_samples=200)
show_metrics("data/generated_gat_colab_long.csv", requested_samples=200)
show_metrics("data/generated_diffusion_colab_long.csv", requested_samples=200)
```

## 13. Optional: Run Everything With One Command

If you want the automated runner instead of manual per-model cells, use:

```bash
!python src/run_vollab_gpu_pipeline.py \
  --run_name colab_long \
  --epochs 20 \
  --max_length 96 \
  --num_return_sequences 200
```

By default, the pipeline resumes from existing checkpoints and generated CSVs. To force a full rerun:

```bash
!python src/run_vollab_gpu_pipeline.py \
  --run_name colab_long \
  --epochs 20 \
  --max_length 96 \
  --num_return_sequences 200 \
  --force_train_existing \
  --force_generate_existing
```

## 14. Save Outputs Back To Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/Project_drug_fromsmiles_colab_outputs
!cp -r model /content/drive/MyDrive/Project_drug_fromsmiles_colab_outputs/
!cp -r data /content/drive/MyDrive/Project_drug_fromsmiles_colab_outputs/
```

## 15. Expected Output Files

After a full run, the main outputs are:

- `model/gpt_colab_long/`
- `model/conditional_colab_long.pt`
- `model/conditional_gat_colab_long.pt`
- `model/diffusion_colab_long.pt`
- `data/generated_gpt_colab_long.csv`
- `data/generated_conditional_colab_long.csv`
- `data/generated_gat_colab_long.csv`
- `data/generated_diffusion_colab_long.csv`

If Colab disconnects during long training, save outputs to Drive periodically or rerun with the automated pipeline and its resume behavior.