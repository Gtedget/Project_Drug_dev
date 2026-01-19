# SMILES Transformer Drug Generator

This project trains a GPT-style Transformer model to generate novel drug-like molecules using canonical SMILES strings. It includes data preprocessing, chemically-aware tokenization, model training, molecule generation, postprocessing, and evaluation.

## Features
- Data cleaning and filtering for valid, unique SMILES
- Option to use SELFIES for robust molecular representation
- Custom tokenizer training
- Transformer-based sequence modeling (GPT architecture)
- Generation and validation of new molecules
- Drug-likeness filtering (QED, molecular weight, etc.)
- Comprehensive evaluation metrics and visualization

## Workflow

### 1. Data Preparation
- Place your raw SMILES data in `data/raw_smiles.csv` (must have a `smiles` column).
- Run preprocessing to filter valid, unique SMILES:
  ```bash
  python src/preprocess_smiles.py
  ```
- (Optional) Convert SMILES to SELFIES for guaranteed validity:
  ```bash
  python src/convert_smiles_to_selfies.py
  ```

### 2. Tokenizer Training
- Train a custom tokenizer on filtered SMILES or SELFIES:
  ```bash
  python src/tokenizer_train.py
  ```

### 3. Model Training
- Train the GPT-style Transformer model:
  ```bash
  python src/train.py
  ```

### 4. Molecule Generation
- Generate new molecules using the trained model:
  ```bash
  python src/generate.py
  ```

### 5. Evaluation & Postprocessing
- Filter generated molecules for validity and drug-likeness:
  ```bash
  python src/evaluate.py
  python src/postprocess_generated_smiles.py
  ```
- Save valid molecules to `data/generated_smiles_valid.csv` and drug-like molecules to `data/generated_smiles_filtered.csv`.

### 6. Metrics & Visualization
- Evaluate validity, uniqueness, novelty, and visualize sample molecules:
  ```bash
  python src/evaluate_metrics.py
  ```
- Sample molecule grid saved to `data/sample_molecules.png`.

## Dependencies
- RDKit
- PyTorch (installed via conda)
- HuggingFace Transformers
- tokenizers
- pandas
- selfies (optional, for SELFIES support)

## Advanced Features
- Curriculum learning and early stopping for robust training
- Mixed precision training for computational efficiency
- Transfer learning from chemical language models
- Rule-based filtering for drug-likeness

## Output Files
- `data/filtered_smiles.csv`: Valid, unique SMILES for training
- `data/selfies.csv`: SELFIES representation (optional)
- `tokenizer/smiles_tokenizer.json`: Trained tokenizer
- `model/gpt_smiles/`: Trained model and tokenizer
- `data/generated_smiles.csv`: Raw generated molecules
- `data/generated_smiles_valid.csv`: Valid generated molecules
- `data/generated_smiles_filtered.csv`: Drug-like generated molecules
- `data/sample_molecules.png`: Visualization of sample molecules

## Getting Started
1. Set up a conda environment (Python 3.11 recommended)
2. Install dependencies:
   ```bash
   conda install -y -c conda-forge rdkit
   conda install -y pytorch cpuonly -c pytorch
   pip install transformers tokenizers pandas selfies
   ```
3. Follow the workflow steps above

## License
MIT

## Contact
For questions or contributions, open an issue or pull request on GitHub.
