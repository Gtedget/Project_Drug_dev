# SMILES Transformer Generator

This project trains a **GPT-style Transformer** on canonical SMILES strings
and generates **novel drug-like molecules**.

## Approach
- SMILES treated as a language
- Regex-based tokenizer (chemically aware)
- Decoder-only Transformer (GPT)
- RDKit-based validation & filtering

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Place a CSV with a `smiles` column in:
```
data/raw_smiles.csv
```

## Train
```bash
python src/train.py
```

## Generate Molecules
```bash
python src/generate.py
```

## Evaluate Molecules
```bash
python src/evaluate.py
```

## Output
Generated molecules are written to:
```
data/generated_smiles.csv
```

## Dependencies
- RDKit
- PyTorch
- HuggingFace Transformers
- tokenizers
- pandas
