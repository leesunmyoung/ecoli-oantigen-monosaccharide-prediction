# ecoli-oantigen-monosaccharide-prediction
# E. coli O-Antigen Monosaccharide Composition Prediction

**Genomic prediction of *E. coli* O-antigen monosaccharide composition using Random Forest regression**

---

## Overview

This repository contains the data and analysis scripts for the study:

> **"Prediction of *E. coli* O-antigen monosaccharide composition from genomic features using machine learning"**  
> *[Author names] — BMC Microbiology (submitted)*

### Key Finding
Non-CAZyme biosynthetic pathway genes (nucleotide sugar biosynthesis, product name tokens, enzyme keyword annotations) contribute more to O-antigen monosaccharide prediction than CAZyme (glycosyltransferase/glycoside hydrolase family) annotations alone.

---

## Repository Structure

```
ecoli-oantigen-prediction/
│
├── README.md
│
├── data/
│   ├── README_data.md                              # Data description
│   ├── supervised_dataset.tsv                      # X (CAZyme features) + Y (monosaccharides), n=203
│   ├── oantigen_locus_feature_table_mlX_binTok.tsv # X (non-CAZyme + CAZyme features), n=160
│   └── Y_monosaccharide.tsv                        # Y labels only, n=216
│
├── scripts/
│   ├── data_preprocessing.py    # Data loading, deduplication, X/Y split
│   ├── feature_importance.py    # MDI feature importance (CAZyme vs non-CAZyme)
│   ├── model_comparison.py      # Full / CAZyme-only / non-CAZyme-only model comparison
│   └── pathway_analysis.py      # Scatter plot + Mann-Whitney pathway group analysis
│
└── figures/
    ├── Fig1_feature_importance.png
    ├── Fig2_model_comparison.png
    └── Fig3_pathway_analysis.png
```

---

## Data

| File | Samples | Description |
|------|---------|-------------|
| `supervised_dataset.tsv` | 203 | X (106 CAZyme features) + Y (82 monosaccharides) |
| `oantigen_locus_feature_table_mlX_binTok.tsv` | 160 | X with non-CAZyme features (kw_*, prod_tok_*) |
| `Y_monosaccharide.tsv` | 216 | Y labels: monosaccharide counts per O-antigen subtype |

**Note:** Of 216 O-antigen subtypes, 160 had complete genomic feature data and were used for machine learning analysis. The 43 missing subtypes were excluded due to incomplete feature extraction from genome sequences.

---

## Requirements

This project involves two independent stages with separate environments.

### Stage 1 — Genomic feature extraction (upstream pipeline, not included here)
CAZyme annotation was performed using dbCAN in a dedicated conda environment.
This stage produced the TSV feature files included in `data/`.

```bash
conda activate dbcan   # dbCAN >= 4.0 required
# run_dbcan <genome.fasta> ...
```

The output files (`oantigen_locus_feature_table_mlX_binTok.tsv`, etc.) are
provided in `data/` and do **not** require re-running dbCAN to reproduce the
ML analysis.

### Stage 2 — Machine learning analysis (this repository)
All scripts in `scripts/` run in a standard Python environment,
independent of the dbCAN conda environment.

```bash
pip install -r requirements.txt
```

```
python >= 3.9
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.10
```

---

## Usage

Run scripts in order:

```bash
# Step 1: Data preprocessing
python scripts/01_data_preprocessing.py

# Step 2: Feature importance analysis
python scripts/02_feature_importance.py

# Step 3: Model comparison (Full / CAZyme-only / non-CAZyme-only)
python scripts/03_model_comparison.py

# Step 4: Pathway group analysis
python scripts/04_pathway_analysis.py
```

All output figures are saved to the `figures/` directory.

---

## Methods Summary

- **Model:** Random Forest Regressor (`scikit-learn`) with `MultiOutputRegressor`
- **Parameters:** `n_estimators=200`, `min_samples_leaf=2`, `random_state=42`
- **Evaluation:** 5-fold cross-validation, R² and MAE
- **Feature importance:** Mean Decrease Impurity (MDI), averaged across all output estimators
- **Feature groups:**
  - *CAZyme features (n=19):* GT/GH family counts from dbCAN annotation
  - *non-CAZyme features (n=62):* enzyme keyword annotations (`kw_*`), product name tokens (`prod_tok_*`), structural gene presence (`has_wzx`, `has_wzz`), locus gene count

---

## Results Summary

| Metric | Full model | CAZyme-only | non-CAZyme-only |
|--------|-----------|-------------|-----------------|
| Mean R² (5-fold CV) | 0.087 | -0.060 | 0.052 |
| Feature importance share | — | 27.2% | 72.8% |

Best predicted monosaccharide: **Rha** (R² = 0.476), supported by dTDP-rhamnose pathway genes (`rmlA-D`) captured in non-CAZyme features.

---

## AI Tools Disclosure

ChatGPT (OpenAI) and Claude (Anthropic) were used to assist in genomic feature extraction, machine learning pipeline development, statistical analysis, and figure generation. All analyses were verified and interpreted by the authors, who take full responsibility for the accuracy and integrity of the reported results.

---

## Citation

If you use this code or data, please cite:

> [Author names] (2026). E. coli O-antigen monosaccharide composition prediction — analysis code (v1.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

This project is licensed under the MIT License.
