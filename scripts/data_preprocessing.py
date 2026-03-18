"""
data_preprocessing.py
========================
Data loading, deduplication, and X/Y separation for
E. coli O-antigen monosaccharide composition prediction.

Output:
    data/X_processed.tsv   -- X features after zero-variance removal
    data/Y_common.tsv      -- Y labels filtered to >= 5 samples
"""

import pandas as pd
import numpy as np

# ── File paths ────────────────────────────────────────────────────────
X_PATH = "data/oantigen_locus_feature_table_mlX_binTok.tsv"  # X features (160 samples)
Y_PATH = "data/Y_monosaccharide.tsv"                          # Y labels  (216 samples)

# ── 1. Load data ──────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data")
print("=" * 60)

X_raw = pd.read_csv(X_PATH, sep='\t', index_col=0)
Y_raw = pd.read_csv(Y_PATH, sep='\t', index_col=0)

print(f"X_raw shape: {X_raw.shape}  (samples x features)")
print(f"Y_raw shape: {Y_raw.shape}  (samples x monosaccharides)")

# ── 2. Handle duplicate Y indices ────────────────────────────────────
# Some O-antigen subtypes appear in multiple rows; merge by mean
dup_idx = Y_raw.index[Y_raw.index.duplicated(keep=False)].unique()
if len(dup_idx) > 0:
    print(f"\nDuplicate indices found ({len(dup_idx)}): {list(dup_idx)}")
    print("  Merging duplicates by mean values")
    Y_raw = Y_raw.groupby(Y_raw.index).mean()
    print(f"  Y after deduplication: {Y_raw.shape}")

# ── 3. Extract common samples (X intersection Y) ──────────────────────
common_idx = X_raw.index.intersection(Y_raw.index)
print(f"\nCommon samples (X and Y): {len(common_idx)}")
print(f"Y-only samples excluded (no genomic features available): "
      f"{len(set(Y_raw.index) - set(X_raw.index))}")

X = X_raw.loc[common_idx].sort_index()
Y = Y_raw.loc[common_idx].sort_index()

assert list(X.index) == list(Y.index), "Index mismatch after alignment!"
print("Index alignment confirmed")

# ── 4. Remove zero-variance features ─────────────────────────────────
# Features with no variation across samples provide no predictive signal
zero_var = [c for c in X.columns if X[c].std() == 0]
X = X.drop(columns=zero_var)
print(f"\nZero-variance features removed ({len(zero_var)}): {zero_var}")
print(f"X after filtering: {X.shape}")

# ── 5. Filter Y: keep monosaccharides with >= 5 positive samples ──────
# Monosaccharides present in fewer than 5 samples are too sparse for
# reliable regression and are excluded from model training.
y_nonzero = (Y > 0).sum()
Y_common  = Y.loc[:, y_nonzero >= 5]
Y_rare    = Y.loc[:, (y_nonzero > 0) & (y_nonzero < 5)]

print(f"\nMonosaccharides used for ML (>= 5 samples): {Y_common.shape[1]}")
print(f"  {list(Y_common.columns)}")
print(f"Rare monosaccharides excluded (< 5 samples): {Y_rare.shape[1]}")

# ── 6. Define CAZyme vs non-CAZyme feature groups ─────────────────────
# CAZyme features: GT/GH/CBM/CE/PL family counts annotated by dbCAN
# non-CAZyme features: enzyme keyword annotations (kw_*),
#                      product name tokens (prod_tok_*),
#                      structural gene presence (has_wzx, has_wzz),
#                      total locus gene count
CAZYME_COLS = [c for c in X.columns if
               c.startswith('locus_G') or
               c in ['dbcan_locus_matched_proteins', 'locus_GT_total',
                     'locus_GH_total', 'locus_CE_total',
                     'locus_PL_total', 'locus_CBM_total']]
NON_CAZYME_COLS = [c for c in X.columns if c not in CAZYME_COLS]

print(f"\nCAZyme features     : {len(CAZYME_COLS)}")
print(f"non-CAZyme features : {len(NON_CAZYME_COLS)}")

# ── 7. Monosaccharide frequency summary ───────────────────────────────
print("\n" + "=" * 60)
print("Monosaccharide frequency in dataset:")
print("=" * 60)
nz    = (Y > 0).sum().sort_values(ascending=False)
total = len(Y)
for col, cnt in nz[nz > 0].items():
    bar = '#' * int(cnt / total * 30)
    tag = " <- used for ML" if cnt >= 5 else " (rare, excluded)"
    print(f"  {col:20s}: {cnt:3d}/{total} ({cnt/total*100:5.1f}%) {bar}{tag}")

# ── 8. Save processed data ────────────────────────────────────────────
X.to_csv("data/X_processed.tsv", sep='\t')
Y_common.to_csv("data/Y_common.tsv", sep='\t')

print("\nSaved: data/X_processed.tsv")
print("Saved: data/Y_common.tsv")
print("\nPreprocessing complete. Proceed to feature_importance.py")

# Export variables for import by downstream scripts
__all__ = ['X', 'Y_common', 'CAZYME_COLS', 'NON_CAZYME_COLS', 'total']