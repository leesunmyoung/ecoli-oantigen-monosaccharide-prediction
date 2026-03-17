# Data Description

## Files

### `supervised_dataset.tsv`
- **Samples:** 203 O-antigen subtypes
- **Columns:** 188 total
  - X features (106): CAZyme family counts (GT, GH, CBM, CE, PL) from dbCAN + structural genes
  - Y labels (82): monosaccharide counts per O-antigen subtype
- **Source:** Genomic features extracted from NCBI genome sequences; monosaccharide composition from ECODAB

### `oantigen_locus_feature_table_mlX_binTok.tsv`
- **Samples:** 160 O-antigen subtypes (43 excluded due to feature extraction failure)
- **Columns:** 90 features
  - non-CAZyme (62): `kw_*` (enzyme keyword annotations), `prod_tok_*` (product name tokens), `locus_gene_count`, `has_wzx/wzz`
  - CAZyme (19): GT/GH family counts, `dbcan_locus_matched_proteins`, `locus_GT_total`
  - Zero-variance (9): removed before analysis
- **Used for:** Main ML analysis (non-CAZyme vs CAZyme comparison)

### `Y_monosaccharide.tsv`
- **Samples:** 216 O-antigen subtypes (including duplicates)
- **Columns:** 83 monosaccharides (count per subtype)
- **Note:** 13 duplicate indices — handled by `groupby().mean()` in preprocessing
- **Monosaccharides used for ML (>= 5 samples):** GlcNAc, Gal, Glc, GalNAc, Rha, Man, GlcA, Fuc, GalA, Ribf, Galf, FucNAc, Gro, DGro, QuiNAc

## Generated files (after running scripts)
- `X_processed.tsv`: X after zero-variance removal (160 samples × 81 features)
- `Y_common.tsv`: Y filtered to >= 5 samples (160 samples × 15 monosaccharides)
- `feature_importance_MDI.csv`: MDI importance per feature
- `cv_results.csv`: 5-fold CV R² per monosaccharide for all three models
