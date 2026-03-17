"""
03_model_comparison.py
======================
5-fold cross-validation R² comparison across three models:
  - Full model          (all 81 features)
  - CAZyme-only model   (19 CAZyme features)
  - non-CAZyme-only     (62 non-CAZyme features)

DeltaR2 = R2(non-CAZyme) - R2(CAZyme) is computed per monosaccharide.
Monosaccharides are classified by biosynthetic pathway type.

Usage:
    python scripts/03_model_comparison.py
    (Run 01_data_preprocessing.py first)

Output:
    figures/Fig2_model_comparison.png
    data/cv_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

# ── Load preprocessed data ────────────────────────────────────────────
X        = pd.read_csv("data/X_processed.tsv", sep='\t', index_col=0)
Y_common = pd.read_csv("data/Y_common.tsv",    sep='\t', index_col=0)

# Define feature groups
CAZYME_COLS = [c for c in X.columns if
               c.startswith('locus_G') or
               c in ['dbcan_locus_matched_proteins', 'locus_GT_total',
                     'locus_GH_total', 'locus_CE_total',
                     'locus_PL_total', 'locus_CBM_total']]
NON_CAZYME_COLS = [c for c in X.columns if c not in CAZYME_COLS]

# ── Biosynthetic pathway classification ───────────────────────────────
# Locus-encoded: biosynthetic genes characteristically located within
#                the O-antigen locus (wb/rfb cluster)
# Central metabolic precursor: derived from housekeeping pathways;
#                              present regardless of O-antigen type
PATHWAY_INFO = {
    'Rha':    ('Locus-encoded biosynthetic',    'rmlA-D'),
    'Fuc':    ('Locus-encoded biosynthetic',    'gmd, fcl'),
    'Man':    ('Locus-encoded biosynthetic',    'manA, manB, manC'),
    'Galf':   ('Locus-encoded biosynthetic',    'glf (UGM)'),
    'FucNAc': ('Locus-encoded biosynthetic',    'fdtA/B/C, wecB/C'),
    'QuiNAc': ('Locus-encoded biosynthetic',    'wlbA/B/C, fnlA/B/C'),
    'GlcNAc': ('Central metabolic precursor',   'glmU, glmS'),
    'Gal':    ('Central metabolic precursor',   'galE, galU'),
    'Glc':    ('Central metabolic precursor',   'pgm, galU'),
    'GalNAc': ('Central metabolic precursor',   'galE, glmU'),
    'GlcA':   ('Central metabolic precursor',   'ugd'),
    'GalA':   ('Central metabolic precursor',   'uxaC, uxaB'),
    'Ribf':   ('Central metabolic precursor',   'ribB, ribE'),
    'Gro':    ('Central metabolic precursor',   'wb cluster (not well characterized)'),
    'DGro':   ('Central metabolic precursor',   'wb cluster (not well characterized)'),
}

# ── Model parameters and CV setup ─────────────────────────────────────
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 5-fold CV for all monosaccharides x 3 models ──────────────────────
print("Running 5-fold cross-validation for all monosaccharides...")
print("(This may take a few minutes)\n")

results = []
for sugar in Y_common.columns:
    y      = Y_common[sugar].values
    nz_pct = (y > 0).sum() / len(y) * 100

    r2_full = cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X, y,
        cv=kf, scoring='r2').mean()
    r2_caz  = cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X[CAZYME_COLS], y,
        cv=kf, scoring='r2').mean()
    r2_nc   = cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X[NON_CAZYME_COLS], y,
        cv=kf, scoring='r2').mean()
    mae     = -cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X, y,
        cv=kf, scoring='neg_mean_absolute_error').mean()

    pathway, key_genes = PATHWAY_INFO.get(sugar, ('Unknown', 'unknown'))

    results.append({
        'monosaccharide': sugar,
        'frequency_pct' : round(nz_pct, 1),
        'pathway_type'  : pathway,
        'key_genes'     : key_genes,
        'R2_full'       : round(r2_full, 3),
        'R2_cazyme'     : round(r2_caz,  3),
        'R2_noncazyme'  : round(r2_nc,   3),
        'delta_R2'      : round(r2_nc - r2_caz, 3),
        'MAE_full'      : round(mae, 3),
    })
    print(f"  {sugar:12s}: Full={r2_full:+.3f}  CAZyme={r2_caz:+.3f}  "
          f"non-CAZyme={r2_nc:+.3f}  MAE={mae:.3f}")

df = pd.DataFrame(results).sort_values('R2_full', ascending=False)

print(f"\nMean R2:  Full={df.R2_full.mean():.3f}  "
      f"CAZyme={df.R2_cazyme.mean():.3f}  "
      f"non-CAZyme={df.R2_noncazyme.mean():.3f}")
print(f"DeltaR2 > 0 (non-CAZyme superior): {(df.delta_R2 > 0).sum()}/{len(df)}")

# ── Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5),
                         gridspec_kw={'width_ratios': [1.4, 1.6]})

# Left panel: grouped bar chart per monosaccharide
ax = axes[0]
x  = np.arange(len(df))
w  = 0.26
ax.bar(x-w, df.R2_full,      width=w, label='Full model',        color='#8E44AD', alpha=0.85)
ax.bar(x,   df.R2_cazyme,    width=w, label='CAZyme-only',       color='#E55A5A', alpha=0.85)
ax.bar(x+w, df.R2_noncazyme, width=w, label='non-CAZyme-only',   color='#2ECC71', alpha=0.85)
ax.axhline(0, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(df.monosaccharide, rotation=40, ha='right', fontsize=9)
ax.set_ylabel('R2 Score (5-fold CV)', fontsize=11)
ax.set_title('Prediction Performance Comparison\n'
             'Full vs CAZyme-only vs non-CAZyme-only (n=160, 5-fold CV)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(-0.45, 0.62)
ax.grid(True, alpha=0.2, linestyle='--', axis='y')

means = [df.R2_full.mean(), df.R2_cazyme.mean(), df.R2_noncazyme.mean()]
ax.text(0.01, 0.97,
        'Mean R2:  Full=%.3f  |  CAZyme=%.3f  |  non-CAZyme=%.3f' % tuple(means),
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#ccc'))

# Right panel: summary table
ax2 = axes[1]
ax2.axis('off')

col_labels = ['Monosaccharide', 'Frequency\n(%)', 'Biosynthetic\npathway type',
              'Key biosynthetic\ngenes', 'R2\n(Full)', 'R2\n(CAZyme)',
              'R2\n(non-CAZyme)', 'DeltaR2\n(NC-CAZ)']

table_data = [[
    r.monosaccharide,
    '%s%%' % r.frequency_pct,
    r.pathway_type,
    r.key_genes,
    '%+.3f' % r.R2_full,
    '%+.3f' % r.R2_cazyme,
    '%+.3f' % r.R2_noncazyme,
    '%+.3f' % r.delta_R2,
] for r in df.itertuples()]

table = ax2.table(cellText=table_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7.8)
table.scale(1.18, 1.52)

LOCUS_COLOR   = '#FEF9E7'  # yellow: locus-encoded pathway
CENTRAL_COLOR = '#F0F4F8'  # blue-grey: central metabolic precursor

for j in range(len(col_labels)):
    table[(0,j)].set_facecolor('#2C3E50')
    table[(0,j)].set_text_props(color='white', fontweight='bold')

for i, row in enumerate(df.itertuples(), 1):
    rc = LOCUS_COLOR if 'Locus' in row.pathway_type else CENTRAL_COLOR
    for j in range(len(col_labels)):
        table[(i,j)].set_facecolor(rc)
    # Grey out negative R2 cells
    for ci, val in [(4, row.R2_full),(5, row.R2_cazyme),(6, row.R2_noncazyme)]:
        if val < 0:
            table[(i,ci)].set_facecolor('#EAECEE')
            table[(i,ci)].set_text_props(color='#7F8C8D')
    # Color DeltaR2: green = non-CAZyme superior, red = CAZyme superior
    if row.delta_R2 > 0:
        table[(i,7)].set_facecolor('#D5F5E3')
        table[(i,7)].set_text_props(color='#1E8449', fontweight='bold')
    else:
        table[(i,7)].set_facecolor('#FADBD8')
        table[(i,7)].set_text_props(color='#C0392B', fontweight='bold')
    # Purple italic for genes not well characterized
    if row.monosaccharide in ['Gro', 'DGro']:
        table[(i,3)].set_facecolor('#E8DAEF')
        table[(i,3)].set_text_props(color='#6C3483', fontstyle='italic')

ax2.set_title(
    'R2 summary with biosynthetic pathway classification\n'
    'Yellow = locus-encoded  |  Blue = central metabolic precursor  |  '
    'DeltaR2 = R2(non-CAZyme) - R2(CAZyme)',
    fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/Fig2_model_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: figures/Fig2_model_comparison.png")

df.to_csv('data/cv_results.csv', index=False)
print("Saved: data/cv_results.csv")
print("\nProceed to 04_pathway_analysis.py")