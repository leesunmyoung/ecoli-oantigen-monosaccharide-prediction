"""
fig_model_comparison.py
=======================
Figure: Per-monosaccharide R2 comparison across three models
        (Full / CAZyme-only / non-CAZyme-only)
        with summary table showing DeltaR2 and key biosynthetic genes.

Usage:
    python fig_model_comparison.py

Requirements:
    pip install pandas numpy matplotlib

Output:
    Fig2_model_comparison.png

Notes:
    - cv_results are hard-coded from 5-fold CV results (n=160)
    - To update with new CV results, modify the cv_data list
    - DGro has been merged into Gro (Y_monosaccharide_merged.tsv)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Remove if running interactively
import matplotlib.pyplot as plt

# ── CV results (5-fold cross-validation, n=160) ───────────────────────
# Columns: monosaccharide, frequency(%), pathway_type,
#          key_genes, R2_full, R2_cazyme, R2_noncazyme
# DeltaR2 = R2_noncazyme - R2_cazyme (computed automatically below)
# Sorted by R2_full descending
cv_data = [
    # sugar       freq  pathway_type                  key_genes              R2_full  R2_caz   R2_nc
    ('Rha',        33, 'Locus-encoded\nbiosynthetic', 'rmlA-D',              0.476,  -0.029,   0.471),
    ('Fuc',        10, 'Locus-encoded\nbiosynthetic', 'gmd, fcl',            0.335,  -0.012,   0.304),
    ('Man',        19, 'Locus-encoded\nbiosynthetic', 'manA, manB, manC',    0.244,   0.092,   0.254),
    ('QuiNAc',      3, 'Locus-encoded\nbiosynthetic', 'wlbA/B/C, fnlA/B/C', 0.143,  -0.001,   0.116),
    ('Galf',        5, 'Locus-encoded\nbiosynthetic', 'glf (UGM)',           0.143,   0.075,  -0.105),
    ('FucNAc',      5, 'Locus-encoded\nbiosynthetic', 'fdtA/B/C, wecB/C',   0.119,  -0.065,   0.093),
    ('GlcNAc',     68, 'Central metabolic\nprecursor','glmU, glmS',          0.122,  -0.122,   0.045),
    ('Gal',        46, 'Central metabolic\nprecursor','galE, galU',          0.058,  -0.008,   0.060),
    ('GalNAc',     35, 'Central metabolic\nprecursor','galE, glmU',          0.056,  -0.101,   0.026),
    ('Gro',        10, 'Central metabolic\nprecursor','wb cluster (n.c.)',   0.056,  -0.001,   0.048),
    ('GlcA',       15, 'Central metabolic\nprecursor','ugd',                -0.059,  -0.199,  -0.080),
    ('GalA',        8, 'Central metabolic\nprecursor','uxaC, uxaB',         -0.079,  -0.334,  -0.049),
    ('Glc',        42, 'Central metabolic\nprecursor','pgm, galU',          -0.081,  -0.104,  -0.123),
    ('Ribf',        6, 'Central metabolic\nprecursor','ribB, ribE',         -0.092,  -0.087,  -0.150),
    ('DGro',        4, 'Central metabolic\nprecursor','wb cluster (n.c.)',  -0.132,  -0.008,  -0.124),
]

# ── Unpack data ───────────────────────────────────────────────────────
sugars    = [d[0] for d in cv_data]
freq      = [d[1] for d in cv_data]
pathways  = [d[2] for d in cv_data]
key_genes = [d[3] for d in cv_data]
r2_full   = [d[4] for d in cv_data]
r2_caz    = [d[5] for d in cv_data]
r2_nc     = [d[6] for d in cv_data]
delta_r2  = [nc - cz for nc, cz in zip(r2_nc, r2_caz)]

means = [
    sum(r2_full) / len(r2_full),
    sum(r2_caz)  / len(r2_caz),
    sum(r2_nc)   / len(r2_nc),
]

# ── Colors ────────────────────────────────────────────────────────────
LOCUS_ROW_COLOR   = '#FEF9E7'   # yellow: locus-encoded pathway
CENTRAL_ROW_COLOR = '#F0F4F8'   # blue-grey: central metabolic precursor
NEG_CELL_COLOR    = '#EAECEE'   # grey: negative R2
NEG_CELL_TEXT     = '#7F8C8D'
POS_DELTA_COLOR   = '#D5F5E3'   # green: non-CAZyme superior
POS_DELTA_TEXT    = '#1E8449'
NEG_DELTA_COLOR   = '#FADBD8'   # red: CAZyme superior
NEG_DELTA_TEXT    = '#C0392B'
NC_GENE_COLOR     = '#E8DAEF'   # purple: not well characterized
NC_GENE_TEXT      = '#6C3483'

# ── Figure layout ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5),
                         gridspec_kw={'width_ratios': [1.4, 1.6]})

# ══════════════════════════════════════════════════════════════════════
# Left panel: Grouped bar chart
# ══════════════════════════════════════════════════════════════════════
ax = axes[0]
x = np.arange(len(sugars))
w = 0.26

ax.bar(x - w, r2_full, width=w, label='Full model',        color='#8E44AD', alpha=0.85)
ax.bar(x,     r2_caz,  width=w, label='CAZyme-only',       color='#E55A5A', alpha=0.85)
ax.bar(x + w, r2_nc,   width=w, label='non-CAZyme-only',   color='#2ECC71', alpha=0.85)

ax.axhline(0, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(sugars, rotation=40, ha='right', fontsize=9)
ax.set_ylabel('R² Score (5-fold CV)', fontsize=11)
ax.set_title('Prediction Performance Comparison\n'
             'Full vs CAZyme-only vs non-CAZyme-only (n=160, 5-fold CV)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(-0.45, 0.62)
ax.grid(True, alpha=0.2, linestyle='--', axis='y')

ax.text(0.01, 0.97,
        'Mean R²:  Full=%.3f  |  CAZyme=%.3f  |  non-CAZyme=%.3f' % tuple(means),
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#ccc'))

# ══════════════════════════════════════════════════════════════════════
# Right panel: Summary table
# ══════════════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.axis('off')

col_labels = [
    'Monosaccharide', 'Frequency\n(%)', 'Biosynthetic\npathway type',
    'Key biosynthetic\ngenes',
    'R²\n(Full)', 'R²\n(CAZyme)', 'R²\n(non-CAZyme)', 'ΔR²\n(NC−CAZ)'
]

table_data = []
for i in range(len(sugars)):
    table_data.append([
        sugars[i],
        '%d%%' % freq[i],
        pathways[i],
        key_genes[i],
        '%+.3f' % r2_full[i],
        '%+.3f' % r2_caz[i],
        '%+.3f' % r2_nc[i],
        '%+.3f' % delta_r2[i],
    ])

table = ax2.table(cellText=table_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7.8)
table.scale(1.18, 1.52)

# Header row styling
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#2C3E50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Data row styling
for i in range(len(sugars)):
    row_color = LOCUS_ROW_COLOR if 'Locus' in pathways[i] else CENTRAL_ROW_COLOR
    for j in range(len(col_labels)):
        table[(i+1, j)].set_facecolor(row_color)

    # Grey out negative R2 cells (columns 4, 5, 6)
    for col_idx, val in [(4, r2_full[i]), (5, r2_caz[i]), (6, r2_nc[i])]:
        if val < 0:
            table[(i+1, col_idx)].set_facecolor(NEG_CELL_COLOR)
            table[(i+1, col_idx)].set_text_props(color=NEG_CELL_TEXT)

    # Color DeltaR2 cell (column 7)
    if delta_r2[i] > 0:
        table[(i+1, 7)].set_facecolor(POS_DELTA_COLOR)
        table[(i+1, 7)].set_text_props(color=POS_DELTA_TEXT, fontweight='bold')
    else:
        table[(i+1, 7)].set_facecolor(NEG_DELTA_COLOR)
        table[(i+1, 7)].set_text_props(color=NEG_DELTA_TEXT, fontweight='bold')

    # Purple italic for genes not well characterized
    if 'n.c.' in key_genes[i]:
        table[(i+1, 3)].set_facecolor(NC_GENE_COLOR)
        table[(i+1, 3)].set_text_props(color=NC_GENE_TEXT, fontstyle='italic')

ax2.set_title(
    'R² summary with biosynthetic pathway classification\n'
    'Yellow = locus-encoded  |  Blue = central metabolic precursor  |  '
    'ΔR² = R²(non-CAZyme) − R²(CAZyme)',
    fontsize=8.5, fontweight='bold')

# ── Overall figure title ──────────────────────────────────────────────
fig.suptitle(
    'E. coli O-Antigen Monosaccharide Composition Prediction\n'
    'Comparison of Full, CAZyme-only, and non-CAZyme-only Models',
    fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('figures/three_model_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: Fig2_model_comparison.png')