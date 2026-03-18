"""
scatter_pathway_final.py
========================
Figure: Monosaccharide frequency vs prediction R²
        + Boxplot by biosynthetic pathway type

Usage:
    python scatter_pathway_final.py

Requirements:
    pip install pandas numpy matplotlib scipy

Output:
    scatter_pathway_final.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Remove this line if running interactively (Jupyter etc.)
import matplotlib.pyplot as plt
from scipy import stats

# ── Data: CV results per monosaccharide ──────────────────────────────
# Columns: sugar name, R2 (5-fold CV, Full model), frequency (%), pathway type
cv_data = [
    ('Fuc',     0.335, 11, 'Locus-encoded\npathway'),
    ('FucNAc',  0.119,  6, 'Locus-encoded\npathway'),
    ('Gal',     0.058, 46, 'Central metabolic\nprecursor'),
    ('GalA',   -0.079,  8, 'Central metabolic\nprecursor'),
    ('GalNAc',  0.056, 35, 'Central metabolic\nprecursor'),
    ('Galf',    0.143,  5, 'Locus-encoded\npathway'),
    ('Glc',    -0.081, 42, 'Central metabolic\nprecursor'),
    ('GlcA',   -0.059, 15, 'Central metabolic\nprecursor'),
    ('GlcNAc',  0.122, 68, 'Central metabolic\nprecursor'),
    ('Man',     0.244, 19, 'Locus-encoded\npathway'),
    ('QuiNAc',  0.143,  3, 'Locus-encoded\npathway'),
    ('Rha',     0.476, 33, 'Locus-encoded\npathway'),
    ('Ribf',   -0.092,  6, 'Central metabolic\nprecursor'),
    ('Gro',     0.056,  10, 'Central metabolic\nprecursor'),
]
df = pd.DataFrame(cv_data, columns=['sugar', 'R2', 'nz_pct', 'pathway'])

# ── Colors ────────────────────────────────────────────────────────────
COLOR = {
    'Locus-encoded\npathway'       : '#E74C3C',   # red
    'Central metabolic\nprecursor' : '#3498DB'    # blue
}

# ── Statistical tests ─────────────────────────────────────────────────
locus   = df[df.pathway == 'Locus-encoded\npathway']['R2'].values
central = df[df.pathway == 'Central metabolic\nprecursor']['R2'].values
_, p_mw  = stats.mannwhitneyu(locus, central, alternative='greater')
r_p, p_p = stats.pearsonr(df.nz_pct, df.R2)

# ── Font size for data point labels ──────────────────────────────────
LABEL_FS = 10.5   # Adjust here to change label size

# ── Label offsets (x, y) to avoid overlap ────────────────────────────
OFFSETS = {
    'Rha'   : ( 1.5,  0.015),
    'Fuc'   : ( 1.5,  0.015),
    'Man'   : ( 1.5,  0.015),
    'FucNAc': ( 1.5,  0.025),
    'Galf'  : ( 1.5, -0.038),
    'QuiNAc': ( -0.9,  0.025),
    'GlcNAc': (-2.0, -0.042),
    'Gal'   : ( 1.5,  0.015),
    'Glc'   : ( 1.5,  0.015),
    'GalNAc': ( 1.5, -0.038),
    'GlcA'  : ( 1.5,  0.015),
    'GalA'  : ( -1.5,  0.015),
    'Ribf'  : ( -0.4, -0.04),
    'Gro'   : ( 1.5,  0.015),
    'DGro'  : ( 1.5,  0.015),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left panel: Scatter plot ──────────────────────────────────────────
ax = axes[0]

for grp, sub in df.groupby('pathway'):
    ax.scatter(sub.nz_pct, sub.R2,
               color=COLOR[grp], s=130, zorder=3,
               edgecolors='white', linewidth=0.8,
               label=grp.replace('\n', ' '))

for _, row in df.iterrows():
    ox, oy = OFFSETS.get(row.sugar, (1.5, 0.015))
    ax.annotate(row.sugar,
                xy=(row.nz_pct, row.R2),
                xytext=(row.nz_pct + ox, row.R2 + oy),
                fontsize=LABEL_FS, fontweight='bold', color='#222')

ax.axhline(0, color='gray', linewidth=1.0, alpha=0.4)
ax.set_xlabel('Frequency in dataset (% of 160 samples)', fontsize=11)
ax.set_ylabel('R² Score (5-fold CV)', fontsize=11)
ax.set_title('Monosaccharide Frequency vs Prediction R²\n'
             'Colored by Biosynthetic Pathway Type',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5, loc='lower right')
ax.set_xlim(-5, 80)
ax.set_ylim(-0.25, 0.60)
ax.grid(True, alpha=0.2, linestyle='--')
ax.text(0.97, 0.97,
        f'Pearson r = {r_p:.2f}, p = {p_p:.2f} (n.s.)',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc'))

# ── Right panel: Boxplot ──────────────────────────────────────────────
ax2 = axes[1]
groups     = ['Locus-encoded\npathway', 'Central metabolic\nprecursor']
group_data = [locus, central]
box_colors = ['#E74C3C', '#3498DB']

bp = ax2.boxplot(group_data, patch_artist=True, widths=0.45,
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

rng = np.random.RandomState(42)
for i, (vals, grp, sub) in enumerate(
        zip(group_data, groups,
            [df[df.pathway == g] for g in groups]), 1):
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax2.scatter([i + j for j in jitter], vals,
                color=box_colors[i-1], s=70, zorder=3,
                edgecolors='white', linewidth=0.8, alpha=0.9)
    
    
    # 박스플롯용 offset 딕셔너리 추가
    BOX_OFFSETS = {
        'Rha'   : ( 0.12,  0.010),
        'Fuc'   : ( 0.12,  0.010),
        'Man'   : ( 0.12,  0.010),
        'FucNAc': ( 0.12,  0.025),
        'Galf'  : (-0.15,  0.010),
        'QuiNAc': ( 0.12, -0.030),
        'GlcNAc': ( -0.02,  0.010),
        'Gal'   : ( 0.12,  0.0020),
        'GalNAc': (-0.15,  0.010),
        'Gro'   : (-0.15, -0.025),
        'GlcA'  : ( 0.12,  0.010),
        'GalA'  : ( 0.12, -0.025),
        'Glc'   : (-0.15,  0.010),
        'Ribf'  : (0.15, -0.045),
        'DGro'  : ( 0.12,  0.010),
    }

    # modify annotate 
    for j, (v, row) in enumerate(zip(vals, sub.itertuples())):
        ox, oy = BOX_OFFSETS.get(row.sugar, (0.09, 0.008))  
        ha = 'right' if ox < 0 else 'left'                  
        ax2.annotate(row.sugar,
                    xy=(i + jitter[j], v),
                    xytext=(i + jitter[j] + ox, v + oy),
                    fontsize=LABEL_FS, fontweight='bold', color='#222',
                    ha=ha)

# Significance bracket
sig_str = '**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else f'p={p_mw:.3f}')
y_max = max(max(locus), max(central)) + 0.07
ax2.plot([1, 1, 2, 2], [y_max-0.03, y_max, y_max, y_max-0.03],
         color='black', linewidth=1.2)
ax2.text(1.5, y_max + 0.01, sig_str,
         ha='center', fontsize=13, fontweight='bold')

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Locus-encoded\npathway',
                     'Central metabolic\nprecursor'], fontsize=10)
ax2.set_ylabel('R² Score (5-fold CV)', fontsize=11)
ax2.set_title(f'Prediction R² by Biosynthetic Pathway Type\n'
              f'Mann-Whitney U test (one-sided), p = {p_mw:.4f}',
              fontsize=11, fontweight='bold')
ax2.axhline(0, color='gray', linewidth=1.0, alpha=0.4, linestyle='--')
ax2.set_ylim(-0.25, 0.72)
ax2.grid(True, alpha=0.2, linestyle='--', axis='y')

# Mean labels: black, offset left/right to avoid overlap
ax2.text(0.82, np.mean(locus) + 0.03,
         f'mean={np.mean(locus):.3f}',
         ha='center', fontsize=9, color='black', fontweight='bold')
ax2.text(2.18, np.mean(central) + 0.03,
         f'mean={np.mean(central):.3f}',
         ha='center', fontsize=9, color='black', fontweight='bold')

fig.suptitle(
    'E. coli O-Antigen Monosaccharide Prediction\n'
    'Locus-encoded vs Central Metabolic Precursor Pathway Groups',
    fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('figures/scatter_pathway_final.png', dpi=300, bbox_inches='tight')
print('Saved: scatter_pathway_final.png')