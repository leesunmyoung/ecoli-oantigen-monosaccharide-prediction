"""
04_pathway_analysis.py
======================
Scatter plot: monosaccharide frequency (%) vs prediction R2
Boxplot: locus-encoded biosynthetic vs central metabolic precursor pathways
Statistical comparison: Mann-Whitney U test (one-sided)

Usage:
    python scripts/04_pathway_analysis.py
    (Run 03_model_comparison.py first to generate data/cv_results.csv)

Output:
    figures/Fig3_pathway_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load CV results from Step 3 ───────────────────────────────────────
try:
    df = pd.read_csv('data/cv_results.csv')
except FileNotFoundError:
    raise FileNotFoundError(
        "data/cv_results.csv not found. "
        "Please run 03_model_comparison.py first."
    )

# ── Binary pathway group classification ───────────────────────────────
df['pathway_group'] = df['pathway_type'].apply(
    lambda x: 'Locus-encoded\nbiosynthetic' if 'Locus' in x
              else 'Central metabolic\nprecursor'
)

COLOR = {
    'Locus-encoded\nbiosynthetic' : '#E74C3C',
    'Central metabolic\nprecursor': '#3498DB'
}

# ── Statistical tests ─────────────────────────────────────────────────
locus   = df[df.pathway_group == 'Locus-encoded\nbiosynthetic']['R2_full'].values
central = df[df.pathway_group == 'Central metabolic\nprecursor']['R2_full'].values

# Mann-Whitney U test: one-sided (locus-encoded > central metabolic)
_, p_mw = stats.mannwhitneyu(locus, central, alternative='greater')
r_pearson, p_pearson = stats.pearsonr(df.frequency_pct, df.R2_full)

print(f"Locus-encoded (n={len(locus)}):    mean R2 = {np.mean(locus):.3f}")
print(f"Central metabolic (n={len(central)}): mean R2 = {np.mean(central):.3f}")
print(f"Mann-Whitney U test (one-sided): p = {p_mw:.4f}")
print(f"Pearson r (frequency vs R2):     r = {r_pearson:.3f}, p = {p_pearson:.3f} (n.s.)")

# ── Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left panel: scatter plot ──────────────────────────────────────────
ax = axes[0]
for grp, sub in df.groupby('pathway_group'):
    ax.scatter(sub.frequency_pct, sub.R2_full,
               color=COLOR[grp], s=130, zorder=3,
               edgecolors='white', linewidth=0.8,
               label=f'{grp} (n={len(sub)})')

# Label individual points (manual offsets to avoid overlap)
OFFSETS = {
    'GlcNAc': (-14,  0.013), 'Gal':    (-12,  0.013),
    'GalNAc': (  1.5,-0.038), 'Glc':   (  1.5,-0.038),
    'Rha':    (-10,  0.015),
}
for _, row in df.iterrows():
    ox, oy = OFFSETS.get(row.monosaccharide, (1.5, 0.013))
    ax.annotate(row.monosaccharide,
                xy=(row.frequency_pct, row.R2_full),
                xytext=(row.frequency_pct + ox, row.R2_full + oy),
                fontsize=8.5, color='#333')

ax.axhline(0, color='gray', linewidth=1.0, alpha=0.4)
ax.set_xlabel('Frequency in dataset (% of 160 samples)', fontsize=11)
ax.set_ylabel('R2 Score (5-fold CV, Full model)', fontsize=11)
ax.set_title('Monosaccharide Frequency vs Prediction R2\n'
             'Colored by Biosynthetic Pathway Type', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-5, 80)
ax.set_ylim(-0.25, 0.58)
ax.grid(True, alpha=0.2, linestyle='--')
ax.text(0.97, 0.04,
        f'Pearson r={r_pearson:.2f}, p={p_pearson:.2f} (n.s.)',
        transform=ax.transAxes, fontsize=9, ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc'))

# ── Right panel: boxplot with individual data points ──────────────────
ax2 = axes[1]
groups     = ['Locus-encoded\nbiosynthetic', 'Central metabolic\nprecursor']
group_data = [locus, central]
box_colors = ['#E74C3C', '#3498DB']

bp = ax2.boxplot(group_data, patch_artist=True, widths=0.45,
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual points with jitter for transparency
rng = np.random.RandomState(42)
for i, (vals, grp, sub) in enumerate(
        zip(group_data, groups,
            [df[df.pathway_group == g] for g in groups]), 1):
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax2.scatter([i + j for j in jitter], vals,
                color=box_colors[i-1], s=70, zorder=3,
                edgecolors='white', linewidth=0.8, alpha=0.9)
    for j, (v, row) in enumerate(zip(vals, sub.itertuples())):
        ax2.annotate(row.monosaccharide,
                     xy=(i + jitter[j], v),
                     xytext=(i + jitter[j] + 0.09, v + 0.008),
                     fontsize=7.5, color='#444')

# Significance bracket
sig_str = '**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else f'p={p_mw:.3f}')
y_max   = max(max(locus), max(central)) + 0.07
ax2.plot([1,1,2,2], [y_max-0.03, y_max, y_max, y_max-0.03],
         color='black', linewidth=1.2)
ax2.text(1.5, y_max + 0.01, sig_str, ha='center', fontsize=13, fontweight='bold')

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Locus-encoded\nbiosynthetic',
                     'Central metabolic\nprecursor'], fontsize=10)
ax2.set_ylabel('R2 Score (5-fold CV, Full model)', fontsize=11)
ax2.set_title(
    f'Prediction R2 by Biosynthetic Pathway Type\n'
    f'Mann-Whitney U test (one-sided), p = {p_mw:.4f}',
    fontsize=11, fontweight='bold')
ax2.axhline(0, color='gray', linewidth=1.0, alpha=0.4, linestyle='--')
ax2.set_ylim(-0.25, 0.65)
ax2.grid(True, alpha=0.2, linestyle='--', axis='y')

for i, (vals, color) in enumerate(zip(group_data, box_colors), 1):
    ax2.text(i, np.mean(vals) + 0.03,
             f'mean={np.mean(vals):.3f}',
             ha='center', fontsize=9, color=color, fontweight='bold')

fig.suptitle(
    'E. coli O-Antigen Monosaccharide Prediction\n'
    'Locus-encoded vs Central Metabolic Precursor Pathway Groups',
    fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('figures/Fig3_pathway_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: figures/Fig3_pathway_analysis.png")
print("\nAll analysis complete.")
print("Generated figures:")
print("  figures/Fig1_feature_importance.png")
print("  figures/Fig2_model_comparison.png")
print("  figures/Fig3_pathway_analysis.png")