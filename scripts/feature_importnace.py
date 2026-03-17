"""
02_feature_importance.py
========================
MDI (Mean Decrease Impurity) feature importance analysis.
Compares total contribution of CAZyme vs non-CAZyme feature groups.

Usage:
    python scripts/02_feature_importance.py
    (Run 01_data_preprocessing.py first)

Output:
    figures/Fig1_feature_importance.png
    data/feature_importance_MDI.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ── Load preprocessed data ────────────────────────────────────────────
X        = pd.read_csv("data/X_processed.tsv",  sep='\t', index_col=0)
Y_common = pd.read_csv("data/Y_common.tsv",      sep='\t', index_col=0)

# Define CAZyme vs non-CAZyme feature groups
CAZYME_COLS = [c for c in X.columns if
               c.startswith('locus_G') or
               c in ['dbcan_locus_matched_proteins', 'locus_GT_total',
                     'locus_GH_total', 'locus_CE_total',
                     'locus_PL_total', 'locus_CBM_total']]
NON_CAZYME_COLS = [c for c in X.columns if c not in CAZYME_COLS]

# ── Random Forest hyperparameters ─────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ── Train multi-output Random Forest ──────────────────────────────────
print("Training Random Forest (MultiOutputRegressor)...")
model = MultiOutputRegressor(RandomForestRegressor(**RF_PARAMS))
model.fit(X, Y_common)
print("Training complete")

# ── Compute MDI feature importance ────────────────────────────────────
# Average importance across all per-output estimators
fi_matrix = np.array([est.feature_importances_ for est in model.estimators_])
fi_mean   = fi_matrix.mean(axis=0)
fi_std    = fi_matrix.std(axis=0)

fi_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': fi_mean,
    'std'       : fi_std,
    'group'     : ['CAZyme' if c in CAZYME_COLS else 'non-CAZyme' for c in X.columns]
}).sort_values('importance', ascending=False).reset_index(drop=True)

# ── Summarize group contributions ─────────────────────────────────────
caz_sum = fi_df[fi_df.group == 'CAZyme']['importance'].sum()
nc_sum  = fi_df[fi_df.group == 'non-CAZyme']['importance'].sum()
total   = caz_sum + nc_sum

print(f"\nFeature importance by group:")
print(f"  CAZyme     : {caz_sum:.4f} ({caz_sum/total*100:.1f}%)")
print(f"  non-CAZyme : {nc_sum:.4f}  ({nc_sum/total*100:.1f}%)")
print(f"\nTop 15 features:")
print(fi_df.head(15).to_string(index=False))

# ── Visualization ─────────────────────────────────────────────────────
COLOR   = {'CAZyme': '#E55A5A', 'non-CAZyme': '#2ECC71'}
patches = [mpatches.Patch(color=v, label=k) for k, v in COLOR.items()]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left panel: Top 20 MDI importance bar chart
ax1    = axes[0]
top20  = fi_df.head(20)
colors = [COLOR[g] for g in top20['group']]
ax1.barh(range(len(top20)), top20['importance'],
         xerr=top20['std'], color=colors, capsize=3,
         alpha=0.85, edgecolor='white')
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['feature'], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Mean Decrease Impurity (MDI)', fontsize=11)
ax1.set_title('Top 20 Feature Importance (MDI)\nRed = CAZyme  |  Green = non-CAZyme',
              fontsize=11, fontweight='bold')
ax1.legend(handles=patches, fontsize=9)
ax1.grid(True, alpha=0.2, axis='x', linestyle='--')

# Right panel: Pie chart — total CAZyme vs non-CAZyme contribution
ax2 = axes[1]
ax2.pie(
    [caz_sum, nc_sum],
    labels=[f'CAZyme\n{caz_sum/total*100:.1f}%',
            f'non-CAZyme\n{nc_sum/total*100:.1f}%'],
    colors=['#E55A5A', '#2ECC71'],
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
ax2.set_title('Total Feature Importance\nCAZyme vs non-CAZyme',
              fontsize=12, fontweight='bold')

fig.suptitle(
    'E. coli O-Antigen Monosaccharide Prediction\n'
    'Feature Importance Analysis (Random Forest MDI, n=160)',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('figures/Fig1_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: figures/Fig1_feature_importance.png")

# ── Save results ──────────────────────────────────────────────────────
fi_df.to_csv('data/feature_importance_MDI.csv', index=False)
print("Saved: data/feature_importance_MDI.csv")
print("\nProceed to 03_model_comparison.py")