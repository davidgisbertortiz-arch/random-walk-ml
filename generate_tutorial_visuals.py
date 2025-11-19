"""
Quick script to generate tutorial visualizations without interactive pauses.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enhanced_model import (
    WalkConfig, BiasDistribution,
    generate_random_walks_1d,
    FeatureConfig,
    make_windows_from_walks_enhanced,
    group_train_test_split,
    build_pipeline,
    evaluate,
    get_feature_importance
)

print("🚀 Generando visualizaciones del tutorial...")

# Create output directory
Path('outputs').mkdir(exist_ok=True)

# ============================================================================
# VISUALIZATION 1: Fair vs Biased Walk
# ============================================================================
print("\n📊 1. Generando: Fair vs Biased walk...")

np.random.seed(42)

# Fair walk
config_fair = WalkConfig(n_walks=1, n_steps=500, bias_mode="fair", seed=42)
walks_fair, _ = generate_random_walks_1d(config_fair)
walk_fair = walks_fair[0]

# Biased walk
config_biased = WalkConfig(n_walks=1, n_steps=500, bias_mode="biased", p_up=0.6, seed=43)
walks_biased, _ = generate_random_walks_1d(config_biased)
walk_biased = walks_biased[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(walk_fair, linewidth=2, color='steelblue')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('Fair Random Walk\n(50% up, 50% down)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Step Number')
axes[0].set_ylabel('Position')
axes[0].grid(alpha=0.3)

axes[1].plot(walk_biased, linewidth=2, color='coral')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Biased Random Walk\n(60% up, 40% down)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Step Number')
axes[1].set_ylabel('Position')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tutorial_01_fair_vs_biased.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_01_fair_vs_biased.png")

# ============================================================================
# VISUALIZATION 2: Multiple Mixed Walks
# ============================================================================
print("\n📊 2. Generando: Multiple mixed walks...")

bias_dist = BiasDistribution(
    fair_prob=0.5,
    positive_bias_prob=0.5,
    negative_bias_prob=0.0
)

config_mixed = WalkConfig(
    n_walks=50,
    n_steps=500,
    bias_mode="mixed",
    bias_distribution=bias_dist,
    seed=42
)

walks, p_ups = generate_random_walks_1d(config_mixed)

fig, ax = plt.subplots(figsize=(14, 6))

# Plot biased walks (p_up > 0.52)
biased_idx = np.where(p_ups > 0.52)[0][:10]
for idx in biased_idx:
    ax.plot(walks[idx], alpha=0.6, color='coral', linewidth=1.5)

# Plot fair walks (p_up close to 0.5)
fair_idx = np.where(np.abs(p_ups - 0.5) < 0.02)[0][:10]
for idx in fair_idx:
    ax.plot(walks[idx], alpha=0.6, color='steelblue', linewidth=1.5)

ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Sample of Mixed Random Walks\n(Coral = Biased, Blue = Fair)', fontsize=14, fontweight='bold')
ax.set_xlabel('Step Number')
ax.set_ylabel('Position')
ax.grid(alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='coral', lw=2, label='Biased (p > 0.52)'),
    Line2D([0], [0], color='steelblue', lw=2, label='Fair (p ≈ 0.50)')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('outputs/tutorial_02_mixed_walks.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_02_mixed_walks.png")

# ============================================================================
# VISUALIZATION 3: Feature Engineering
# ============================================================================
print("\n📊 3. Generando: Feature engineering comparison...")

window_size = 20

# Basic features
feat_config_basic = FeatureConfig(use_raw_deltas=True, use_statistics=False, use_trend=False)
X_basic, y_basic, groups_basic = make_windows_from_walks_enhanced(walks, window_size, feature_config=feat_config_basic)

# Enhanced features
feat_config_enhanced = FeatureConfig(
    use_raw_deltas=True,
    use_statistics=True,
    statistics=["mean", "std", "skew"],
    use_trend=True
)
X_enhanced, y_enhanced, groups_enhanced = make_windows_from_walks_enhanced(walks, window_size, feature_config=feat_config_enhanced)

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Basic\n(Raw Deltas)', 'Enhanced\n(+ Stats + Trend)']
feature_counts = [X_basic.shape[1], X_enhanced.shape[1]]
sample_counts = [X_basic.shape[0], X_enhanced.shape[0]]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, feature_counts, width, label='Features', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, [c/10 for c in sample_counts], width, label='Samples (÷10)', color='coral', alpha=0.8)

ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Feature Engineering Impact', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/tutorial_03_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_03_feature_engineering.png")

# ============================================================================
# VISUALIZATION 4: Group-Aware Validation
# ============================================================================
print("\n📊 4. Generando: Group-aware validation comparison...")

# Use enhanced features
X = X_enhanced
y = y_enhanced
groups = groups_enhanced

# Group-aware split
X_train, X_test, y_train, y_test, groups_train, groups_test = group_train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sample distribution
ax1 = axes[0]
categories = ['Training', 'Testing']
samples = [len(X_train), len(X_test)]
walks = [len(np.unique(groups_train)), len(np.unique(groups_test))]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, samples, width, label='Samples', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, [w*50 for w in walks], width, label='Walks (×50)', color='coral', alpha=0.8)

ax1.set_ylabel('Count', fontweight='bold')
ax1.set_title('Group-Aware Split Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Walk overlap verification
ax2 = axes[1]
train_groups_set = set(groups_train)
test_groups_set = set(groups_test)
overlap = train_groups_set.intersection(test_groups_set)

categories = ['Train Only', 'Test Only', 'Overlap\n(SHOULD BE 0!)']
counts = [len(train_groups_set), len(test_groups_set), len(overlap)]
colors = ['steelblue', 'coral', 'red' if overlap else 'lightgreen']

bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
ax2.set_ylabel('Number of Walks', fontweight='bold')
ax2.set_title('Walk Separation Verification', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/tutorial_04_group_aware_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_04_group_aware_validation.png")

# ============================================================================
# VISUALIZATION 5: Model Comparison (Quick version)
# ============================================================================
print("\n📊 5. Generando: Model comparison (esto puede tomar ~30 segundos)...")

models_to_test = ["dummy_stratified", "logreg", "hgb"]
results = []

for model_name in models_to_test:
    print(f"   Entrenando {model_name}...")
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_train, y_train, X_test, y_test)
    
    results.append({
        'Model': model_name,
        'Train_ROC_AUC': metrics['train']['roc_auc'],
        'Test_ROC_AUC': metrics['test']['roc_auc']
    })

fig, ax = plt.subplots(figsize=(10, 6))

model_names = [r['Model'] for r in results]
train_scores = [r['Train_ROC_AUC'] for r in results]
test_scores = [r['Test_ROC_AUC'] for r in results]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, train_scores, width, label='Training', color='lightgreen', alpha=0.8)
bars2 = ax.bar(x + width/2, test_scores, width, label='Testing', color='steelblue', alpha=0.8)

ax.axhline(0.5, color='red', linestyle='--', label='Random Chance', linewidth=2, alpha=0.5)
ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(0.4, max(max(train_scores), max(test_scores)) * 1.1)
ax.grid(axis='y', alpha=0.3)

for i, (model, train, test) in enumerate(zip(model_names, train_scores, test_scores)):
    ax.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/tutorial_05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_05_model_comparison.png")

# ============================================================================
# VISUALIZATION 6: Feature Importance
# ============================================================================
print("\n📊 6. Generando: Feature importance (Random Forest)...")

rf_pipeline = build_pipeline("rf")
rf_pipeline.fit(X_train, y_train)

importance_df = get_feature_importance(rf_pipeline, feat_config_enhanced, window_size, top_k=15)

fig, ax = plt.subplots(figsize=(12, 8))

colors_map = {'raw': 'steelblue', 'stats': 'coral', 'trend': 'mediumseagreen'}

colors = []
for feat in importance_df['feature']:
    if feat in ['mean', 'std', 'skew', 'kurtosis']:
        colors.append(colors_map['stats'])
    elif feat == 'trend':
        colors.append(colors_map['trend'])
    else:
        colors.append(colors_map['raw'])

bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors, alpha=0.8)
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_ylabel('Feature', fontweight='bold')
ax.set_title('Top 15 Feature Importance for Bias Detection\n(Random Forest)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_map['raw'], alpha=0.8, label='Raw Position Features'),
    Patch(facecolor=colors_map['stats'], alpha=0.8, label='Statistical Features'),
    Patch(facecolor=colors_map['trend'], alpha=0.8, label='Trend Features')
]
ax.legend(handles=legend_elements, loc='lower right')

for i, (feat, imp) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
    ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/tutorial_06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: tutorial_06_feature_importance.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎉 ¡TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE!")
print("=" * 70)
print("\n📁 Archivos creados en outputs/:")
print("   1. tutorial_01_fair_vs_biased.png")
print("   2. tutorial_02_mixed_walks.png")
print("   3. tutorial_03_feature_engineering.png")
print("   4. tutorial_04_group_aware_validation.png")
print("   5. tutorial_05_model_comparison.png")
print("   6. tutorial_06_feature_importance.png")
print("\n✨ Listo para commit!")
