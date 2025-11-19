"""
Interactive Tutorial: Step-by-Step ML on Random Walks

This script provides an interactive, commented tutorial that you can run
to learn the concepts while seeing them in action.

Run this script section by section, reading the explanations and observing
the outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
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

print("=" * 70)
print("INTERACTIVE TUTORIAL: Machine Learning on Random Walks")
print("=" * 70)

# ==============================================================================
# SECTION 1: Understanding Random Walks
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Understanding Random Walks")
print("=" * 70)

print("\n📚 CONCEPT:")
print("A random walk is like flipping a coin repeatedly:")
print("  - Heads (+1): Take a step forward")
print("  - Tails (-1): Take a step backward")
print("\nA 'fair' walk has p=0.5 for each. A 'biased' walk has p≠0.5.")

# Generate one fair walk
print("\n🎲 Let's create a FAIR random walk (p=0.5):")
np.random.seed(42)
fair_steps = np.random.choice([-1, 1], size=100, p=[0.5, 0.5])
fair_position = np.cumsum(fair_steps)

print(f"First 20 steps: {fair_steps[:20]}")
print(f"First 20 positions: {fair_position[:20]}")
print(f"Final position: {fair_position[-1]}")
print("👀 Notice: The walk wanders randomly without clear direction")

# Generate one biased walk
print("\n🎲 Now let's create a BIASED random walk (p=0.7):")
biased_steps = np.random.choice([-1, 1], size=100, p=[0.3, 0.7])
biased_position = np.cumsum(biased_steps)

print(f"First 20 steps: {biased_steps[:20]}")
print(f"First 20 positions: {biased_position[:20]}")
print(f"Final position: {biased_position[-1]}")
print("👀 Notice: More +1 than -1, so it trends upward!")

print("\n📊 Visualizing the difference...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(fair_position, label='Fair Walk (p=0.5)')
ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Starting point')
ax1.set_title('Fair Random Walk\n(No bias - pure randomness)')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Position')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(biased_position, color='green', label='Biased Walk (p=0.7)')
ax2.axhline(0, color='red', linestyle='--', alpha=0.5, label='Starting point')
ax2.set_title('Biased Random Walk\n(Tends upward)')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Position')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tutorial_01_walks_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: outputs/tutorial_01_walks_comparison.png")
plt.close()

# input("\n⏸️  Press ENTER to continue to Section 2...")

# ==============================================================================
# SECTION 2: Generating Multiple Walks with Our Framework
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Generating Multiple Walks")
print("=" * 70)

print("\n📚 CONCEPT:")
print("In real applications, we need MANY walks to train ML models.")
print("We'll mix fair and biased walks to create a realistic dataset.")

# Configure mixed walks
bias_dist = BiasDistribution(
    fair_prob=0.3,              # 30% will be fair
    positive_bias_prob=0.35,    # 35% biased upward
    negative_bias_prob=0.35,    # 35% biased downward
    positive_bias_range=(0.6, 0.75),
    negative_bias_range=(0.25, 0.4)
)

config = WalkConfig(
    n_walks=50,
    n_steps=200,
    bias_mode="mixed",
    bias_distribution=bias_dist,
    seed=42
)

print("\n🎲 Generating 50 mixed walks...")
positions, p_ups = generate_random_walks_1d(config)

print(f"\n✅ Generated:")
print(f"  Shape: {positions.shape} (50 walks × 200 steps)")
print(f"  Probability values: {p_ups[:10]}")
print(f"\nDistribution of biases:")
print(f"  Fair (p≈0.5): {np.sum(np.abs(p_ups - 0.5) < 0.05)} walks")
print(f"  Positive bias (p>0.55): {np.sum(p_ups > 0.55)} walks")
print(f"  Negative bias (p<0.45): {np.sum(p_ups < 0.45)} walks")

# Visualize distribution
print("\n📊 Visualizing all walks and their bias distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot walks colored by bias
for i in range(len(positions)):
    if abs(p_ups[i] - 0.5) < 0.05:
        color, alpha = 'gray', 0.3
    elif p_ups[i] > 0.5:
        color, alpha = 'green', 0.5
    else:
        color, alpha = 'red', 0.5
    ax1.plot(positions[i], color=color, alpha=alpha, linewidth=1)

ax1.set_title('50 Mixed Random Walks\n(Gray=Fair, Green=Upward, Red=Downward)')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Position')
ax1.grid(True, alpha=0.3)

# Histogram of p_up values
ax2.hist(p_ups, bins=20, alpha=0.7, edgecolor='black')
ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Fair (p=0.5)')
ax2.set_title('Distribution of Bias Values')
ax2.set_xlabel('Probability of +1 step (p_up)')
ax2.set_ylabel('Number of walks')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tutorial_02_mixed_walks.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: outputs/tutorial_02_mixed_walks.png")
plt.close()

# input("\n⏸️  Press ENTER to continue to Section 3...")

# ==============================================================================
# SECTION 3: Feature Engineering - Extracting Information
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Feature Engineering")
print("=" * 70)

print("\n📚 CONCEPT:")
print("ML models need 'features' (input variables) to make predictions.")
print("We extract features from sliding windows of the walk.")
print("\nExample: From window [10, 11, 9, 10, 12], we extract:")
print("  - Raw deltas: [1, -2, 1, 2]")
print("  - Mean: 0.5 (average step)")
print("  - Std: 1.3 (volatility)")
print("  - Skewness: 0.2 (asymmetry)")

# Create features with different configurations
print("\n🔧 Extracting features with different configurations...")

configs = {
    "Raw deltas only": FeatureConfig(use_raw_deltas=True),
    "Raw + Statistics": FeatureConfig(
        use_raw_deltas=True,
        use_statistics=True,
        statistics=["mean", "std", "skew"]
    ),
    "All features": FeatureConfig(
        use_raw_deltas=True,
        use_statistics=True,
        use_trend=True,
        statistics=["mean", "std", "skew"]
    )
}

for name, feat_config in configs.items():
    X, y, groups = make_windows_from_walks_enhanced(
        positions,
        window=20,
        feature_config=feat_config
    )
    print(f"\n{name}:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Labels (0=fair, 1=biased): {np.bincount(y)}")
    print(f"  First sample features: {X[0][:5]}... (showing first 5)")

# input("\n⏸️  Press ENTER to continue to Section 4...")

# ==============================================================================
# SECTION 4: The Critical Concept - Group-Aware Validation
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Group-Aware Validation (CRITICAL!)")
print("=" * 70)

print("\n📚 CONCEPT:")
print("⚠️  THIS IS THE MOST IMPORTANT CONCEPT IN THE PROJECT!")
print("\nProblem: Sliding windows from the same walk are highly correlated.")
print("If we split randomly, we'll have windows from the same walk in")
print("both train and test sets → DATA LEAKAGE → Inflated performance!")
print("\nSolution: Keep all windows from each walk together in train OR test.")

# Demonstrate with actual data
print("\n🔧 Creating features for demonstration...")
X, y, groups = make_windows_from_walks_enhanced(
    positions,
    window=20,
    feature_config=FeatureConfig(use_raw_deltas=True)
)

print(f"\n✅ Created:")
print(f"  Total samples: {len(X)}")
print(f"  Unique walks (groups): {len(np.unique(groups))}")
print(f"  Samples per walk: ~{len(X) // len(np.unique(groups))}")

# Show group structure
print(f"\nGroup structure (first 200 samples):")
print(f"  Groups: {groups[:200]}")
print("  👀 Notice: Many samples from the same walk (same group ID)")

# Compare split methods
from sklearn.model_selection import train_test_split

print("\n⚠️  Comparing split methods...")

# BAD: Normal split (with leakage)
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get indices for train/test splits
train_indices = list(range(len(X_train_bad)))
test_indices = list(range(len(X_train_bad), len(X)))

groups_train_bad = groups[train_indices]
groups_test_bad = groups[test_indices]

print(f"\n❌ Normal split (BAD):")
print(f"  Unique walks in train: {len(np.unique(groups_train_bad))}")
print(f"  Unique walks in test: {len(np.unique(groups_test_bad))}")
print(f"  ⚠️  LEAKAGE: Walks appear in both train and test!")

# GOOD: Group-aware split
X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
    X, y, groups, test_size=0.2, seed=42
)

train_walks = set(g_train)
test_walks = set(g_test)
overlap = train_walks & test_walks

print(f"\n✅ Group-aware split (GOOD):")
print(f"  Unique walks in train: {len(train_walks)}")
print(f"  Unique walks in test: {len(test_walks)}")
print(f"  Overlap: {len(overlap)} walks")
print(f"  ✅ No leakage! Each walk is entirely in train OR test")

# input("\n⏸️  Press ENTER to continue to Section 5...")

# ==============================================================================
# SECTION 5: Training and Comparing Models
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Training ML Models")
print("=" * 70)

print("\n📚 CONCEPT:")
print("We'll train 4 models and compare their performance:")
print("  1. Dummy (baseline) - always predicts majority class")
print("  2. Logistic Regression - linear model")
print("  3. Random Forest - ensemble of decision trees")
print("  4. Gradient Boosting - iterative error correction")

models_to_try = [
    ("dummy_majority", "Baseline (Dummy)"),
    ("logreg", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("hgb", "Gradient Boosting")
]

results = {}

for model_name, model_label in models_to_try:
    print(f"\n🤖 Training {model_label}...")
    
    model = build_pipeline(model_name)
    
    # Configure HGB to avoid overfitting
    if model_name == "hgb":
        model.set_params(
            clf__max_depth=6,
            clf__learning_rate=0.1,
            clf__min_samples_leaf=20
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    results[model_label] = metrics
    
    print(f"  ✅ Results:")
    print(f"     ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"     Accuracy: {metrics['accuracy']:.3f}")
    print(f"     Precision: {metrics['precision']:.3f}")
    print(f"     Recall: {metrics['recall']:.3f}")
    print(f"     F1-Score: {metrics['f1']:.3f}")

# Visualize comparison
print("\n📊 Creating comparison visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC comparison
model_names = list(results.keys())
roc_aucs = [results[m]['roc_auc'] for m in model_names]
colors = ['red', 'orange', 'lightblue', 'green']

axes[0].barh(model_names, roc_aucs, color=colors, alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
axes[0].set_xlabel('ROC-AUC Score', fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
axes[0].set_xlim(0, 1)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

for i, v in enumerate(roc_aucs):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

# Multiple metrics comparison
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(metrics_names))
width = 0.2

for i, model_name in enumerate(model_names):
    values = [results[model_name][m] for m in metrics_names]
    axes[1].bar(x + i*width, values, width, label=model_name, alpha=0.7)

axes[1].set_xlabel('Metrics', fontweight='bold')
axes[1].set_ylabel('Score', fontweight='bold')
axes[1].set_title('Multiple Metrics Comparison', fontweight='bold', fontsize=12)
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels([m.capitalize() for m in metrics_names])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/tutorial_03_model_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: outputs/tutorial_03_model_comparison.png")
plt.close()

# Best model summary
best_model_name = max(results.keys(), key=lambda m: results[m]['roc_auc'])
best_score = results[best_model_name]['roc_auc']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {best_score:.3f}")
print(f"\n💡 Interpretation:")
if best_score < 0.55:
    print("   ⚠️  Very weak signal. May not be useful.")
elif best_score < 0.65:
    print("   🟨 Weak signal. Acceptable for exploration.")
elif best_score < 0.75:
    print("   🟩 Good signal! Useful for applications.")
else:
    print("   🟩🟩 Excellent signal! High confidence predictions.")

# input("\n⏸️  Press ENTER to continue to Section 6...")

# ==============================================================================
# SECTION 6: Feature Importance Analysis
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Understanding What the Model Learned")
print("=" * 70)

print("\n📚 CONCEPT:")
print("Feature importance tells us which features the model uses most")
print("for predictions. This helps us understand what patterns it learned.")

# Train Random Forest (best for feature importance)
print("\n🔧 Training Random Forest with all features...")
X_full, y_full, groups_full = make_windows_from_walks_enhanced(
    positions,
    window=20,
    feature_config=FeatureConfig(
        use_raw_deltas=True,
        use_statistics=True,
        use_trend=True,
        statistics=["mean", "std", "skew", "range"]
    )
)

X_train_full, X_test_full, y_train_full, y_test_full, _, _ = group_train_test_split(
    X_full, y_full, groups_full, test_size=0.2
)

rf_model = build_pipeline("rf")
rf_model.fit(X_train_full, y_train_full)

# Feature names
feature_names = []
feature_names.extend([f"delta_{i}" for i in range(20)])  # Raw deltas
feature_names.extend(["mean", "std", "skew", "range"])  # Statistics
feature_names.extend(["trend_slope", "trend_correlation"])  # Trend

# Get importance
importance_df = get_feature_importance(rf_model, feature_names)

print("\n🔍 Top 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# Visualize
print("\n📊 Creating feature importance visualization...")
fig, ax = plt.subplots(figsize=(10, 8))

top_15 = importance_df.head(15)
ax.barh(top_15['feature'], top_15['importance'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Top 15 Feature Importance', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

for i, (feat, imp) in enumerate(zip(top_15['feature'], top_15['importance'])):
    ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/tutorial_04_feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: outputs/tutorial_04_feature_importance.png")
plt.close()

print("\n💡 Insights:")
stat_importance = importance_df[importance_df['feature'].isin(
    ['mean', 'std', 'skew', 'range', 'trend_slope', 'trend_correlation']
)]['importance'].sum()
print(f"   Statistical features total importance: {stat_importance:.3f}")
print(f"   👀 Statistical features are {'VERY' if stat_importance > 0.3 else 'moderately'} important!")

# ==============================================================================
# CONCLUSION
# ==============================================================================
print("\n" + "=" * 70)
print("TUTORIAL COMPLETE! 🎉")
print("=" * 70)

print("\n📝 SUMMARY:")
print("\n1. ✅ Random Walks: Understood fair vs biased walks")
print("2. ✅ Data Generation: Created mixed datasets for ML")
print("3. ✅ Feature Engineering: Extracted meaningful features")
print("4. ✅ Group-Aware Validation: Prevented data leakage (CRITICAL!)")
print("5. ✅ Model Training: Compared 4 different models")
print("6. ✅ Feature Importance: Understood what the model learned")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC: {best_score:.3f}")
print(f"\n📊 All visualizations saved in: outputs/")
print("   - tutorial_01_walks_comparison.png")
print("   - tutorial_02_mixed_walks.png")
print("   - tutorial_03_model_comparison.png")
print("   - tutorial_04_feature_importance.png")

print("\n📚 NEXT STEPS:")
print("   1. Read LEARNING_GUIDE.md for deeper explanations")
print("   2. Try the exercises in LEARNING_GUIDE.md")
print("   3. Explore examples/ for real-world applications")
print("   4. Experiment with different parameters!")
print("   5. Run the Jupyter notebook for interactive exploration")

print("\n💡 KEY TAKEAWAY:")
print("   ML can detect hidden bias patterns in sequential data,")
print("   but ONLY if we avoid data leakage with group-aware validation!")

print("\n" + "=" * 70)
print("Happy Learning! 🚀")
print("=" * 70)
