"""
Enhanced model.py — Extended utilities for random walk ML prediction with
parameterized bias sampling, feature engineering, and multi-dimensional support.

Author: You (Physics + ML)
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Literal, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier

RNG = np.random.default_rng


# -------------------------
# Enhanced Configuration
# -------------------------
@dataclass
class BiasDistribution:
    """Configuration for bias sampling in mixed mode."""
    fair_prob: float = 0.5
    positive_bias_prob: float = 0.25
    negative_bias_prob: float = 0.25
    positive_bias_range: Tuple[float, float] = (0.55, 0.7)
    negative_bias_range: Tuple[float, float] = (0.3, 0.45)
    
    def __post_init__(self):
        total = self.fair_prob + self.positive_bias_prob + self.negative_bias_prob
        if not np.isclose(total, 1.0):
            raise ValueError(f"Bias probabilities must sum to 1.0, got {total}")


@dataclass
class WalkConfig:
    n_walks: int = 500
    n_steps: int = 500
    bias_mode: Literal["mixed", "fair", "biased"] = "mixed"
    p_up: Optional[float] = None  # Only used if bias_mode == "biased"
    dimensions: int = 1  # Support for multi-dimensional walks
    bias_distribution: BiasDistribution = field(default_factory=BiasDistribution)
    seed: int = 42


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    use_raw_deltas: bool = True
    use_statistics: bool = False
    use_trend: bool = False
    statistics: List[str] = field(default_factory=lambda: ["mean", "std", "skew"])
    

# -------------------------
# Enhanced Data Generation
# -------------------------
def sample_p_up(bias_mode: str, bias_dist: BiasDistribution, rng: np.random.Generator) -> Union[float, np.ndarray]:
    """
    Sample the probability of stepping +1 according to the parameterized mode.
    For multi-dimensional walks, returns array of probabilities per dimension.
    """
    if bias_mode == "fair":
        return 0.5
    elif bias_mode == "mixed":
        r = rng.random()
        if r < bias_dist.fair_prob:
            return 0.5
        elif r < bias_dist.fair_prob + bias_dist.positive_bias_prob:
            return rng.uniform(*bias_dist.positive_bias_range)
        else:
            return rng.uniform(*bias_dist.negative_bias_range)
    raise ValueError("sample_p_up should not be called with mode 'biased' without p_up.")


def generate_random_walks_nd(cfg: WalkConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate N-dimensional random walks.
    Returns:
        positions: shape (n_walks, n_steps+1, dimensions)
        p_ups: shape (n_walks, dimensions) - probability of +1 step per dimension
    """
    rng = RNG(cfg.seed)
    positions = np.zeros((cfg.n_walks, cfg.n_steps + 1, cfg.dimensions), dtype=np.int32)
    p_ups = np.zeros((cfg.n_walks, cfg.dimensions), dtype=np.float32)

    for i in range(cfg.n_walks):
        for dim in range(cfg.dimensions):
            if cfg.bias_mode == "biased":
                assert cfg.p_up is not None, "Provide p_up when bias_mode='biased'."
                p_up = cfg.p_up
            else:
                p_up = sample_p_up(cfg.bias_mode, cfg.bias_distribution, rng)
            p_ups[i, dim] = p_up

            steps = rng.choice([1, -1], size=cfg.n_steps, p=[p_up, 1 - p_up])
            positions[i, 1:, dim] = np.cumsum(steps)

    return positions, p_ups


def generate_random_walks_1d(cfg: WalkConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Backward compatibility wrapper for 1D walks."""
    cfg_1d = WalkConfig(**{**cfg.__dict__, 'dimensions': 1})
    positions_3d, p_ups_2d = generate_random_walks_nd(cfg_1d)
    return positions_3d[:, :, 0], p_ups_2d[:, 0]


# -------------------------
# Enhanced Feature Engineering
# -------------------------
def compute_window_statistics(deltas: np.ndarray, stats_list: List[str]) -> np.ndarray:
    """
    Compute statistical features for a window of deltas.
    
    Args:
        deltas: shape (window_size,) or (window_size, dimensions)
        stats_list: list of statistics to compute
    
    Returns:
        features: array of statistical features
    """
    features = []
    
    # Handle both 1D and multi-dimensional cases
    if deltas.ndim == 1:
        deltas = deltas.reshape(-1, 1)
    
    for dim in range(deltas.shape[1]):
        dim_deltas = deltas[:, dim]
        for stat in stats_list:
            if stat == "mean":
                features.append(np.mean(dim_deltas))
            elif stat == "std":
                features.append(np.std(dim_deltas))
            elif stat == "skew":
                features.append(stats.skew(dim_deltas) if len(dim_deltas) > 2 else 0.0)
            elif stat == "kurtosis":
                features.append(stats.kurtosis(dim_deltas) if len(dim_deltas) > 3 else 0.0)
            elif stat == "min":
                features.append(np.min(dim_deltas))
            elif stat == "max":
                features.append(np.max(dim_deltas))
            elif stat == "range":
                features.append(np.max(dim_deltas) - np.min(dim_deltas))
            else:
                warnings.warn(f"Unknown statistic: {stat}")
    
    return np.array(features)


def compute_trend_features(deltas: np.ndarray) -> np.ndarray:
    """
    Compute trend features (linear regression slope, correlation with time).
    """
    features = []
    
    if deltas.ndim == 1:
        deltas = deltas.reshape(-1, 1)
    
    time_idx = np.arange(len(deltas))
    
    for dim in range(deltas.shape[1]):
        dim_deltas = deltas[:, dim]
        # Linear trend (slope)
        if len(set(dim_deltas)) > 1:  # Avoid perfect correlation issues
            slope, _, r_value, _, _ = stats.linregress(time_idx, dim_deltas)
            features.extend([slope, r_value])
        else:
            features.extend([0.0, 0.0])
    
    return np.array(features)


def make_windows_from_walks_enhanced(
    positions: np.ndarray,
    window: int = 20,
    horizon: int = 1,
    feature_config: FeatureConfig = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhanced version with configurable feature engineering.
    
    Args:
        positions: shape (n_walks, n_steps+1) for 1D or (n_walks, n_steps+1, dimensions) for ND
        window: lookback window size
        horizon: prediction horizon (always 1 for now)
        feature_config: configuration for feature engineering
    
    Returns:
        X: enhanced feature matrix
        y: target labels (for 1D: binary; for ND: could be extended)
        groups: group identifiers
    """
    if feature_config is None:
        feature_config = FeatureConfig()
    
    # Handle both 1D and multi-dimensional cases
    if positions.ndim == 2:
        positions = positions[:, :, np.newaxis]  # Add dimension axis
    
    deltas = np.diff(positions, axis=1)  # shape (n_walks, n_steps, dimensions)
    n_walks, n_steps, n_dims = deltas.shape

    X_list, y_list, g_list = [], [], []

    for i in range(n_walks):
        for t in range(window, n_steps):
            # Extract window of deltas
            window_deltas = deltas[i, t - window:t, :]  # shape (window, dimensions)
            
            # Build features
            features = []
            
            # Raw deltas (flattened if multi-dimensional)
            if feature_config.use_raw_deltas:
                features.extend(window_deltas.flatten())
            
            # Statistical features
            if feature_config.use_statistics:
                stat_features = compute_window_statistics(window_deltas, feature_config.statistics)
                features.extend(stat_features)
            
            # Trend features
            if feature_config.use_trend:
                trend_features = compute_trend_features(window_deltas)
                features.extend(trend_features)
            
            # Target (for now, just use first dimension and binary classification)
            target = 1 if deltas[i, t, 0] == 1 else 0
            
            X_list.append(features)
            y_list.append(target)
            g_list.append(i)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    groups = np.array(g_list, dtype=np.int32)
    
    return X, y, groups


# Backward compatibility
def make_windows_from_walks(positions: np.ndarray, window: int = 20, horizon: int = 1):
    """Backward compatible version."""
    return make_windows_from_walks_enhanced(
        positions, window, horizon, FeatureConfig(use_raw_deltas=True)
    )


# -------------------------
# Enhanced Modeling with Baselines
# -------------------------
def build_pipeline(model: Literal["logreg", "rf", "hgb", "dummy_majority", "dummy_stratified"] = "logreg") -> Pipeline:
    """
    Build a scikit-learn pipeline with baseline options.
    """
    if model == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ])
    elif model == "rf":
        return Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
        ])
    elif model == "hgb":
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ])
    elif model == "dummy_majority":
        return Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent", random_state=42)),
        ])
    elif model == "dummy_stratified":
        return Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=42)),
        ])
    else:
        raise ValueError("Unknown model. Choose from {'logreg','rf','hgb','dummy_majority','dummy_stratified'}.")


def default_param_grid(model: str) -> Dict[str, List]:
    """Extended parameter grids including dummy classifiers."""
    if model == "logreg":
        return {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],  # Works with both L1 and L2
        }
    elif model == "rf":
        return {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", "log2"],
        }
    elif model == "hgb":
        return {
            "clf__max_depth": [None, 3, 6, 10],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_leaf_nodes": [15, 31, 63],
        }
    elif model in ["dummy_majority", "dummy_stratified"]:
        return {}  # No parameters to tune for dummy classifiers
    else:
        raise ValueError("Unknown model for param grid.")


# -------------------------
# Window Size Analysis
# -------------------------
def analyze_window_sizes(
    positions: np.ndarray,
    window_sizes: List[int],
    model_name: str = "hgb",
    test_size: float = 0.2,
    cv_splits: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Analyze how different window sizes affect model performance.
    
    Returns:
        DataFrame with columns: window_size, cv_score_mean, cv_score_std, test_roc_auc, test_accuracy
    """
    results = []
    
    for window in window_sizes:
        # Create dataset
        X, y, groups = make_windows_from_walks(positions, window=window)
        
        if len(X) == 0:
            continue
            
        # Split data
        X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
            X, y, groups, test_size=test_size, seed=seed
        )
        
        # Train model
        pipe = build_pipeline(model_name)
        param_grid = default_param_grid(model_name)
        
        if param_grid:  # Only if there are parameters to tune
            gs = tune_with_cv(pipe, X_train, y_train, g_train, param_grid, cv_splits=cv_splits)
            cv_score_mean = gs.best_score_
            cv_score_std = gs.cv_results_['std_test_score'][gs.best_index_]
            best_model = gs.best_estimator_
        else:
            # For dummy classifiers
            pipe.fit(X_train, y_train)
            cv_score_mean = np.nan
            cv_score_std = np.nan
            best_model = pipe
        
        # Test evaluation
        test_metrics = evaluate(best_model, X_test, y_test)
        
        results.append({
            'window_size': window,
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'test_roc_auc': test_metrics['roc_auc'],
            'test_accuracy': test_metrics['accuracy'],
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
    
    return pd.DataFrame(results)


# -------------------------
# Real-world Applications Framework
# -------------------------
@dataclass
class ApplicationConfig:
    """Configuration for real-world applications."""
    name: str
    description: str
    expected_bias_strength: float  # How strong bias signal is expected to be
    noise_level: float = 0.0  # Additional noise to add
    trend_component: bool = False  # Whether to add trend
    seasonal_component: bool = False  # Whether to add seasonality


def simulate_financial_returns(
    n_series: int = 100,
    n_periods: int = 500,
    base_volatility: float = 0.02,
    bias_strength: float = 0.001,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate financial return series with hidden bias (momentum/reversion).
    
    Returns:
        returns: shape (n_series, n_periods)
        bias_types: array indicating bias type per series
    """
    rng = RNG(seed)
    returns = np.zeros((n_series, n_periods))
    bias_types = rng.choice(['none', 'momentum', 'reversion'], size=n_series, p=[0.4, 0.3, 0.3])
    
    for i in range(n_series):
        # Base random returns
        series_returns = rng.normal(0, base_volatility, n_periods)
        
        # Add bias based on type
        if bias_types[i] == 'momentum':
            # Positive autocorrelation
            for t in range(1, n_periods):
                series_returns[t] += bias_strength * series_returns[t-1]
        elif bias_types[i] == 'reversion':
            # Negative autocorrelation (mean reversion)
            for t in range(1, n_periods):
                series_returns[t] -= bias_strength * series_returns[t-1]
        
        returns[i] = series_returns
    
    return returns, bias_types


def simulate_sensor_drift(
    n_sensors: int = 50,
    n_measurements: int = 1000,
    base_noise: float = 0.1,
    drift_probability: float = 0.3,
    drift_strength: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sensor measurements with potential drift.
    
    Returns:
        measurements: shape (n_sensors, n_measurements)
        has_drift: boolean array indicating which sensors have drift
    """
    rng = RNG(seed)
    measurements = np.zeros((n_sensors, n_measurements))
    has_drift = rng.random(n_sensors) < drift_probability
    
    for i in range(n_sensors):
        # Base measurements with noise
        base_signal = rng.normal(0, base_noise, n_measurements)
        
        if has_drift[i]:
            # Add gradual drift
            drift = np.linspace(0, drift_strength * n_measurements, n_measurements)
            drift += rng.normal(0, drift_strength * 0.1, n_measurements)  # Noisy drift
            measurements[i] = base_signal + drift
        else:
            measurements[i] = base_signal
    
    return measurements, has_drift


# Keep all the original functions for backward compatibility
# [Previous functions remain unchanged: group_train_test_split, tune_with_cv, evaluate]

def group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Group-aware split: ensures windows from the same walk do not leak across splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], groups[train_idx], groups[test_idx]


def tune_with_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: Dict[str, List],
    cv_splits: int = 5,
    n_jobs: int = -1,
    scoring: str = "roc_auc",
    search_type: str = "grid"  # New parameter: "grid" or "random"
) -> GridSearchCV:
    """
    Enhanced grid-search with GroupKFold and option for RandomizedSearch.
    
    Args:
        search_type: "grid" for exhaustive search, "random" for RandomizedSearchCV
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    gkf = GroupKFold(n_splits=cv_splits)
    
    if search_type == "random":
        # Use RandomizedSearchCV for faster performance
        n_iter = min(20, np.prod([len(v) for v in param_grid.values()]))  # Don't exceed total combinations
        gs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=gkf.split(X, y, groups),
            n_jobs=n_jobs,
            verbose=0,
            refit=True,
            random_state=42,
            return_train_score=True,
        )
    else:
        # Original GridSearchCV
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=gkf.split(X, y, groups),
            n_jobs=n_jobs,
            verbose=0,
            refit=True,
            return_train_score=True,
        )
    
    gs.fit(X, y)
    return gs


def fast_baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    groups_train: np.ndarray = None,
    models: List[str] = None,
    use_grid_search: bool = False
) -> Dict[str, Dict]:
    """
    Fast baseline comparison optimized for quick results.
    
    Args:
        models: List of model names to compare. If None, uses ["hgb"] for speed
        use_grid_search: If False, uses good default parameters
    
    Returns:
        Dictionary with model results
    """
    if models is None:
        models = ["hgb"]  # Single best model for speed
    
    results = OrderedDict()
    
    for name in models:
        pipe = build_pipeline(name)
        param_grid = default_param_grid(name)
        
        # Fast mode: Use good defaults instead of grid search
        if not use_grid_search or not param_grid:
            if name == "hgb":
                pipe.set_params(clf__max_depth=6, clf__learning_rate=0.1, clf__max_leaf_nodes=31)
            elif name == "rf":
                pipe.set_params(clf__n_estimators=100, clf__max_depth=10, clf__min_samples_leaf=3)
            elif name == "logreg":
                pipe.set_params(clf__C=1.0, clf__penalty='l2')
            
            pipe.fit(X_train, y_train)
            best_model = pipe
            cv_score = np.nan
            
        # Full mode: Use randomized search for efficiency
        else:
            if param_grid and groups_train is not None:
                gs = tune_with_cv(
                    pipe, X_train, y_train, groups_train, 
                    param_grid, cv_splits=3, search_type="random"
                )
                best_model = gs.best_estimator_
                cv_score = gs.best_score_
            else:
                pipe.fit(X_train, y_train)
                best_model = pipe
                cv_score = np.nan
        
        test_metrics = evaluate(best_model, X_test, y_test)
        results[name] = {**test_metrics, "cv_roc_auc": cv_score}
    
    return results


def evaluate(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray):
    """Compute a suite of classification metrics."""
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        decision = model.decision_function(X_test)
        y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)

    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = float("nan")
    try:
        ap = average_precision_score(y_test, y_proba)
    except ValueError:
        ap = float("nan")
    try:
        brier = brier_score_loss(y_test, y_proba)
    except ValueError:
        brier = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "avg_precision": float(ap),
        "brier": float(brier),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
