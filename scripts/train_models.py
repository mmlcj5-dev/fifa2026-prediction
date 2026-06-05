# =============================================================================
# scripts/train_models.py
# -----------------------------------------------------------------------------
# Retrain all prediction models using the enriched feature set.
#
# Improvements over the original notebook (03_model_training.ipynb):
#   - Time-based train/test split (no data leakage across dates)
#   - Expanded from 7 → 19 features (form + H2H added)
#   - Adds Random Forest as a third model
#   - Adds soft-voting Ensemble across all three models
#   - Cross-validation for more reliable accuracy estimates
#   - Saves feature column list alongside models for consistent inference
#
# Run from project root:
#   python scripts/train_models.py
#
# Output models saved to models/:
#   logistic_regression.pkl, scaler.pkl, xgboost.pkl,
#   random_forest.pkl, ensemble.pkl, feature_cols.pkl
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# All features the retrained models will use
# First 7 match the original model inputs for backward compatibility
FEATURE_COLS = [
    # --- Original ELO + context features ---
    "home_elo",
    "away_elo",
    "elo_diff",
    "is_home_host",
    "is_away_host",
    "is_neutral",
    "is_friendly",
    # --- Rolling form features (last 5 matches) ---
    "home_win_rate",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_points_per_game",
    "away_win_rate",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_points_per_game",
    # --- Head-to-head features (last 5 meetings) ---
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goal_diff",
]

# Time-based split: train on everything before this date, test on after
SPLIT_DATE = "2022-01-01"


# =============================================================================
# SECTION 2 — Load and Prepare Data
# =============================================================================

def load_and_prepare() -> tuple:
    """Load features.csv, build binary flags, split by date, return X/y splits."""
    print(f"\n[1/5] Loading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"      Shape: {df.shape}")

    # Build binary context flags (same logic as original notebook)
    if 'is_home_host' not in df.columns:
        df['is_home_host'] = (df['home_team'] == df.get('country', '')).astype(int)
    if 'is_away_host' not in df.columns:
        df['is_away_host'] = (df['away_team'] == df.get('country', '')).astype(int)
    if 'is_neutral' not in df.columns:
        df['is_neutral'] = df['neutral'].astype(int) if 'neutral' in df.columns else 0
    if 'is_friendly' not in df.columns:
        df['is_friendly'] = (df['tournament'] == 'Friendly').astype(int) if 'tournament' in df.columns else 0

    # Target: 1 = home win
    df['target'] = (df['home_goals'] > df['away_goals']).astype(int)

    # Check which feature columns are actually present
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"      ⚠️  Missing columns (rolling features not yet built?): {missing}")
        print(f"      ➡️  Falling back to available features only: {available}")
    else:
        print(f"      ✅ All {len(FEATURE_COLS)} feature columns present")

    # Time-based split — avoids leakage from future matches into training
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    train_df = df[df['date'] < SPLIT_DATE]
    test_df  = df[df['date'] >= SPLIT_DATE]

    print(f"\n[2/5] Time-based split at {SPLIT_DATE}:")
    print(f"      Train: {len(train_df):,} matches "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"      Test:  {len(test_df):,} matches "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    X_train = train_df[available].fillna(0)
    X_test  = test_df[available].fillna(0)
    y_train = train_df['target']
    y_test  = test_df['target']

    print(f"\n      Class balance (train): "
          f"{y_train.value_counts().to_dict()}")
    print(f"      Class balance (test):  "
          f"{y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, available


# =============================================================================
# SECTION 3 — Train Models
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols):
    """Train LR, XGBoost, Random Forest, and Ensemble. Print metrics. Save all."""

    # --- Scaling (required for Logistic Regression) ---
    print(f"\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # -------------------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------------------
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy: {acc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["Not Home Win", "Home Win"]))
    results['logistic_regression'] = acc_lr

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"Accuracy: {acc_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=["Not Home Win", "Home Win"]))
    results['xgboost'] = acc_xgb

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["Not Home Win", "Home Win"]))
    results['random_forest'] = acc_rf

    # Feature importance from Random Forest
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print("\nTop 10 feature importances (Random Forest):")
    print(importances.sort_values(ascending=False).head(10).to_string())

    # -------------------------------------------------------------------------
    # Soft-Voting Ensemble (LR + XGB + RF)
    # -------------------------------------------------------------------------
    print("\n--- Ensemble (LR + XGB + RF, soft voting) ---")
    # Wrap LR to accept unscaled data by building a pipeline
    from sklearn.pipeline import Pipeline
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])

    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('xgb', xgb),
            ('rf', rf),
        ],
        voting='soft',
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    print(f"Accuracy: {acc_ens:.4f}")
    print(classification_report(y_test, y_pred_ens, target_names=["Not Home Win", "Home Win"]))
    results['ensemble'] = acc_ens

    # -------------------------------------------------------------------------
    # Cross-validation on XGBoost (most reliable single model)
    # -------------------------------------------------------------------------
    print("\n[4/5] Cross-validation (XGBoost, 5-fold, time-aware)...")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    X_all = pd.concat([X_train, X_test]).fillna(0)
    y_all = pd.concat([y_train, y_test])
    cv_scores = cross_val_score(
        XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                      random_state=42, eval_metric='logloss', verbosity=0),
        X_all, y_all, cv=tscv, scoring='accuracy', n_jobs=-1
    )
    print(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return lr, scaler, xgb, rf, ensemble, results


# =============================================================================
# SECTION 4 — Save Models
# =============================================================================

def save_models(lr, scaler, xgb, rf, ensemble, feature_cols, results):
    """Save all models and the feature column list to models/."""
    print(f"\n[5/5] Saving models to {MODELS_DIR}...")

    joblib.dump(lr,        MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler,    MODELS_DIR / "scaler.pkl")
    joblib.dump(xgb,       MODELS_DIR / "xgboost.pkl")
    joblib.dump(rf,        MODELS_DIR / "random_forest.pkl")
    joblib.dump(ensemble,  MODELS_DIR / "ensemble.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    print("\n✅ Saved:")
    for name in ["logistic_regression.pkl", "scaler.pkl", "xgboost.pkl",
                 "random_forest.pkl", "ensemble.pkl", "feature_cols.pkl"]:
        size = (MODELS_DIR / name).stat().st_size / 1024
        print(f"   {name:35s} {size:6.1f} KB")

    print("\n📊 Accuracy Summary:")
    baseline = 0.56  # original XGBoost accuracy
    for model, acc in results.items():
        delta = acc - baseline
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(f"   {model:25s} {acc:.4f}  {arrow} {abs(delta):+.4f} vs baseline")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FIFA 2026 — Model Training Pipeline")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare()
    lr, scaler, xgb, rf, ensemble, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_cols
    )
    save_models(lr, scaler, xgb, rf, ensemble, feature_cols, results)

    print("\n" + "=" * 60)
    print("Done! Next step: update streamlit_app/match_predictor.py")
    print("to load ensemble.pkl and feature_cols.pkl.")
    print("=" * 60)
