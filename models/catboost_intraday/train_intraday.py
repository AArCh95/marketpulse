#!/usr/bin/env python3
# D:\AARCH\models\catboost_intraday\train.py
# Train CatBoost model for 5-minute forward return prediction

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier, Pool

# Import from local config
from config_intraday import (
    MODEL_PATH, FEATURE_ORDER_PATH, FEATURE_COLUMNS, CAT_FEATURES,
    UP_THRESH, DOWN_THRESH, CATBOOST_PARAMS, CLASS_NAMES, HORIZON_MIN
)
from db_intraday import get_conn, fetch_training_frame_5m

# Ensure model directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Convert continuous forward returns to 3-class labels.
    
    fwd_ret_5m > UP_THRESH   → 'up'    (2)
    fwd_ret_5m < DOWN_THRESH → 'down'  (0)
    else                      → 'flat'  (1)
    """
    fwd_col = f"fwd_ret_{HORIZON_MIN}m"
    
    if fwd_col not in df.columns:
        raise ValueError(f"Missing forward return column: {fwd_col}")
    
    labels = pd.Series("flat", index=df.index)
    labels[df[fwd_col] > UP_THRESH] = "up"
    labels[df[fwd_col] < DOWN_THRESH] = "down"
    
    # Map to integers for CatBoost
    label_map = {"down": 0, "flat": 1, "up": 2}
    return labels.map(label_map)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all features in FEATURE_COLUMNS exist, fill missing with sensible defaults.
    """
    X = df.copy()
    
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            # Fill missing features with neutral values
            if col in CAT_FEATURES:
                X[col] = "unknown"
            else:
                X[col] = 0.0
    
    # Convert categoricals to string
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)
    
    # Select only model features (drop ts, session, etc.)
    return X[FEATURE_COLUMNS]


def main():
    print(f"[{datetime.now()}] Starting 5-minute CatBoost training...")
    print(f"Horizon: {HORIZON_MIN} minutes")
    print(f"Thresholds: up={UP_THRESH:.4f}, down={DOWN_THRESH:.4f}")
    
    # Step 1: Fetch training data
    print("\n[1/5] Fetching training data from DuckDB...")
    with get_conn() as conn:
        df = fetch_training_frame_5m(conn)
    
    if df.empty:
        print("ERROR: No training data returned. Check DB and filters.")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(df):,} bars across {df['symbol'].nunique()} symbols")
    print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")
    
    # Step 2: Create labels
    print("\n[2/5] Creating labels...")
    y = create_labels(df)
    
    # Check class distribution
    class_counts = y.value_counts().sort_index()
    print("  Class distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_id, 0)
        pct = 100 * count / len(y) if len(y) > 0 else 0
        print(f"    {name:5s} ({cls_id}): {count:8,} ({pct:5.2f}%)")
    
    # Drop if any class is too small
    if (class_counts < 100).any():
        print("  WARNING: Some classes have <100 samples. Model may be unstable.")
    
    # Step 3: Prepare features
    print("\n[3/5] Preparing features...")
    X = prepare_features(df)
    
    # Drop rows with NaN in critical features
    valid_mask = y.notna()  # Keep rows even if features are NaN
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    print(f"  ✓ Feature matrix: {X_clean.shape}")
    print(f"  Dropped {(~valid_mask).sum():,} rows with missing values")
    print(f"  Final training size: {len(X_clean):,}")
    
    if len(X_clean) < 1000:
        print("ERROR: Too few training samples after cleaning.")
        sys.exit(1)
    
    # Step 4: Train/validation split (80/20 time-based)
    print("\n[4/5] Splitting train/validation...")
    split_idx = int(0.8 * len(X_clean))
    
    X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # Identify categorical feature indices
    cat_idx = [X_train.columns.get_loc(c) for c in CAT_FEATURES if c in X_train.columns]
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    
    # Step 5: Train model
    print("\n[5/5] Training CatBoost...")
    print(f"  Params: {CATBOOST_PARAMS}")
    
    # Check GPU availability and adjust if needed
    params = CATBOOST_PARAMS.copy()
    try:
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False,
        )
    except Exception as e:
        if "GPU" in str(e):
            print("  GPU training failed, falling back to CPU...")
            params["task_type"] = "CPU"
            params.pop("devices", None)
            model = CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                plot=False,
            )
        else:
            raise
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    y_pred = model.predict(val_pool)
    y_proba = model.predict_proba(val_pool)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_val, y_pred)
    
    print("\nConfusion Matrix:")
    print("         ", "  ".join(f"{c:>6s}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>6s}   ", "  ".join(f"{x:6d}" for x in row))
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, digits=3))
    
    # Feature importance (top 15)
    print("\nTop 15 Feature Importances:")
    feat_imp = model.get_feature_importance()
    feat_names = X_train.columns
    
    importances = sorted(zip(feat_names, feat_imp), key=lambda x: x[1], reverse=True)[:15]
    for rank, (name, imp) in enumerate(importances, 1):
        print(f"  {rank:2d}. {name:25s} {imp:8.2f}")
    
    # Save model
    print(f"\n[SAVE] Saving model to {MODEL_PATH}...")
    model.save_model(str(MODEL_PATH))
    
    # Save feature order
    print(f"[SAVE] Saving feature order to {FEATURE_ORDER_PATH}...")
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(list(FEATURE_COLUMNS), f, indent=2)
    
    # Save metadata
    meta_path = MODEL_PATH.parent / "training_metadata.json"
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "horizon_minutes": HORIZON_MIN,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "symbols": int(df["symbol"].nunique()),
        "date_range": [str(df["ts"].min()), str(df["ts"].max())],
        "class_distribution": {
            CLASS_NAMES[i]: int(class_counts.get(i, 0))
            for i in range(len(CLASS_NAMES))
        },
        "thresholds": {"up": UP_THRESH, "down": DOWN_THRESH},
        "feature_columns": FEATURE_COLUMNS,
        "catboost_params": params,
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SAVE] Metadata saved to {meta_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Use this model for 5-minute forward prediction inference.")


if __name__ == "__main__":
    main()
