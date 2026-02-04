# =====================================================================
# Notebook: Retrain CatBoost on 5-minute labels using Database.xlsx
# - No changes to your project structure: imports config & features
# - Uses the same FEATURE_COLUMNS / CAT_FEATURES / CATBOOST_PARAMS
# - Saves model to config.MODEL_PATH and feature order JSON
# =====================================================================

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, classification_report, classification_report, confusion_matrix


# === Project config & helpers (reuse existing) ===
from config import (
    CATBOOST_PARAMS,
    MODEL_PATH,
    FEATURE_ORDER_PATH,
    FEATURE_COLUMNS,
    CAT_FEATURES,
    CLASS_NAMES
)
from features import save_feature_order

# --------------------------------------------------------------------------------
# Knobs (safe defaults for 5-minute horizon; adjust if you want to sweep later)
# --------------------------------------------------------------------------------
EXCEL_PATH = "D:\AARCH\models\catboost_core\dataset.csv"  # the attached file (input+output per row)
UP_THRESH = 0.04             # +0.20% => label 'up'
DOWN_THRESH = -0.04          # -0.20% => label 'down'
MARGIN_CUTOFF = 0.15          # serve-style gate: margin < cutoff => flat
PROB_GAP = 0.10               # serve-style gate: |p_up - p_down| < gap => flat
VALID_SPLIT_FRAC = 0.2        # last 20% of unique dates for validation

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def make_tilted_class_weights(y: pd.Series, tilt_down: float = 1.30, normalize: bool = True) -> list[float]:
    counts = Counter(y.tolist())
    total = sum(counts.values())
    classes = sorted(counts.keys())
    wmap = {cls: total / (len(classes) * counts[cls]) for cls in classes}  # inverse freq
    w = np.array([wmap.get(0, 1.0), wmap.get(1, 1.0), wmap.get(2, 1.0)], dtype=float)
    w[0] *= float(tilt_down)  # tilt DOWN
    if normalize:
        w = w * (len(w) / w.sum())  # keep mean ≈ 1
    return list(w)

def train_eval_with_tilt(tilt: float, iters: int = 500):
    cw = make_tilted_class_weights(y_train, tilt_down=tilt, normalize=True)
    p = dict(CATBOOST_PARAMS)
    p["class_weights"] = cw
    p["iterations"] = min(p.get("iterations", iters), iters)

    m = CatBoostClassifier(**p)
    m.fit(Pool(X_train, y_train, cat_features=cat_idx),
          eval_set=Pool(X_valid, y_valid, cat_features=cat_idx),
          use_best_model=True, verbose=False)

    probs = m.predict_proba(Pool(X_valid, y_valid, cat_features=cat_idx))
    pred = np.argmax(probs, axis=1)

    # apply serve rules
    sp = np.sort(probs, axis=1)
    margins = sp[:, -1] - sp[:, -2]
    pred_adj = pred.copy()
    for i in range(len(pred_adj)):
        if margins[i] < MARGIN_CUTOFF or abs(probs[i,2] - probs[i,0]) < PROB_GAP:
            pred_adj[i] = 1

    rep = classification_report(y_valid, pred_adj, labels=[0,1,2],
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    acc = (pred_adj == y_valid.values).mean()
    cm = confusion_matrix(y_valid, pred_adj, labels=[0,1,2])

    return {
        "tilt": tilt,
        "weights": cw,
        "acc": acc,
        "recall_down": rep["down"]["recall"],
        "f1_down": rep["down"]["f1-score"],
        "recall_up": rep["up"]["recall"],
        "f1_up": rep["up"]["f1-score"],
        "f1_flat": rep["flat"]["f1-score"],
        "cm": cm,
    }

def make_tilted_class_weights(y: pd.Series, tilt_down: float = 1.25, normalize: bool = True) -> list[float]:
    """
    Start from inverse-frequency weights; multiply DOWN (class 0) by tilt_down.
    Optionally renormalize so average weight ≈ 1 for stable loss scale.
    Returns weights ordered as [down(0), flat(1), up(2)].
    """
    counts = Counter(y.tolist())
    total = sum(counts.values())
    classes = sorted(counts.keys())
    # inverse frequency baseline
    wmap = {cls: total / (len(classes) * counts[cls]) for cls in classes}
    w = np.array([wmap.get(0, 1.0), wmap.get(1, 1.0), wmap.get(2, 1.0)], dtype=float)

    # tilt DOWN
    w[0] *= float(tilt_down)

    if normalize:
        w = w * (len(w) / w.sum())  # mean ≈ 1

    return list(w)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure all expected columns exist; create missing numeric ones as NaN."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _prep_categoricals_for_catboost(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast categoricals to string and fill missing with sentinel that matches serve-side."""
    if not cat_cols:
        return df
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype("string").fillna("__NA__")
    return df

def robust_time_split(df: pd.DataFrame, ts_col: str = "ts", date_col: str = "date", valid_frac: float = 0.2):
    """
    Create non-empty train/valid masks with time ordering.
    1) Try by unique dates (session-wise split).
    2) Fallback: by timestamp quantile split.
    3) Last resort: split by row index (time-sorted).
    """
    assert valid_frac > 0 and valid_frac < 1, "valid_frac must be in (0,1)"
    df = df.copy()

    # Ensure date column exists (UTC date from ts)
    if date_col not in df.columns:
        df[date_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.date

    # --- Attempt 1: by unique dates ---
    dates = sorted([d for d in df[date_col].unique() if pd.notna(d)])
    if len(dates) >= 2:
        cut = int(len(dates) * (1 - valid_frac))
        cut = min(max(cut, 1), len(dates) - 1)  # keep both sides non-empty
        train_dates = set(dates[:cut])
        valid_dates = set(dates[cut:])
        train_mask = df[date_col].isin(train_dates)
        valid_mask = df[date_col].isin(valid_dates)
        if train_mask.any() and valid_mask.any():
            return train_mask, valid_mask

    # --- Attempt 2: by timestamp quantile ---
    df = df.sort_values(ts_col)
    n = len(df)
    q_idx = int(n * (1 - valid_frac))
    q_idx = min(max(q_idx, 1), n - 1)
    cut_ts = df.iloc[q_idx][ts_col]
    train_mask = df[ts_col] < cut_ts
    valid_mask = df[ts_col] >= cut_ts
    if train_mask.any() and valid_mask.any():
        return train_mask, valid_mask

    # --- Attempt 3: by row index (time-sorted) ---
    train_mask = pd.Series(False, index=df.index)
    valid_mask = pd.Series(False, index=df.index)
    train_mask.iloc[:q_idx] = True
    valid_mask.iloc[q_idx:] = True
    return train_mask, valid_mask


def _to_float(series: pd.Series) -> pd.Series:
    """Robust float parser for columns that may contain comma decimals."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.replace(",", ".", regex=False)
    # handle empty strings gracefully
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")

def _compute_class_weights(y: pd.Series) -> list[float]:
    """Inverse-frequency class weights for classes 0,1,2."""
    counts = Counter(y.tolist())
    total = sum(counts.values())
    classes = sorted(counts.keys())
    wmap = {cls: (total / (len(classes) * counts[cls])) for cls in classes}
    return [float(wmap.get(0, 1.0)), float(wmap.get(1, 1.0)), float(wmap.get(2, 1.0))]

def _time_split(df: pd.DataFrame, date_col: str = "date", valid_frac: float = 0.2):
    dates = sorted(df[date_col].unique())
    cut = int(len(dates) * (1 - valid_frac))
    train_dates = set(dates[:cut])
    valid_dates = set(dates[cut:])
    train_mask = df[date_col].isin(train_dates)
    valid_mask = df[date_col].isin(valid_dates)
    return train_mask, valid_mask

# --------------------------------------------------------------------------------
# 1) Load the attached Excel and standardize fields
# --------------------------------------------------------------------------------
df = pd.read_csv(EXCEL_PATH)

# Parse timestamps & sort
df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

# Ensure numeric features are floats (comma decimals -> dot)
numeric_like = set(FEATURE_COLUMNS) - set(CAT_FEATURES)
for col in numeric_like:
    if col in df.columns:
        df[col] = _to_float(df[col])

# We'll compute 5-minute forward return from 'close_d'
if "close_d" not in df.columns:
    raise KeyError("Expected 'close_d' column in Database.xlsx to compute fwd_ret_5m")

df["close_d"] = _to_float(df["close_d"])

# --------------------------------------------------------------------------------
# 2) Compute 5-minute forward return & labels
# --------------------------------------------------------------------------------
df["close_fwd_5d"] = df.groupby("symbol")["close_d"].shift(-5)
df["fwd_ret_5d"] = df["close_fwd_5d"] / df["close_d"] - 1.0


# Label: 0=down, 1=flat, 2=up
df["y"] = np.where(df["fwd_ret_5m"] <= DOWN_THRESH, 0,
            np.where(df["fwd_ret_5m"] >= UP_THRESH, 2, 1))

# Remove rows without label (tail of the day)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)

# A convenience date column for time-based split
df["date"] = df["ts"].dt.tz_convert("America/Costa_Rica").dt.date

print("Label distribution:", df["y"].value_counts().to_dict())

# --------------------------------------------------------------------------------
# 3) Build X (features), prep categoricals, cat indices
# --------------------------------------------------------------------------------
df = _ensure_columns(df, FEATURE_COLUMNS)
X_all = df[FEATURE_COLUMNS].copy()

# Prepare categoricals
X_all = _prep_categoricals_for_catboost(X_all, [c for c in CAT_FEATURES if c in FEATURE_COLUMNS])
cat_idx = [FEATURE_COLUMNS.index(c) for c in CAT_FEATURES if c in FEATURE_COLUMNS]

# --------------------------------------------------------------------------------
# 4) Time-based split (by date) to avoid leakage
# --------------------------------------------------------------------------------
train_mask, valid_mask = robust_time_split(df, ts_col="ts", date_col="date", valid_frac=VALID_SPLIT_FRAC)

X_train, y_train = X_all.loc[train_mask], df.loc[train_mask, "y"]
X_valid, y_valid = X_all.loc[valid_mask], df.loc[valid_mask, "y"]

print(f"Train rows: {len(X_train)}  |  Valid rows: {len(X_valid)}")

# --------------------------------------------------------------------------------
# 5) Class weights for imbalance
# --------------------------------------------------------------------------------

TILT_DOWN = 1.30       # try 1.10–1.35
NORMALIZE_WEIGHTS = True

class_weights = make_tilted_class_weights(y_train, tilt_down=TILT_DOWN, normalize=True)
print("Class weights (tilted):", class_weights)


# --------------------------------------------------------------------------------
# 6) Train CatBoost using your existing config
# --------------------------------------------------------------------------------
params = dict(CATBOOST_PARAMS)
params["class_weights"] = class_weights
params.setdefault("loss_function", "MultiClass")
params.setdefault("eval_metric", "TotalF1")
params.setdefault("od_type", "Iter")
params.setdefault("od_wait", 100)
params.setdefault("random_seed", 42)

train_pool = Pool(X_train, label=y_train, cat_features=cat_idx)
use_valid = len(X_valid) > 0
valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_idx) if use_valid else None

model = CatBoostClassifier(**params)
model.fit(
    train_pool,
    eval_set=valid_pool if use_valid else None,
    use_best_model=use_valid
)

# --------------------------------------------------------------------------------
# 7) Save model + exact inference feature order (unchanged integration)
# --------------------------------------------------------------------------------
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
model.save_model(MODEL_PATH)
save_feature_order(FEATURE_ORDER_PATH, order=FEATURE_COLUMNS)
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved feature order to: {FEATURE_ORDER_PATH}")


# --------------------------------------------------------------------------------
# 8) Evaluate (raw argmax vs serve-style adjusted)
# --------------------------------------------------------------------------------
if use_valid:
    probs_valid = model.predict_proba(valid_pool)
    pred_argmax = np.argmax(probs_valid, axis=1)
    sorted_probs = np.sort(probs_valid, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    # serve-style adjustment
    pred_adj = pred_argmax.copy()
    for i in range(len(pred_adj)):
        if margins[i] < MARGIN_CUTOFF:
            pred_adj[i] = 1  # flat
        elif abs(probs_valid[i, 2] - probs_valid[i, 0]) < PROB_GAP:
            pred_adj[i] = 1  # flat

    print("\n=== RAW ARGMAX ===")
    print("Confusion matrix [down,flat,up]:")
    print(confusion_matrix(y_valid, pred_argmax, labels=[0, 1, 2]))
    print("Classification report:")
    print(classification_report(y_valid, pred_argmax, labels=[0, 1, 2], target_names=CLASS_NAMES))

    print("\n=== ADJUSTED (margin + prob gap) ===")
    print("Confusion matrix [down,flat,up]:")
    print(confusion_matrix(y_valid, pred_adj, labels=[0, 1, 2]))
    print("Classification report:")
    print(classification_report(y_valid, pred_adj, labels=[0, 1, 2], target_names=CLASS_NAMES))
else:
    print("\n(No validation set available after robust split — model trained on all data.)")

# --------------------------------------------------------------------------------
# 9) (Optional) quick sweep of margin cutoffs to inspect the trade-off
# --------------------------------------------------------------------------------
def evaluate_adjusted(y_true: np.ndarray, probs: np.ndarray, margin_cutoff: float, prob_gap: float):
    pred = np.argmax(probs, axis=1)
    sorted_p = np.sort(probs, axis=1)
    margins_local = sorted_p[:, -1] - sorted_p[:, -2]
    for i in range(len(pred)):
        if margins_local[i] < margin_cutoff:
            pred[i] = 1
        elif abs(probs[i, 2] - probs[i, 0]) < prob_gap:
            pred[i] = 1
    acc = (pred == y_true).mean()
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    return acc, cm

print("\n=== Sweep (margin, fixed prob_gap=0.10) ===")
for mc in [0.10, 0.13, 0.15, 0.20]:
    acc_s, _cm = evaluate_adjusted(y_valid.values, probs_valid, margin_cutoff=mc, prob_gap=PROB_GAP)
    print(f"margin={mc:.2f} -> acc={acc_s:.3%}")



tilts = [1.00, 1.10, 1.20, 1.25, 1.30, 1.35]
res = [train_eval_with_tilt(t) for t in tilts]
tbl = pd.DataFrame([{
    "tilt": r["tilt"], "acc": round(r["acc"],4),
    "recall_down": round(r["recall_down"],3), "f1_down": round(r["f1_down"],3),
    "recall_up": round(r["recall_up"],3), "f1_up": round(r["f1_up"],3),
    "f1_flat": round(r["f1_flat"],3), "weights": r["weights"],
} for r in res]).sort_values(["f1_down","acc"], ascending=False)
print(tbl.to_string(index=False))

best = max(res, key=lambda r: (r["f1_down"], r["acc"]))
print("\nBest tilt:", best["tilt"], "weights:", best["weights"], "acc:", f"{best['acc']:.3%}")
print("Confusion matrix [down, flat, up]:\n", best["cm"])