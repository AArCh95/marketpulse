import os, json, time, math, duckdb, threading
import numpy as np
import pandas as pd

from fastapi import Query, FastAPI, HTTPException, Request, Header, Depends
from catboost import CatBoostClassifier, Pool  # type: ignore
from typing import Any, Dict, List
from dotenv import load_dotenv
from pathlib import Path

from config import (MODEL_PATH, FEATURE_ORDER_PATH, 
                    FEATURE_COLUMNS, CAT_FEATURES, CLASS_NAMES,
                    MARGIN_CUTOFF, PROB_GAP, SHORT_MIN_PROB)

from intraday_features import latest_snapshot_features, resolve_db_path
from schemas import InferBatch
from features import rows_to_frame

# -------------------------------------------------------------------
# Config & globals
# -------------------------------------------------------------------
load_dotenv(dotenv_path=(Path(__file__).resolve().parent / ".env"))

# Always resolve the active DB via the snapshot pointer first
DB_PATH = resolve_db_path()
SIGNALS_DB = os.getenv("SIGNALS_DB", r"D:\AARCH\DBs\signals.duckdb")
WRITE_SIGNALS = os.getenv("WRITE_SIGNALS", "1") == "1"

API_VERSION = os.getenv("API_VERSION", "1.0.0")
REQUIRED_API_KEY = os.getenv("API_KEY")  # None means "no auth" (local dev)

# Intraday knobs (kept for metrics; enrichment now reads snapshot only)
# Bias the model to be more sensitive to DOWN (class 0)
TILT_DOWN = 1.25   # try 1.15–1.35; you’ll sweep below
NORMALIZE_WEIGHTS = True  # keep average class weight ≈ 1 (stability)

INTRADAY_ENRICH = True
INTRADAY_MINUTES = int(os.getenv("INTRADAY_MINUTES", "5"))
INTRADAY_OVERWRITE = os.getenv("INTRADAY_OVERWRITE", "0") == "1"

_sig_lock = threading.Lock()

# Columns we will backfill from snapshot
MAT_COLS = [
    "rsi_14","macd_line","macd_signal","macd_hist","dist_vwap_bps",
    "close_d","mkt_ret_1d","sector_ret_1d","rel_sector_vs_mkt",
    "rv_30m","dist_to_high_bps","dist_to_low_bps","minute_of_day_pct","vol_zscore_today",
]

app = FastAPI(title="CatBoost Core (MarketPulse)", version=API_VERSION)
_model = None
_feature_order = None
_cat_idx = None


# For per-row fills/metrics
INTRADAY_COLS = [
    "rsi_14","macd_line","macd_signal","macd_hist","dist_vwap_bps",
    "mkt_ret_1d","sector_ret_1d","rel_sector_vs_mkt","close_d",
    "rv_30m","dist_to_high_bps","dist_to_low_bps","minute_of_day_pct","vol_zscore_today",
]

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

LABELS = np.array(["down","flat","up"])

def decide_from_probs(p_down: float, p_flat: float, p_up: float):
    probs = np.array([p_down, p_flat, p_up], dtype=float)
    # margin = top1 - top2
    top2 = np.partition(probs, -2)[-2:]
    margin = float(top2.max() - top2.min())
    pred_idx = int(np.argmax(probs))

    # 1) margin gate
    if margin < MARGIN_CUTOFF:
        return "flat", margin

    # 2) probability gap between up & down
    if abs(probs[2] - probs[0]) < PROB_GAP:
        return "flat", margin

    # 3) asymmetric guard for shorts (optional)
    if pred_idx == 0 and probs[0] < SHORT_MIN_PROB:
        return "flat", margin

    return LABELS[pred_idx], margin



def _apply_materialized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast fill using the latest snapshot row per symbol from features_intraday_1m.
    Snapshot-only; no live OHLCV reads.
    """
    if df.empty:
        return df
    syms = sorted({str(s) for s in df["symbol"].astype(str).tolist() if s})
    if not syms:
        return df

    syms_df = pd.DataFrame({"symbol": syms})
    try:
        with duckdb.connect(DB_PATH, read_only=True) as conn:
            conn.register("syms", syms_df)
            mat = conn.execute("""
                SELECT *
                FROM features_intraday_1m
                WHERE symbol IN (SELECT symbol FROM syms)
                QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY snapshot_ts_utc DESC) = 1
            """).df()
            conn.unregister("syms")
    except Exception:
        return df  # silently fall back

    if mat.empty:
        return df

    mat["symbol"] = mat["symbol"].astype(str)
    m = mat.set_index("symbol")

    for i in range(len(df)):
        s = str(df.iloc[i]["symbol"])
        if s in m.index:
            row = m.loc[s]
            for c in MAT_COLS:
                val = row.get(c, None)
                if pd.notna(val):
                    df.at[df.index[i], c] = val
    return df


def verify_key(x_aarch_key: str | None = Header(default=None)):
    if REQUIRED_API_KEY and x_aarch_key != REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def load_artifacts():
    """
    Load CatBoost model and lock inference to the exact training feature set.
    We DO NOT trust feature_order.json here—serving must always use FEATURE_COLUMNS
    (NUM+CAT only), exactly as used at training time.
    """
    global _model, _feature_order, _cat_idx

    if _model is None:
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
        _model = model

    # Always use the training feature order (NUM+CAT only)
    _feature_order = list(FEATURE_COLUMNS)

    # Indices of categorical features in the serving order
    _cat_idx = [_feature_order.index(c) for c in CAT_FEATURES if c in _feature_order]



def _prep_categoricals_for_catboost(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    for c in cat_cols:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = df[c].astype("string").fillna("__NA__")
    return df


def _df_records_with_nulls(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    recs = df.where(pd.notnull(df), None).to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, np.integer)):
                r[k] = v.item()
    return recs


def _json_sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ---- add this block after enrichment and before _predict(...) ----
CR_TZ = "America/Costa_Rica"
now_cr = pd.Timestamp.now(tz=CR_TZ).floor("s")

def _to_cr_tz(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    ts = pd.to_datetime(v, errors="coerce", utc=False)
    if ts is None or pd.isna(ts):
        return None
    # If naive, assume the field labeled "asof_utc" was UTC and localize first
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(CR_TZ)


def _predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run CatBoost inference.
    Ensures all training-time feature columns (NUM+CAT only) exist
    and in the correct order before prediction.
    """
    # Guarantee every expected column exists
    for c in _feature_order:
        if c not in df.columns:
            df[c] = np.nan

    # Lock to training feature set (NUM+CAT only)
    X = df[_feature_order].copy()

    # Prepare categoricals with sentinel values
    X = _prep_categoricals_for_catboost(X, CAT_FEATURES)

    pool = Pool(X, cat_features=_cat_idx)
    proba = _model.predict_proba(pool)

    out = pd.DataFrame(proba, columns=CLASS_NAMES)
    return out


# -------------------------------------------------------------------
# Snapshot-only enrichment (per row)
# -------------------------------------------------------------------
def _enrich_with_intraday(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Snapshot-only enrichment path:
    For each symbol, fetch the latest features from features_intraday_1m
    via latest_snapshot_features(); never touches live OHLCV.
    """
    metrics = {
        "enabled": True, "minutes": INTRADAY_MINUTES, "overwrite": INTRADAY_OVERWRITE,
        "source": "snapshot",
        "symbols_processed": 0, "cache_hits": 0, "cache_misses": 0,
        "fields_filled": 0, "errors": [], "time_ms": 0, "symbols_details": []
    }
    if df.empty:
        return df, metrics

    t0 = time.perf_counter()

    # Ensure all needed columns exist with sane defaults
    needed = INTRADAY_COLS + ["mkt_etf", "sector_etf", "symbol", "session", "asof_utc"]
    for c in needed:
        if c not in df.columns:
            df[c] = None if c in ("symbol", "session", "asof_utc", "mkt_etf", "sector_etf") else np.nan


    pre_mask = df[INTRADAY_COLS].notna().copy()

    cache: dict[str, tuple[dict, dict]] = {}
    seen = set()

    for i in range(len(df)):
        sym = str(df.iloc[i].get("symbol") or "").strip()
        if not sym:
            continue

        if sym not in cache:
            try:
                feats, _session, info = latest_snapshot_features(DB_PATH, sym)
                cache[sym] = (feats, info)
                metrics["cache_misses"] += 1
            except Exception as e:
                metrics["errors"].append(f"{sym}: {e}")
                continue
        else:
            feats, info = cache[sym]
            metrics["cache_hits"] += 1

        metrics["symbols_processed"] += 1

        def _maybe_fill(col, val):
            if val is None:
                return
            cur = df.iloc[i].get(col, None)
            if INTRADAY_OVERWRITE or (pd.isna(cur) or cur is None):
                df.at[df.index[i], col] = val

        for c in INTRADAY_COLS:
            _maybe_fill(c, feats.get(c))

        if sym not in seen:
            d = {"symbol": sym, **{k: info.get(k) for k in (
                "latest_ts_utc","latest_ts_ny","stale_sec","in_rth",
                "bars_today","enough_rsi","enough_macd","enough_vwap","used_rth_only"
            )}}
            metrics["symbols_details"].append(d)
            seen.add(sym)

    post_mask = df[INTRADAY_COLS].notna()
    metrics["fields_filled"] = int((post_mask & ~pre_mask).sum().sum())
    metrics["time_ms"] = int((time.perf_counter() - t0) * 1000)
    return df, metrics

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
load_artifacts()

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "version": API_VERSION}

@app.post("/ingest_signals")
async def ingest_signals(request: Request, _: bool = Depends(verify_key)):
    body = await request.json()
    rows = body.get("rows", [])
    _append_signals(rows)
    return {"ok": True, "ingested": len(rows)}

@app.post("/score_batch")
async def score_batch(
    request: Request,
    use_materialized: int = Query(default=1),  # default to snapshot
    _: bool = Depends(verify_key),
):
    payload = await request.json()
    try:
        batch = InferBatch.model_validate(payload)
        data = batch.root
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    jobs = [j.model_dump() for j in data]
    raw_rows_flat = [r for job in jobs for r in job["rows"]]
    df_full = rows_to_frame(jobs)

    # Enrich
    if use_materialized:
        df_full = _apply_materialized(df_full)  # bulk fast path
        intraday_meta = {"source": "materialized"}
    else:
        df_full, intraday_meta = _enrich_with_intraday(df_full)  # per-row snapshot

    # ---- Normalize timestamps to UTC-6 (America/Costa_Rica) for API response
    CR_TZ = "America/Costa_Rica"
    now_cr = pd.Timestamp.now(tz=CR_TZ).floor("s")

    def _to_cr_tz(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        ts = pd.to_datetime(v, errors="coerce", utc=False)
        if ts is None or pd.isna(ts):
            return None
        # If naive or UTC (e.g., ends with Z), localize to UTC then convert to CR
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(CR_TZ)

    # Ensure asof_utc exists and is in UTC-6; default to "now" if missing
    if "asof_utc" in df_full.columns:
        df_full["asof_utc"] = df_full["asof_utc"].apply(_to_cr_tz).fillna(now_cr)
    else:
        df_full["asof_utc"] = now_cr

    # Also add a per-row "ts" (now in UTC-6) for hydrated output
    df_full["ts"] = now_cr

    # Predict
    preds = _predict(df_full)

    # Compute serve-style decision + margin per row
    decisions = []
    for i in range(len(preds)):
        p_down = float(preds.iloc[i]["down"])
        p_flat = float(preds.iloc[i]["flat"])
        p_up   = float(preds.iloc[i]["up"])
        lbl, mg = decide_from_probs(p_down, p_flat, p_up)
        decisions.append({"label": lbl, "margin": mg})


    # Build response rows
    hydrated_records = _df_records_with_nulls(df_full)
    preds_records = _df_records_with_nulls(preds)

    rows_out = []
    for i in range(len(df_full)):
        rows_out.append({
            "input": raw_rows_flat[i],
            "hydrated": hydrated_records[i],
            "prediction": preds_records[i],        # raw probabilities (down/flat/up)
            "decision": decisions[i],              # label + margin from helper
        })


    resp = {
        "ok": True,
        "count": len(rows_out),
        "classes": CLASS_NAMES,
        "feature_order": _feature_order,
        "meta": {"intraday": intraday_meta},
        "rows": rows_out
    }

    if WRITE_SIGNALS:
        try:
            _append_signals(rows_out)
        except Exception as e:
            print(f"[signals] append failed: {e}")

    return _json_sanitize(resp)


@app.post("/score")
async def score(
    request: Request,
    use_materialized: int = Query(default=1),   # default to snapshot
    _: bool = Depends(verify_key),
    ):
    payload = await request.json()

    # Normalize to a batch
    if isinstance(payload, dict) and payload.get("mode") == "infer":
        payload = [payload]
    elif isinstance(payload, dict) and "rows" in payload:
        payload = [{"mode": "infer", "schema_version": 1, "rows": payload["rows"]}]

    # Validate
    try:
        batch = InferBatch.model_validate(payload)
        data = batch.root
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # Jobs → DataFrame (preserve order)
    jobs = [j.model_dump() for j in data]
    raw_rows_flat = [r for job in jobs for r in job["rows"]]
    df_full = rows_to_frame(jobs)

    # Enrich from snapshot
    if use_materialized:
        df_full = _apply_materialized(df_full)
        intraday_meta = {"source": "materialized"}
    else:
        df_full, intraday_meta = _enrich_with_intraday(df_full)

    # ---- NEW: normalize timestamps to UTC-6 (America/Costa_Rica) for the API response
    CR_TZ = "America/Costa_Rica"
    now_cr = pd.Timestamp.now(tz=CR_TZ).floor("s")

    def _to_cr_tz(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        ts = pd.to_datetime(v, errors="coerce", utc=False)
        if ts is None or pd.isna(ts):
            return None
        # If the incoming "asof_utc" is naive or ends with Z (UTC),
        # localize to UTC first, then convert to CR time.
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(CR_TZ)

    # Ensure asof_utc present and converted to UTC-6; default to "now" if missing
    if "asof_utc" in df_full.columns:
        df_full["asof_utc"] = df_full["asof_utc"].apply(_to_cr_tz)
        df_full["asof_utc"] = df_full["asof_utc"].fillna(now_cr)
    else:
        df_full["asof_utc"] = now_cr

    # Also add a per-row "ts" (now in UTC-6) for hydrated output
    df_full["ts"] = now_cr

    preds = _predict(df_full)

    # Compute serve-style decision + margin per row
    decisions = []
    for i in range(len(preds)):
        p_down = float(preds.iloc[i]["down"])
        p_flat = float(preds.iloc[i]["flat"])
        p_up   = float(preds.iloc[i]["up"])
        lbl, mg = decide_from_probs(p_down, p_flat, p_up)
        decisions.append({"label": lbl, "margin": mg})


    hydrated_records = _df_records_with_nulls(df_full)
    preds_records = _df_records_with_nulls(preds)

    rows_out = []
    for i in range(len(df_full)):
        rows_out.append({
            "input": raw_rows_flat[i],
            "hydrated": hydrated_records[i],
            "prediction": preds_records[i],        # probabilities
            "decision": decisions[i],              # helper result
        })


    resp = {
        "ok": True,
        "count": len(rows_out),
        "classes": CLASS_NAMES,
        "feature_order": _feature_order,
        "meta": {"intraday": intraday_meta},
        "rows": rows_out,
    }

    if WRITE_SIGNALS:
        try:
            _append_signals(rows_out)
        except Exception as e:
            print(f"[signals] append failed: {e}")

    return _json_sanitize(resp)


# -------------------------------------------------------------------
# Signals logging (unchanged)
# -------------------------------------------------------------------
def _append_signals(rows_out: list[dict]) -> None:
    if not rows_out or not SIGNALS_DB:
        return

    recs = []
    now = pd.Timestamp.now(tz="America/Costa_Rica")   # UTC-6

    for r in rows_out:
        hyd = r.get("hydrated", {}) or {}
        pred = r.get("prediction", {}) or {}
        symbol = hyd.get("symbol")
        if not symbol:
            continue

        p_up   = pred.get("up",   pred.get("p_up"))
        p_down = pred.get("down", pred.get("p_down"))
        p_flat = pred.get("flat", pred.get("p_flat"))

        # Use serve rules to compute decision + margin
        p_up_f   = float(p_up)   if isinstance(p_up,   (int, float)) else 0.0
        p_down_f = float(p_down) if isinstance(p_down, (int, float)) else 0.0
        p_flat_f = float(p_flat) if isinstance(p_flat, (int, float)) else 0.0

        decision_lbl, margin = decide_from_probs(p_down_f, p_flat_f, p_up_f)


        feat_subset = {
            k: hyd.get(k)
            for k in ("rsi_14", "dist_vwap_bps", "rel_sector_vs_mkt", "vol_zscore_today", "rv_30m")
            if k in hyd
        }

        recs.append({
            "ts_utc": now.to_pydatetime(),   # always UTC-6
            "symbol": symbol,
            "session": hyd.get("session"),
            "decision": decision_lbl,
            "p_up": float(p_up) if p_up is not None else None,
            "p_down": float(p_down) if p_down is not None else None,
            "p_flat": float(p_flat) if p_flat is not None else None,
            "margin": margin,
            "reasons": json.dumps(r.get("decision", {}).get("reasons", [])),
            "features": json.dumps(feat_subset),
            "asof_utc": (
                pd.to_datetime(hyd.get("asof_utc"), utc=True)
                  .tz_convert("America/Costa_Rica")
                  .to_pydatetime()
                if hyd.get("asof_utc") else now.to_pydatetime()
            ),
        })

    if not recs:
        return

    df = pd.DataFrame(recs)
    with _sig_lock:
        con = duckdb.connect(SIGNALS_DB)
        con.register("df", df)
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
              ts_utc TIMESTAMP,
              symbol TEXT,
              session TEXT,
              decision TEXT,
              p_up DOUBLE, p_down DOUBLE, p_flat DOUBLE,
              margin DOUBLE,
              reasons JSON,
              features JSON,
              asof_utc TIMESTAMP
            )
        """)
        con.execute("INSERT INTO signals SELECT * FROM df")
        con.unregister("df")
        con.close()


# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8500"))
    uvicorn.run("serve:app", host="127.0.0.1", port=port, reload=False)
