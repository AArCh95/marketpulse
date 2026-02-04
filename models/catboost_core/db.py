# D:\AARCH\models\catboost_core\db.py
# Robust read-only DuckDB access for CatBoost service (snapshot-first, lock-safe)

from __future__ import annotations

import os
import glob
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timezone

import duckdb
import pandas as pd

# ---- Optional imports from config (don’t crash if missing) ----
CONFIG_DB_PATH = None
CONFIG_HORIZON = None
try:
    from config import DUCKDB_PATH as CONFIG_DB_PATH  # type: ignore
except Exception:
    pass

try:
    from config import HORIZON_D as CONFIG_HORIZON      # type: ignore
except Exception:
    pass

# ---- Environment / defaults ----
DUCKDB_PRIMARY: str = os.getenv("DB_PATH", CONFIG_DB_PATH or r"D:\AARCH\DBs\market.duckdb")
SNAPSHOT_PTR: str   = os.getenv("DB_SNAPSHOT_POINTER", r"D:\AARCH\DBs\CURRENT_SNAPSHOT.txt")
SNAPSHOT_DIR: str   = os.getenv("DB_SNAPSHOT_DIR", str(Path(DUCKDB_PRIMARY).resolve().parent))

# Name pattern used by your snapshot publisher, e.g. market_20250821_1455.duckdb
SNAPSHOT_GLOB: str  = os.getenv("DB_SNAPSHOT_GLOB", "market_*.duckdb")

# Horizon for training labels (days)
HORIZON_D: int = int(os.getenv("HORIZON_D", CONFIG_HORIZON or 5))


# ---------- Path resolvers ----------

def _read_pointer_file(ptr_path: Union[str, Path]) -> Optional[Path]:
    """Read CURRENT_SNAPSHOT.txt; return a valid file Path or None."""
    try:
        p = Path(ptr_path)
        if not p.is_file():
            return None
        raw = p.read_text(encoding="utf-8").strip().strip('"')
        if not raw:
            return None
        target = Path(raw)
        return target if target.is_file() else None
    except Exception:
        return None


def _latest_snapshot_from_dir() -> Optional[Path]:
    """Find newest snapshot in SNAPSHOT_DIR matching SNAPSHOT_GLOB."""
    try:
        patt = str(Path(SNAPSHOT_DIR) / SNAPSHOT_GLOB)
        candidates = glob.glob(patt)
        if not candidates:
            return None
        latest = max(candidates, key=lambda f: os.path.getmtime(f))
        return Path(latest)
    except Exception:
        return None


def _resolve_db_path() -> Path:
    """
    Prefer snapshot pointer -> latest snapshot in dir -> primary DB.
    The CatBoost service should *not* touch the live DB when writers run.
    """
    p = _read_pointer_file(SNAPSHOT_PTR)
    if p:
        return p
    s = _latest_snapshot_from_dir()
    if s:
        return s
    return Path(DUCKDB_PRIMARY)


# ---------- Safe connection (read-only, lock tolerant) ----------

def _connect_ro(db_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Open DuckDB read-only; if Windows reports a lock, copy to a temp file and open that.
    """
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except Exception:
        # Last resort: copy to temp and open the copy RO
        tmp = Path(tempfile.gettempdir()) / f"{db_path.stem}_ro_{os.getpid()}.duckdb"
        try:
            shutil.copy2(db_path, tmp)
            return duckdb.connect(str(tmp), read_only=True)
        except Exception as e2:
            # Surface original path so logs are clear
            raise RuntimeError(f"Failed to open DB (and temp copy) at {db_path}: {e2}") from e2


def get_conn() -> duckdb.DuckDBPyConnection:
    """
    Public entrypoint: always returns a read-only connection to the *snapshot*.
    Use with context manager in callers:  `with get_conn() as conn: ...`
    """
    db_path = _resolve_db_path()
    return _connect_ro(db_path)


# ---------- Time helpers ----------

def _to_utc_iso(dt_like: Union[str, datetime]) -> str:
    """
    Normalize any naive/aware datetime or ISO-ish string to strict UTC ISO (…Z).
    No heavy dependencies; stick to datetime.fromisoformat when possible.
    """
    if isinstance(dt_like, datetime):
        dt = dt_like
    else:
        s = str(dt_like).strip()
        # Try stdlib ISO parse first
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            # Very loose fallback: assume UTC if nothing else works
            dt = datetime.utcnow().replace(tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------- Simple fetch helpers ----------

def fetch_daily_row(conn: duckdb.DuckDBPyConnection, symbol: str, asof_utc: Union[str, datetime]):
    """
    Latest daily row <= asof (UTC). Returns pandas DataFrame with at most 1 row.
    """
    asof_iso = _to_utc_iso(asof_utc)
    q = """
        SELECT *
        FROM ohlcv_1d
        WHERE symbol = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT 1
    """
    return conn.execute(q, [symbol, asof_iso]).fetchdf()


def fetch_minute_row(conn: duckdb.DuckDBPyConnection, symbol: str, asof_utc: Union[str, datetime]):
    """
    Latest 1m row <= asof (UTC). Returns pandas DataFrame with at most 1 row.
    """
    asof_iso = _to_utc_iso(asof_utc)
    q = """
        SELECT *
        FROM ohlcv_1m
        WHERE symbol = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT 1
    """
    return conn.execute(q, [symbol, asof_iso]).fetchdf()


# ---------- Training frame (daily) ----------

def fetch_training_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Minimal daily training frame with RSI/MACD and forward return label.
    Uses HORIZON_D (default 5). Produces columns:
      prev_close, ret_1d_ratio, ret_5d_ratio, fwd_close_{HORIZON_D}, fwd_ret_{HORIZON_D}d
    """
    h = int(HORIZON_D)
    q = f"""
    WITH base AS (
      SELECT
        symbol,
        CAST(ts AS TIMESTAMP) AS ts,
        open, high, low, close, volume,
        rsi_14, macd_line, macd_signal, macd_hist
      FROM ohlcv_1d
    ),
    feat AS (
      SELECT
        symbol, ts, open, high, low, close, volume,
        rsi_14, macd_line, macd_signal, macd_hist,
        LAG(close, 1)  OVER (PARTITION BY symbol ORDER BY ts) AS prev_close,
        close / NULLIF(LAG(close,  1) OVER (PARTITION BY symbol ORDER BY ts), 0) AS ret_1d_ratio,
        close / NULLIF(LAG(close,  5) OVER (PARTITION BY symbol ORDER BY ts), 0) AS ret_5d_ratio,
        LEAD(close, {h}) OVER (PARTITION BY symbol ORDER BY ts) AS fwd_close_{h}
      FROM base
    ),
    label AS (
      SELECT
        *,
        CASE
          WHEN fwd_close_{h} IS NULL OR close IS NULL THEN NULL
          ELSE (fwd_close_{h} / NULLIF(close, 0) - 1.0)
        END AS fwd_ret_{h}d
      FROM feat
    )
    SELECT *
    FROM label
    WHERE prev_close IS NOT NULL
    ORDER BY symbol, ts
    """
    return conn.execute(q).fetchdf()
