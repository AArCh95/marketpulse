#!/usr/bin/env python3
# D:\AARCH\DBs\market_sync_yahoo.py
#
# Purpose
# -------
# Keep two DuckDB tables current using Yahoo Finance:
#   - ohlcv_1d : daily bars + daily indicators (rsi_14, macd_line, macd_signal, macd_hist)
#   - ohlcv_1m : intraday 1-minute bars + per-minute VWAP distance (dist_vwap_bps)
#
# Modes
# -----
# 1) DAILY  (run ONCE each morning, before RTH)
#    - Fetches ~1 year of daily bars (fallback 6mo if needed).
#    - Upserts into ohlcv_1d.
#    - Computes daily indicators (RSI/MACD) and writes them into the same rows.
#    - Publishes a read-only DB snapshot for downstream readers.
#
# 2) INTRADAY (run CONTINUOUSLY during market hours)
#    - Loops through the symbol list, fetching TODAY’s 1-minute data per symbol.
#    - Upserts into ohlcv_1m and recomputes per-minute VWAP distance (dist_vwap_bps).
#    - After each batch, publishes a read-only DB snapshot for readers.
#
# Snapshots (read-only for model/API)
# -----------------------------------
# - After DAILY completes and after each INTRADAY batch, we create a timestamped
#   copy of the DB and write a pointer file:
#       <DB_DIR>\CURRENT_SNAPSHOT.txt
#   that contains the path of the newest snapshot.
# - The CatBoost API opens ONLY the snapshot referenced by that pointer—never the live DB.
#   This avoids Windows file-lock conflicts between the writer and the API.
#
# Typical Scheduling (America/Costa_Rica, UTC-6)
# ----------------------------------------------
# - DAILY:   once ~06:00 CR time (≈ 08:00 ET premarket)
# - INTRADAY: start ~07:25–07:30 CR (≈ 09:25–09:30 ET) and keep it running until the close.
#
# Examples
# --------
# DAILY (one-shot):
#   python market_sync_yahoo.py --duckdb "D:\AARCH\DBs\market.duckdb" --mode daily
#
# INTRADAY (continuous, batch 60 syms, 0.35s throttle between symbols):
#   python market_sync_yahoo.py --duckdb "D:\AARCH\DBs\market.duckdb" --mode intraday --batch-size 60 --sleep 0.35
#
# Notes
# -----
# - The DAILY job updates ONLY ohlcv_1d (and its RSI/MACD fields).
# - The INTRADAY job updates ONLY ohlcv_1m (and its dist_vwap_bps).
# - Together, both tables stay fresh: daily-level indicators + minute-level VWAP context.
# - If you ever need a 5-day, 1-minute backfill (rare), run a one-off backfill script;
#   intraday mode intentionally focuses on TODAY for speed and freshness.



import os, time, math, sys, argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import requests
import duckdb
import pandas as pd
import numpy as np
import shutil
from pathlib import Path


HEADERS_YF = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
YF_HOSTS = [
    "https://query1.finance.yahoo.com/v8/finance/chart",
    "https://query2.finance.yahoo.com/v8/finance/chart",
]

# Daily bars: 1y (fallback 6mo)
DAILY_PARAM_SETS = [
    " ",
    "range=6mo&interval=1d&lang=en-US&region=US&includePrePost=false&events=div,splits",
]

# Intraday bars: prefer 5d of 1m (fallback 1d)
INTRA_PARAM_SETS = [
   "range=5d&interval=1m&lang=en-US&region=US&includePrePost=true",
   "range=1d&interval=1m&lang=en-US&region=US&includePrePost=true",
]

INTRA_PARAM_SETS_FAST = [
    "range=1d&interval=1m&lang=en-US&region=US&includePrePost=false"
]

# --- add near other helpers ---
def daily_count(conn, sym: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM ohlcv_1d WHERE symbol = ?", [sym]).fetchone()[0]

DAILY_SMALL = [
    "range=1mo&interval=1d&lang=en-US&region=US&includePrePost=false&events=div,splits"
]
DAILY_MEDIUM = [
    "range=6mo&interval=1d&lang=en-US&region=US&includePrePost=false&events=div,splits"
]
DAILY_LARGE = [
    "range=1y&interval=1d&lang=en-US&region=US&includePrePost=false&events=div,splits"
]

def fetch_daily_dynamic(conn, sym: str):
    cnt = daily_count(conn, sym)
    # Choose smallest range that meets warm-up needs
    if cnt >= 180:
        param_sets = DAILY_SMALL     # tiny top-up
    elif cnt >= 60:
        param_sets = DAILY_MEDIUM    # stronger warm-up
    else:
        param_sets = DAILY_LARGE     # first bootstrap
    return fetch_chart(sym, param_sets)

SYMBOLS = [
    "AAPL","ABBV","ABT","ACGL","ACN","ADBE","ADM","AEP","AFL","AIG",
    "AMAT","AMD","AMGN","AMT","AMZN","AON","APA","APP","ARE","ARM",
    "ASML","ATO","AVGO","AXON","AXP","AZN","BA","BAC","BALL","BAX",
    "BBY","BEN","BIIB","BK","BKNG","BKR","BLK","BMY","BRK-B","BRO",
    "C","CARR","CAT","CB","CBOE","CCEP","CDNS","CDW","CEG","CHD",
    "CHTR","CL","CMCSA","CMS","CNP","COF","COO","COP","COST","CPRT",
    "CRM","CSCO","CSGP","CTAS","CTSH","CTVA","CVS","CVX","D","DAL",
    "DASH","DDOG","DE","DG","DHI","DHR","DIS","DLR","DLTR","DOV",
    "DRI","DUK","DVN","DXCM","EA","ED","EFX","EIX","EL","EMR",
    "EOG","EQT","ESS","ETN","EXC","F","FAST","FDX","FE","FIS",
    "FITB","FMC","FTNT","GD","GE","GEHC","GFS","GILD","GL","GLW",
    "GM","GOOG","GOOGL","GS","HAL","HAS","HD","HIG","HON","HPE",
    "HPQ","HRL","HSY","HUM","IBM","IDXX","IEX","IFF","INTC","INTU",
    "IP","IPG","IRM","ISRG","IT","JBHT","JCI","JNJ","JPM","K",
    "KEY","KIM","KLAC","KMB","KMI","KMX","KO","KR","KSS","L",
    "LDOS","LEG","LEN","LH","LIN","LKQ","LLY","LMT","LNC","LNT",
    "LOW","LULU","LUMN","LVS","LW","LYB","MA","MAA","MAS","MCD",
    "MCK","MDLZ","MDT","MELI","MET","META","MKC","MLM","MMM","MNST",
    "MO","MOS","MPWR","MRK","MRVL","MS","MSCI","MSFT","MSTR","MTB",
    "MTD","MU","NEE","NFLX","NKE","NOW","NVDA","NXPI","ODFL","ON",
    "ORCL","PANW","PDD","PEP","PFE","PG","PLTR","PM","PYPL","QCOM",
    "REGN","ROP","ROST","RTX","SBUX","SCHW","SHOP","SNPS","SO","SPG",
    "T","TEAM","TGT","TMO","TMUS","TSLA","TTD","TTWO","TXN","UNH",
    "UNP","UPS","USB","V","VRTX","VZ","WBD","WDAY","WFC","WMT",
    "XEL","XOM","ZS"
]

SYMBOL_FIXUPS = {"BRK.B": "BRK-B"}

# ---------- Yahoo fetch helpers ----------
def build_symbol_list() -> List[str]:
    seen, out = set(), []
    for s in SYMBOLS:
        t = SYMBOL_FIXUPS.get(s.strip().upper(), s.strip().upper())
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def fetch_chart(symbol: str, param_sets, timeout=30, sleep_between=0.35) -> Optional[Dict[str, Any]]:
    last_err = None
    for params in param_sets:
        for host in YF_HOSTS:
            url = f"{host}/{requests.utils.quote(symbol)}?{params}"
            try:
                r = requests.get(url, headers=HEADERS_YF, timeout=timeout)
                if r.status_code == 200:
                    chart = r.json().get("chart")
                    if chart and chart.get("result"):
                        return chart
                    last_err = (chart or {}).get("error") or {"code": r.status_code, "description": "empty chart"}
                else:
                    last_err = {"code": r.status_code, "description": r.text[:200]}
            except Exception as e:
                last_err = {"code": "EXC", "description": str(e)}
            time.sleep(sleep_between)
    sys.stderr.write(f"[WARN] Yahoo fail {symbol}: {last_err}\n")
    return None

def assert_daily_indicators_fresh(con):
    """
    Fails if the latest row per symbol lacks RSI/MACD.
    Run after the daily indicators update.
    """
    bad = con.execute("""
        WITH last AS (
          SELECT symbol, MAX(ts) AS max_ts
          FROM ohlcv_1d
          GROUP BY 1
        )
        SELECT l.symbol, l.max_ts
        FROM last l
        JOIN ohlcv_1d d
          ON d.symbol = l.symbol AND d.ts = l.max_ts
        WHERE d.rsi_14 IS NULL
           OR d.macd_line IS NULL
           OR d.macd_signal IS NULL
           OR d.macd_hist IS NULL
        ORDER BY 1
    """).fetchall()

    if bad:
        sample = ", ".join(s for s, _ in bad[:15])
        raise RuntimeError(
            f"Indicator health check failed for {len(bad)} symbols (e.g., {sample})."
        )


def safe_num(arr, i):
    try:
        v = arr[i]; 
        if v is None: return None
        n = float(v)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def chart_to_df(chart: Dict[str, Any], fallback_symbol: str, timeframe: str) -> pd.DataFrame:
    res = (chart or {}).get("result", [None])[0]
    if not res:
        return pd.DataFrame(columns=["symbol","timeframe","ts","open","high","low","close","volume"])
    sym = (res.get("meta") or {}).get("symbol") or fallback_symbol
    ts = res.get("timestamp") or []
    q0 = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    opens  = q0.get("open")   or []
    highs  = q0.get("high")   or []
    lows   = q0.get("low")    or []
    closes = q0.get("close")  or []
    vols   = q0.get("volume") or []

    out = []
    for i in range(len(ts)):
        t = int(ts[i])
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
        out.append({
            "symbol": sym,
            "timeframe": timeframe,
            "ts": ts_iso,
            "open":  safe_num(opens,  i),
            "high":  safe_num(highs,  i),
            "low":   safe_num(lows,   i),
            "close": safe_num(closes, i),
            "volume":safe_num(vols,   i),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def publish_snapshot(primary_path: str, snapshot_dir: str) -> str:
    """
    Create a timestamped read-only snapshot of the DuckDB file and write a pointer file.
    Returns the snapshot path.
    """
    p = Path(primary_path)
    sd = Path(snapshot_dir)
    sd.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap = sd / f"{p.stem}_snap_{ts}.duckdb"
    tmp  = sd / f".{p.stem}_snap_{ts}.duckdb.tmp"

    # Copy atomically: copy -> os.replace
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(p, tmp)
    os.replace(tmp, snap)

    # Update pointer file
    ptr = sd / "CURRENT_SNAPSHOT.txt"
    with open(ptr, "w", encoding="utf-8") as f:
        f.write(str(snap))

    print(f"[SNAPSHOT] Published {snap} and updated {ptr}")
    return str(snap)


# ---------- Indicators ----------
def compute_rsi(close: pd.Series, period=14) -> pd.Series:
    s = close.astype(float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    s = close.astype(float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def compute_vwap_dist_per_day(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative VWAP per day and return (symbol, ts, dist_vwap_bps).
    Handles multiple session dates inside df_1m (e.g., 5d payload).
    """
    if df_1m.empty:
        return pd.DataFrame(columns=["symbol","ts","dist_vwap_bps"])

    df = df_1m.copy().sort_values("ts")
    df["d"] = df["ts"].dt.tz_convert("UTC").dt.date

    # Typical price & volumes
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v  = df["volume"].fillna(0.0)

    # Group by day, cumulative PV and V
    df["pv"] = tp * v
    df["cum_pv"] = df.groupby("d")["pv"].cumsum()
    df["cum_v"]  = df.groupby("d")["volume"].cumsum().replace(0, np.nan)
    df["vwap"]   = df["cum_pv"] / df["cum_v"]

    dist = (df["close"] / df["vwap"] - 1.0) * 10000.0
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "ts": df["ts"],
        "dist_vwap_bps": dist.round(2).astype(float)
    })
    return out



# ---------- DuckDB upserts / updates ----------
def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame):
    if df.empty: return 0
    if table == "ohlcv_1m":
        insert_cols = ["symbol","ts","open","high","low","close","volume","dist_vwap_bps"]
        if "dist_vwap_bps" not in df.columns:
            df = df.assign(dist_vwap_bps=pd.Series([None]*len(df)))
    else:
        insert_cols = ["symbol","ts","open","high","low","close","volume","rsi_14","macd_line","macd_signal","macd_hist"]
        for c in ["rsi_14","macd_line","macd_signal","macd_hist"]:
            if c not in df.columns:
                df[c] = None

    if not pd.api.types.is_datetime64tz_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df = df[insert_cols].copy()
    con.register("tmp_upsert", df)
    con.execute(f"""
        INSERT OR REPLACE INTO {table} ({", ".join(insert_cols)})
        SELECT {", ".join(insert_cols)} FROM tmp_upsert
    """)
    con.unregister("tmp_upsert")
    return len(df)


def cleanup_old_snapshots(snapshot_dir: str, keep: int = 10, pointer_path: str | None = None):
    """
    Keep the newest `keep` snapshots (plus the currently-pointed snapshot),
    delete the rest. Safe on Windows (ignore 'in use' errors).
    """
    sd = Path(snapshot_dir)
    if pointer_path is None:
        pointer_path = str(sd / "CURRENT_SNAPSHOT.txt")

    current = None
    try:
        current = Path(Path(pointer_path).read_text(encoding="utf-8").strip())
    except Exception:
        pass

    snaps = sorted(sd.glob("*_snap_*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Build a list that excludes the current snapshot (never delete it)
    filtered = []
    for p in snaps:
        try:
            if current and p.samefile(current):
                continue
        except Exception:
            if current and str(p) == str(current):
                continue
        filtered.append(p)

    # Delete everything after the first `keep` newest
    for p in filtered[keep:]:
        try:
            p.unlink()
        except Exception:
            pass


def update_daily_indicators(con, sym: str, recalc_all: bool=False):
    limit = 100000 if recalc_all else 260
    df = con.execute("""
        WITH last AS (
          SELECT ts, close
          FROM ohlcv_1d
          WHERE symbol = ?
          ORDER BY ts DESC
          LIMIT ?
        )
        SELECT ts, close FROM last ORDER BY ts
    """, [sym, limit]).df()
    if df.empty or len(df) < 30:
        return 0
    rsi = compute_rsi(df["close"]).round(2)
    macd_line, macd_signal, macd_hist = compute_macd(df["close"])
    feat = pd.DataFrame({
        "symbol": sym, "ts": df["ts"],
        "rsi_14": rsi.astype(float),
        "macd_line": macd_line.astype(float).round(6),
        "macd_signal": macd_signal.astype(float).round(6),
        "macd_hist": macd_hist.astype(float).round(6),
    })
    con.register("tmp_feat", feat)
    con.execute("""
        UPDATE ohlcv_1d AS d
        SET rsi_14 = tf.rsi_14,
            macd_line = tf.macd_line,
            macd_signal = tf.macd_signal,
            macd_hist = tf.macd_hist
        FROM tmp_feat AS tf
        WHERE d.symbol = tf.symbol AND d.ts = tf.ts
    """)
    con.unregister("tmp_feat")
    return len(feat)


def update_intraday_vwap_from_df(con: duckdb.DuckDBPyConnection, df_m: pd.DataFrame):
    """
    Given a 1m DataFrame (possibly up to 5 days), compute per-day VWAP dist and update.
    """
    if df_m.empty:
        return 0
    upd = compute_vwap_dist_per_day(df_m)
    if upd.empty:
        return 0
    con.register("tmp_vw", upd)
    con.execute("""
        UPDATE ohlcv_1m AS m
        SET dist_vwap_bps = tv.dist_vwap_bps
        FROM tmp_vw AS tv
        WHERE m.symbol = tv.symbol AND m.ts = tv.ts
    """)
    con.unregister("tmp_vw")
    return len(upd)

def run_daily_premarket(args):
    con = duckdb.connect(args.duckdb)
    symbols = build_symbol_list()
    if args.max:
        symbols = symbols[:args.max]
    total_d = 0
    for i, sym in enumerate(symbols, 1):
        ch_d = fetch_daily_dynamic(con, sym)
        if ch_d:
            df_d = chart_to_df(ch_d, sym, "1d")
            n1 = upsert_df(con, "ohlcv_1d", df_d)
            # recompute using only the last ~260 bars
            update_daily_indicators(con, sym, recalc_all=False)
            total_d += n1
        time.sleep(args.sleep)
        print(f"[{i}/{len(symbols)}] {sym}: daily upserted={n1 if ch_d else 0}")
    con.close()
    print(f"Done (daily). Upserted={total_d}")



def update_intraday_today(con, sym: str, sleep_between=0.35):
    """Intraday fast: fetch 1d 1m only; compute VWAP distance for *today* only."""
    ch_m = fetch_chart(sym, INTRA_PARAM_SETS_FAST, sleep_between=sleep_between)
    if not ch_m:
        return 0
    df_m = chart_to_df(ch_m, sym, "1m")
    n2 = upsert_df(con, "ohlcv_1m", df_m)
    if n2:
        # Compute VWAP distance for *today* only in SQL (faster than Python loop)
        con.execute("""
            WITH x AS (
              SELECT
                symbol, ts, high, low, close, volume,
                (high + low + close)/3.0 AS tp,
                DATE_TRUNC('day', ts AT TIME ZONE 'America/New_York') AS session
              FROM ohlcv_1m
              WHERE symbol = ? AND ts >= DATE_TRUNC('day', NOW() AT TIME ZONE 'America/New_York')
            ),
            y AS (
              SELECT
                symbol, ts, close,
                SUM(tp*volume) OVER (PARTITION BY symbol, session ORDER BY ts) AS cum_pv,
                NULLIF(SUM(volume) OVER (PARTITION BY symbol, session ORDER BY ts), 0) AS cum_v
              FROM x
            ),
            z AS (
              SELECT
                symbol, ts,
                ROUND(((close / (cum_pv / cum_v)) - 1) * 10000, 2) AS dist_vwap_bps
              FROM y
              WHERE cum_v IS NOT NULL
            )
            UPDATE ohlcv_1m AS m
            SET dist_vwap_bps = z.dist_vwap_bps
            FROM z
            WHERE m.symbol = z.symbol AND m.ts = z.ts
        """, [sym])
    return n2


def run_intraday_fast(args):
    """Cycle through symbols continuously, today-only 1m, no end-of-batch sleep.
       Publishes a read-only snapshot each cycle so readers never touch the live DB.
    """
    import duckdb
    from datetime import datetime, timezone

    con = duckdb.connect(args.duckdb)
    symbols_full = build_symbol_list()
    if args.max:
        symbols_full = symbols_full[:args.max]

    idx = 0
    cycle_count = 0

    while True:
        batch = symbols_full[idx: idx + args.batch_size]
        if not batch:
            idx = 0
            batch = symbols_full[:args.batch_size]

        print(f"\nINTRADAY {datetime.now(timezone.utc).isoformat()} | Batch {idx}-{idx+len(batch)-1}")
        msum = 0
        for j, sym in enumerate(batch, 1):
            try:
                n2 = update_intraday_today(con, sym, sleep_between=args.sleep)
                msum += n2
                print(f"  [{j}/{len(batch)}] {sym}: 1m(today) upserted={n2}")
            except Exception as e:
                print(f"  [{j}/{len(batch)}] {sym}: ERR {e}")

        # Move to next batch immediately
        idx += len(batch)

        # === Publish a consistent snapshot for readers ===
        try:
            # Ensure latest changes are durable before copying
            try:
                con.execute("CHECKPOINT")
            except Exception:
                pass
            con.close()  # release write lock so Windows can copy the file
        except Exception:
            pass

        try:
            publish_snapshot(args.duckdb, args.snapshot_dir)
        finally:
            # Reopen writer connection for the next cycle
            con = duckdb.connect(args.duckdb)

        # Optional: periodic cleanup (keep the newest ~12 snapshots)
        cycle_count += 1
        if cycle_count % 10 == 0:
            try:
                cleanup_old_snapshots(args.snapshot_dir, keep=12)
            except Exception:
                pass




# ---------- per-symbol workflow ----------
def fetch_and_update_for_symbol(con, sym: str, include1m: bool, sleep_between=0.6):
    # 1) daily
    n1 = 0
    ch_d = fetch_chart(sym, DAILY_PARAM_SETS)
    if ch_d:
        df_d = chart_to_df(ch_d, sym, "1d")
        n1 = upsert_df(con, "ohlcv_1d", df_d)
        update_daily_indicators(con, sym, recalc_all=False)

    # 2) intraday (up to 5d)
    n2 = 0
    if include1m:
        ch_m = fetch_chart(sym, INTRA_PARAM_SETS)
        if ch_m:
            df_m = chart_to_df(ch_m, sym, "1m")
            n2 = upsert_df(con, "ohlcv_1m", df_m)   # upsert all minutes (1–5 days)
            if n2:
                update_intraday_vwap_from_df(con, df_m)

    time.sleep(sleep_between)
    return n1, n2

# ---------- runners ----------
def run_once(args):
    con = duckdb.connect(args.duckdb)
    symbols = build_symbol_list()
    if args.max:
        symbols = symbols[:args.max]

    total_d = total_m = 0
    for i, sym in enumerate(symbols, 1):
        n1, n2 = fetch_and_update_for_symbol(con, sym, args.include1m, args.sleep)
        total_d += n1; total_m += n2
        print(f"[{i}/{len(symbols)}] {sym}: upserted 1d={n1}, 1m(<=5d)={n2}")
    con.close()
    print(f"Done. Upserted daily rows: {total_d}, intraday rows: {total_m}")

def run_daemon(args):
    con = duckdb.connect(args.duckdb)
    symbols_full = build_symbol_list()
    idx = 0
    while True:
        batch = symbols_full[idx: idx + args.batch_size]
        if not batch:
            idx = 0
            batch = symbols_full[:args.batch_size]

        print(f"\nCycle @ {datetime.now(timezone.utc).isoformat()} | Batch {idx}-{idx+len(batch)-1} ({len(batch)} syms)")
        dsum = msum = 0
        for j, sym in enumerate(batch, 1):
            n1, n2 = fetch_and_update_for_symbol(con, sym, args.include1m, args.sleep)
            dsum += n1; msum += n2
            print(f"  [{j}/{len(batch)}] {sym}: 1d={n1}, 1m(<=5d)={n2}")
        print(f"Cycle complete: upserted 1d={dsum}, 1m={msum}. Sleeping {args.interval}s.")
        time.sleep(args.interval)
        idx += len(batch)

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Yahoo OHLCV -> DuckDB. Daily premarket + fast intraday today-only.")
    ap.add_argument("--duckdb", required=True, help="Path to DuckDB file (e.g., D:\\AARCH\\DBs\\market.duckdb)")
    ap.add_argument("--mode", choices=["daily","intraday"], required=True, help="Runner mode")
    ap.add_argument("--batch-size", type=int, default=60, help="Symbols per micro-batch")
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between symbols (throttle)")
    ap.add_argument("--max", type=int, default=None, help="Limit symbols for testing")
    ap.add_argument("--snapshot-dir", default=None, help="Directory to store read-only snapshots (defaults to DB dir)")
    args = ap.parse_args()

    # default snapshot directory = same folder as DB
    if args.snapshot_dir is None:
        args.snapshot_dir = str(Path(args.duckdb).resolve().parent)

    if args.mode == "daily":
        run_daily_premarket(args)
        # publish a snapshot after daily completes
        publish_snapshot(args.duckdb, args.snapshot_dir)
        cleanup_old_snapshots(args.snapshot_dir, keep=12)  # keep ~12 recent snapshots
    else:
        run_intraday_fast(args)  # this will publish snapshots each cycle
