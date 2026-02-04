# D:\AARCH\dashboard\db_app_mobile.py
# Mobile-friendly intraday viewer: Candles + VWAP/EMAs + CatBoost signals (OOH supported)

import os, glob, json
from pathlib import Path
from datetime import datetime, date, time as dtime, timezone

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------ Config ------------------
SNAPSHOT_DIR     = os.getenv("DB_SNAPSHOT_DIR", r"D:\AARCH\DBs")
DB_PATH_FALLBACK = os.getenv("DB_PATH", r"D:\AARCH\DBs\market.duckdb")
SIGNALS_DB       = os.getenv("SIGNALS_DB", r"D:\AARCH\DBs\signals.duckdb")
ET               = pytz.timezone("America/New_York")

st.set_page_config(
    page_title="MarketPulse — Mobile",
    layout="wide",                       # use full width on phones
    initial_sidebar_state="collapsed"
)

# ---- UI polish for mobile ----
st.markdown(
    """
    <style>
      .block-container {padding-top: .5rem; padding-bottom: .5rem; padding-left: .5rem; padding-right: .5rem;}
      h1, h2, h3 {margin: 0.15rem 0 .6rem 0;}
      .app-title {display:flex; align-items:center; gap:.5rem; white-space:nowrap; overflow:visible;}
      .app-title h1 {font-size: 1.35rem; line-height: 1.35rem; margin:0;}
      .tiny-cap {opacity: .7; font-weight: 500;}
      [data-testid="stMetricValue"] {font-size: 1.4rem;}
      [data-testid="stMetricDelta"] {font-size: .85rem;}
      /* Make dataframes denser */
      div[data-testid="stDataFrame"] tbody td {padding: 0.25rem .35rem;}
      div[data-testid="stDataFrame"] thead th {padding: 0.25rem .35rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-title"><span>📈</span><h1>MarketPulse <span class="tiny-cap">(Mobile)</span></h1></div>',
    unsafe_allow_html=True
)

# ------------------ Helpers ------------------
def to_et_naive(ts_utc):
    """UTC tz-aware -> ET (naive). Accepts Series or DatetimeIndex or str-like."""
    if isinstance(ts_utc, pd.DatetimeIndex):
        return ts_utc.tz_convert(ET).tz_localize(None)
    s = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    return s.dt.tz_convert(ET).dt.tz_localize(None)

def rth_bounds_utc(session_et: date) -> tuple[datetime, datetime]:
    start_et = ET.localize(datetime.combine(session_et, dtime(9, 30)))
    end_et   = ET.localize(datetime.combine(session_et, dtime(16, 0)))
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)

def find_latest_snapshot() -> str:
    """Prefer market_*.duckdb in SNAPSHOT_DIR; else fallback DB."""
    snap_dir = SNAPSHOT_DIR if os.path.isdir(SNAPSHOT_DIR) else str(Path(DB_PATH_FALLBACK).parent)
    candidates = sorted(
        glob.glob(os.path.join(snap_dir, "market_*.duckdb")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return candidates[0] if candidates else DB_PATH_FALLBACK

def _ensure_dt64ns(x: pd.Series | pd.DatetimeIndex) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    if isinstance(s, pd.DatetimeIndex):
        s = pd.Series(s)
    return s.astype("datetime64[ns]")

def add_vwap_robust(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP from PV/V (intraday cumulative). Prefer dist_vwap_bps when available."""
    if df.empty:
        df["vwap"] = np.nan
        return df
    df = df.sort_values("ts").copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v  = df["volume"].fillna(0.0)
    vwap_calc = (tp.mul(v).cumsum() / v.cumsum().replace(0, np.nan)).astype(float)
    vwap_from_dist = None
    if "dist_vwap_bps" in df.columns:
        with np.errstate(invalid="ignore"):
            vwap_from_dist = df["close"] / (1.0 + (df["dist_vwap_bps"].astype(float) / 10000.0))
    df["vwap"] = vwap_from_dist if vwap_from_dist is not None else vwap_calc
    if vwap_from_dist is not None:
        df["vwap"] = df["vwap"].where(df["vwap"].notna(), vwap_calc)
    return df

def _normalize_signals_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create event_ts_utc (UTC) & ts_plot (ET naive) from asof_utc (preferred) or ts_utc.
    Keeps OOH signals (e.g., weekends/after-hours) visible.
    """
    if df.empty:
        return df
    src = df.get("asof_utc")
    if src is None:
        ts_src = df["ts_utc"].astype(str)
    else:
        ts_src = np.where(src.astype(str).str.len() > 0, src.astype(str), df["ts_utc"].astype(str))
    parsed = pd.to_datetime(ts_src, utc=True, errors="coerce")
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed, index=df.index)
    df["event_ts_utc"] = parsed
    df["ts_plot"] = _ensure_dt64ns(to_et_naive(df["event_ts_utc"]))
    return df.sort_values("event_ts_utc")

# ------------------ Caching ------------------
@st.cache_resource
def get_conn(db_path: str):
    return duckdb.connect(db_path, read_only=True)  # read-only avoids locks

@st.cache_data(ttl=20)
def list_symbols(db_path: str) -> list[str]:
    try:
        con = get_conn(db_path)
        df = con.execute("SELECT DISTINCT symbol FROM ohlcv_1m ORDER BY symbol").df()
        return df["symbol"].tolist()
    except Exception:
        return []

@st.cache_data(ttl=15)
def last_session_for_symbol(db_path: str, symbol: str) -> date | None:
    try:
        con = get_conn(db_path)
        df = con.execute("SELECT max(ts) AS max_ts FROM ohlcv_1m WHERE symbol = ?", [symbol]).df()
        if df.empty or pd.isna(df.loc[0, "max_ts"]): return None
        ts_utc = pd.to_datetime(df.loc[0, "max_ts"], utc=True)
        return ts_utc.tz_convert(ET).date()
    except Exception:
        return None

@st.cache_data(ttl=15)
def load_intraday_rth(db_path: str, symbol: str, session_et: date) -> pd.DataFrame:
    t0, t1 = rth_bounds_utc(session_et)
    con = get_conn(db_path)
    q = """
      SELECT symbol, ts, open, high, low, close, volume, dist_vwap_bps
      FROM ohlcv_1m
      WHERE symbol = ? AND ts BETWEEN ? AND ?
      ORDER BY ts
    """
    df = con.execute(q, [symbol, t0, t1]).df()
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = add_vwap_robust(df)
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ts_plot"] = _ensure_dt64ns(to_et_naive(df["ts"]))
    return df

@st.cache_data(ttl=10)
def load_signals_for_chart(symbol: str, session_et: date, include_ah: bool = True) -> pd.DataFrame:
    """Load signals for the session (all day). If include_ah=False, we trim to RTH window."""
    if not os.path.exists(SIGNALS_DB): return pd.DataFrame()
    con = duckdb.connect(SIGNALS_DB, read_only=True)
    session_str = session_et.strftime("%Y-%m-%d")
    df = con.execute("""
        SELECT ts_utc, asof_utc, symbol, session,
               decision, p_up, p_down, p_flat, margin, reasons, features
        FROM signals
        WHERE symbol = ? AND session = ?
        ORDER BY ts_utc
    """, [symbol, session_str]).df()
    con.close()
    if df.empty: return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = _normalize_signals_ts(df)
    if not include_ah:
        t0, t1 = rth_bounds_utc(session_et)
        df = df[(df["event_ts_utc"] >= t0) & (df["event_ts_utc"] <= t1)]
    return df

# ------------------ Plot ------------------
def make_candles_mobile(df: pd.DataFrame, symbol: str, session_et: date,
                        show_vwap=True, show_ema=True, signals: pd.DataFrame | None = None):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.03,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df["ts_plot"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=f"{symbol} {session_et}"
        ),
        row=1, col=1
    )

    if show_vwap and "vwap" in df and not df["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=df["ts_plot"], y=df["vwap"], mode="lines", name="VWAP"), row=1, col=1)

    if show_ema:
        if "ema9" in df:  fig.add_trace(go.Scatter(x=df["ts_plot"], y=df["ema9"],  mode="lines", name="EMA(9)"),  row=1, col=1)
        if "ema21" in df: fig.add_trace(go.Scatter(x=df["ts_plot"], y=df["ema21"], mode="lines", name="EMA(21)"), row=1, col=1)

    if "volume" in df:
        fig.add_trace(go.Bar(x=df["ts_plot"], y=df["volume"], name="Vol"), row=2, col=1)

    # Signals (snap to nearest candle; keep dtype consistent)
    if signals is not None and not signals.empty:
        base = df[["ts_plot", "close"]].dropna().sort_values("ts_plot").copy()
        sig  = signals[["ts_plot","decision","p_up","p_down","p_flat","margin"]].dropna().sort_values("ts_plot").copy()
        base["ts_plot"] = _ensure_dt64ns(base["ts_plot"])
        sig["ts_plot"]  = _ensure_dt64ns(sig["ts_plot"])

        snap = pd.merge_asof(
            sig, base, on="ts_plot",
            direction="nearest",
            tolerance=pd.Timedelta("2min"),
            allow_exact_matches=True
        ).dropna(subset=["close"])

        color_map = {
            "up":"#16a34a","down":"#dc2626","flat":"#64748b",
            "long":"#16a34a","short":"#dc2626",
            "queue_long":"#10b981","queue_short":"#ef4444"
        }
        colors = [color_map.get(str(d), "#22c55e") for d in snap["decision"].astype(str)]

        fig.add_trace(
            go.Scatter(
                x=snap["ts_plot"], y=snap["close"], mode="markers", name="Signals",
                marker=dict(size=10, symbol="circle", line=dict(width=0.5, color="#111"), color=colors),
                customdata=snap[["decision","p_up","p_down","p_flat","margin"]].values,
                hovertemplate=(
                    "ET: %{x}<br>Decision: %{customdata[0]}<br>"
                    "p_up: %{customdata[1]:.3f} · p_down: %{customdata[2]:.3f} · p_flat: %{customdata[3]:.3f}<br>"
                    "margin: %{customdata[4]:.3f}<extra></extra>"
                )
            ),
            row=1, col=1
        )

    fig.update_layout(
        height=640,
        margin=dict(l=6, r=6, t=28, b=6),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ------------------ UI (compact controls) ------------------
topA, topB = st.columns([0.62, 0.38])
with topA:
    db_path = find_latest_snapshot()
    st.caption(f"📦 Snapshot: `{Path(db_path).name}`  ·  🔔 Signals DB: `{Path(SIGNALS_DB).name}`")
with topB:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

symbols = list_symbols(db_path)
if not symbols:
    st.error("No symbols in ohlcv_1m. Check snapshot path.")
    st.stop()

c1, c2 = st.columns([0.68, 0.32])
with c1:
    symbol = st.selectbox("Symbol", symbols, index=0)
with c2:
    default_session = last_session_for_symbol(db_path, symbol) or date.today()
    session_et = st.date_input("Session (ET)", value=default_session)

c3, c4, c5 = st.columns(3)
with c3: show_vwap   = st.toggle("VWAP", True)
with c4: show_ema    = st.toggle("EMA", True)
with c5: include_ah  = st.toggle("Show OOH", True)   # keep OOH signals available

# ---- Data ----
df = load_intraday_rth(db_path, symbol, session_et)
if df.empty:
    st.warning("No RTH data for that session/symbol.")
    st.stop()

# KPIs (compact)
try:
    o = float(df["open"].iloc[0]); c_ = float(df["close"].iloc[-1])
    chg = (c_ / o - 1.0) * 100.0
    k1, k2, k3 = st.columns(3)
    k1.metric("Open", f"{o:.2f}")
    k2.metric("Last", f"{c_:.2f}", f"{chg:+.2f}%")
    k3.metric("Vol", f"{float(df['volume'].fillna(0).sum()):,.0f}")
except Exception:
    pass

signals = load_signals_for_chart(symbol, session_et, include_ah=include_ah)
fig = make_candles_mobile(df, symbol, session_et, show_vwap, show_ema, signals)
st.plotly_chart(fig, use_container_width=True)

# Minimal signal tape (kept, but collapsed by default)
if signals is not None and not signals.empty:
    with st.expander("Signals (today)", expanded=False):
        show = signals[["ts_plot","decision","margin","p_up","p_down","p_flat"]].copy()
        # display ET only (hh:mm:ss)
        show["ts_plot"] = pd.to_datetime(show["ts_plot"], errors="coerce").dt.strftime("%H:%M:%S")
        st.dataframe(
            show.rename(columns={"ts_plot":"ET"}),
            use_container_width=True, hide_index=True
        )
