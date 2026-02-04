# D:\AARCH\dashboard\trader_dashboard.py
# Portfolio & PnL dashboard (pretty + sortable) using DuckDB snapshots from the trader service.

import os
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------- Config ----------------
TRADER_DB   = os.getenv("TRADER_DB",   r"D:\AARCH\DBs\trader.duckdb")
SIGNALS_DB  = os.getenv("SIGNALS_DB",  r"D:\AARCH\DBs\signals.duckdb")   # optional (for last alerts tab)
APP_TITLE   = "💹 Trader Monitor"
THEME_BG    = "#0f172a"  # slate-900
THEME_ACC   = "#38bdf8"  # sky-400

# Pointer defaults to CURRENT_SNAPSHOT_TRADER.txt
TRADER_SNAPSHOT_POINTER = os.getenv(
    "TRADER_SNAPSHOT_POINTER",
    os.path.join(os.path.dirname(TRADER_DB), "CURRENT_SNAPSHOT_TRADER.txt")
)

st.set_page_config(page_title="Trader Monitor", layout="wide")

# Small CSS polish (tight paddings, non-clipped title)
st.markdown(f"""
<style>
    .block-container {{ padding-top: 0.8rem; padding-bottom: 1rem; }}
    h1, h2, h3 {{ letter-spacing: .2px; }}
    header[data-testid="stHeader"] {{ background: linear-gradient(90deg,{THEME_BG},#111827); }}
    div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)

# ---------------- Snapshot resolver ----------------

def resolve_db_from_pointer(primary_db: str, pointer_file: str) -> str:
    """Return the latest immutable snapshot path from pointer_file if valid; else fallback to primary_db."""
    p = Path(pointer_file)
    if p.exists():
        try:
            snap = p.read_text(encoding="utf-8").strip()
            if snap and Path(snap).exists():
                return snap
        except Exception:
            pass
    return primary_db

DB_TO_USE = resolve_db_from_pointer(TRADER_DB, TRADER_SNAPSHOT_POINTER)

# ---------------- Helpers ----------------
@st.cache_resource
def get_conn(db_path: str):
    p = Path(db_path)
    if not p.exists():
        st.error(f"DB not found: {db_path}")
    return duckdb.connect(str(p), read_only=True)

@st.cache_data(ttl=2)
def now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@st.cache_data(ttl=10)
def today_utc_bounds():
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end

# ---------------- Data loaders ----------------
@st.cache_data(ttl=10)
def load_positions_latest(db_path: str) -> pd.DataFrame:
    con = get_conn(db_path)
    q = """
    WITH latest AS (SELECT max(snapshot_ts) AS ts FROM positions)
    SELECT p.*
    FROM positions p
    JOIN latest l ON p.snapshot_ts = l.ts
    ORDER BY p.symbol
    """
    try:
        df = con.execute(q).df()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce").fillna(0.0)
    for c in ["avg_entry","market_price","market_value","unrealized_pl","unrealized_plpc","realized_pl_day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=10)
def load_orders_today(db_path: str) -> pd.DataFrame:
    con = get_conn(db_path)
    t0, t1 = today_utc_bounds()
    q = """
    SELECT *
    FROM orders
    WHERE created_ts BETWEEN ? AND ?
    ORDER BY created_ts DESC
    """
    try:
        df = con.execute(q, [t0.isoformat(), t1.isoformat()]).df()
    except Exception:
        return pd.DataFrame()
    return df

@st.cache_data(ttl=10)
def load_alerts_today(db_path: str) -> pd.DataFrame:
    con = get_conn(db_path)
    t0, t1 = today_utc_bounds()
    q = """
    SELECT received_ts, symbol, decision, direction, in_rth, change, notify, price, queued_for_open
    FROM alerts
    WHERE received_ts BETWEEN ? AND ?
    ORDER BY received_ts DESC
    """
    try:
        df = con.execute(q, [t0.isoformat(), t1.isoformat()]).df()
    except Exception:
        return pd.DataFrame()
    return df

# ---------------- Derived metrics ----------------

def compute_totals(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return dict(
            invested=0.0, u_pnl=0.0, u_pnl_pc=0.0, r_pnl_day=0.0,
            winners=0, losers=0, symbols=0
        )
    invested = float(pos.get("market_value", pd.Series(dtype=float)).abs().sum())
    u_pnl    = float(pos.get("unrealized_pl", pd.Series(dtype=float)).sum())
    u_pnl_pc = float((u_pnl / invested) * 100.0) if invested > 0 else 0.0
    r_pnl_day = float(pos.get("realized_pl_day", pd.Series(dtype=float)).sum())
    winners = int((pos.get("unrealized_pl", pd.Series(dtype=float)) > 0).sum())
    losers  = int((pos.get("unrealized_pl", pd.Series(dtype=float)) < 0).sum())
    return dict(
        invested=invested, u_pnl=u_pnl, u_pnl_pc=u_pnl_pc,
        r_pnl_day=r_pnl_day, winners=winners, losers=losers,
        symbols=len(pos)
    )


def make_pnl_table(pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty: return pos
    df = pos.copy()
    # per-symbol unrealized % based on market value vs cost basis (approx)
    # use avg_entry and market_price
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret_pct"] = (df["market_price"] / df["avg_entry"] - 1.0) * 100.0
    # friendly columns
    cols = [
        "symbol","qty","avg_entry","market_price","market_value",
        "unrealized_pl","ret_pct"
    ]
    existing = [c for c in cols if c in df.columns]
    out = df[existing].sort_values("unrealized_pl", ascending=False)
    return out

# ---------------- UI ----------------

topL, topR = st.columns([0.7, 0.3])
with topL:
    st.caption(f"pointer: `{TRADER_SNAPSHOT_POINTER}`")
    st.caption(f"DB (read-only): `{DB_TO_USE}`")
    st.caption(f"UTC now: {now_utc_str()}")
with topR:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


tab_port, tab_activity = st.tabs(["📊 Portfolio", "🧾 Activity"])

with tab_port:
    pos = load_positions_latest(DB_TO_USE)

    totals = compute_totals(pos)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Invested", f"${totals['invested']:,.0f}")
    k2.metric("Unrealized P&L", f"${totals['u_pnl']:,.0f}", f"{totals['u_pnl_pc']:+.2f}%")
    k3.metric("Realized P&L (today)", f"${totals['r_pnl_day']:,.0f}")
    k4.metric("Symbols (W / L)", f"{totals['symbols']}  ({totals['winners']} / {totals['losers']})")

    st.subheader("By symbol")
    tbl = make_pnl_table(pos)
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "qty":         st.column_config.NumberColumn("qty", format="%.0f"),
            "avg_entry":   st.column_config.NumberColumn("avg_entry", format="%.2f"),
            "market_price":st.column_config.NumberColumn("last", format="%.2f"),
            "market_value":st.column_config.NumberColumn("value", format="%.0f"),
            "unrealized_pl": st.column_config.NumberColumn("uPnL", format="%.0f"),
            "ret_pct":     st.column_config.NumberColumn("uPnL %", format="%.2f"),
        }
    )

    if not tbl.empty:
        st.subheader("Winners / Losers (unrealized $)")
        # top 10 by magnitude
        mag = tbl.assign(mag=tbl["unrealized_pl"].abs()).nlargest(10, "mag")
        fig = px.bar(
            mag.sort_values("unrealized_pl"),
            x="unrealized_pl", y="symbol",
            orientation="h",
            color=(mag["unrealized_pl"] > 0),
            color_discrete_map={True:"#16a34a", False:"#dc2626"},
            labels={"unrealized_pl":"$ uPnL", "symbol":"symbol", "color":""},
            title=""
        )
        fig.update_layout(height=420, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_activity:
    st.subheader("Latest orders")
    orders = load_orders_today(DB_TO_USE)
    if orders.empty:
        st.info("No orders today.")
    else:
        st.dataframe(orders, use_container_width=True, hide_index=True)

    st.subheader("Latest alerts (gate decisions)")
    alerts = load_alerts_today(DB_TO_USE)
    if alerts.empty:
        st.info("No alerts today.")
    else:
        # pretty checkboxes
        for c in ["in_rth","change","notify","queued_for_open"]:
            if c in alerts.columns:
                alerts[c] = alerts[c].astype(bool)
        st.dataframe(alerts, use_container_width=True, hide_index=True)
