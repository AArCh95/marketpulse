from pathlib import Path

# Paths
DUCKDB_PATH = r"D:\AARCH\DBs\market.duckdb"
PROJECT_ROOT = Path(r"D:\AARCH\models\catboost_core")
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_core.cbm"
FEATURE_ORDER_PATH = PROJECT_ROOT / "models" / "feature_order.json"

# Prefer 5-minute horizon (train.py auto-detects minute forward-return columns like fwd_ret_5m)
HORIZON_MIN = 5                    # preferred minute horizon (used by your ETL to create fwd_ret_5m)
HORIZON_D = 5                      # 5-day forward horizon for swing trading
UP_THRESH = 0.002                  # +0.20%  (tune: 0.2–0.3% works better for 5m)
DOWN_THRESH = -0.002               # -0.20%


# CatBoost params (GPU with safe fallback)
CATBOOST_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="TotalF1",     # better for imbalanced multi-class
    iterations=1000,           # pair with early stopping
    learning_rate=0.1,         # faster convergence (we use od_wait)
    depth=6,
    l2_leaf_reg=4.0,
    bootstrap_type="Bayesian",
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    task_type="GPU",           # will work if GPU available; our train.py wrapper keeps params as-is
    devices="0",
    verbose=200,
)


# Columns (keep in sync with your n8n payload)
NUM_FEATURES = [
    "close_d","ret_1d","ret_5d","high_d","low_d","beta_60d",
    "mkt_ret_1d","sector_ret_1d","rel_sector_vs_mkt",
    "news_cov_60m","news_sent_mean_60m","news_uniqueness_24h","news_sent_std_24h","news_age_min",
    "rsi_14","macd_line","macd_signal","macd_hist","dist_vwap_bps",
    "log_volume","chg_pct_d","dist_to_52w_high_pct","dist_to_52w_low_pct",
    "ytd_ret_ratio","price_rel_spx_13w_pct"
]
CAT_FEATURES = [
    "mkt_etf","sector_etf","sector","industry","exchange","alertType","severity"
]

FEATURE_COLUMNS = NUM_FEATURES + CAT_FEATURES    # <-- fed to CatBoost (train & serve)
ID_COLS = ["symbol", "ts", "session"]            # <-- NOT fed to CatBoost (train.py uses 'ts')

# Do NOT include IDs in feature order. train.py saves EXACT inference order = FEATURE_COLUMNS.
# FEATURE_ORDER is intentionally removed to avoid leaking IDs at serve time.

CLASS_NAMES = ["down", "flat", "up"]

FEATURE_ORDER = ID_COLS + NUM_FEATURES + CAT_FEATURES


# (Optional) centralize serve-side decision rules (used by serve.py if you like)
MARGIN_CUTOFF = 0.20     # you saw ~66–71% with this
PROB_GAP = 0.10          # keep as-is unless you want fewer trades (then 0.12)
SHORT_MIN_PROB = 0.60    # ↑ from 0.55 to keep shorts higher precision

