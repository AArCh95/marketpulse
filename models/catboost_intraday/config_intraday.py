from pathlib import Path

# Paths - NEW intraday model (parallel to existing daily model)
DUCKDB_PATH = r"D:\AARCH\DBs\market.duckdb"
PROJECT_ROOT = Path(r"D:\AARCH\models\catboost_intraday")
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_intraday_5m.cbm"
FEATURE_ORDER_PATH = PROJECT_ROOT / "models" / "feature_order_intraday.json"

# 5-minute horizon for high-frequency trading
HORIZON_MIN = 5                    # 5-minute forward return
UP_THRESH = 0.001                  # +0.10% (tighter for 5m vs daily's 0.20%)
DOWN_THRESH = -0.001               # -0.10%

# Training window
TRAIN_DAYS = 60                    # 60 days of 1-minute bars
TRAIN_SYMBOLS_LIMIT = 50           # Use 50 symbols for fast iteration (C1 approach)
RTH_START = "09:30"                # NYSE RTH start (ET)
RTH_END = "16:00"                  # NYSE RTH end (ET)

# CatBoost params (optimized for larger intraday dataset)
CATBOOST_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    iterations=500,                # Reduced from 1000 (larger dataset)
    learning_rate=0.15,            # Faster learning for more data
    depth=5,                       # Shallower trees (prevent overfitting on 1m bars)
    l2_leaf_reg=5.0,               # Stronger regularization
    bootstrap_type="Bayesian",
    random_seed=42,
    od_type="Iter",
    od_wait=50,                    # Early stopping after 50 iterations
    task_type="GPU",
    devices="0",
    verbose=100,
)

# Feature schema for 1-minute bars (read from ohlcv_1m)
NUM_FEATURES = [
    # Price/returns (computed from 1m bars)
    "close", "high", "low", "open",
    "ret_1m", "ret_5m", "ret_15m", "ret_30m",  # short-term momentum
    "range_pct",                                # (high-low)/close
    
    # Intraday technicals (from ohlcv_1m)
    "dist_vwap_bps",                           # distance to VWAP in bps
    
    # Volume
    "volume",
    "log_volume",
    "vol_ratio_5m",                            # current vol vs recent 5m avg
    
    # Intraday position
    "minute_of_day",                           # 0-389 (minutes since 09:30 ET)
    "minute_of_day_pct",                       # 0-1 normalized
    
    # Volatility
    "rv_5m", "rv_15m", "rv_30m",              # realized vol over windows
    
    # Market context (computed from SPY 1m bars)
    "mkt_ret_5m",                              # SPY 5-minute return
    "mkt_ret_30m",                             # SPY 30-minute return
]

CAT_FEATURES = [
    "symbol",                                  # Stock identifier
    "hour_of_day",                             # Categorical: 9, 10, 11, ..., 15
]

FEATURE_COLUMNS = NUM_FEATURES + CAT_FEATURES
ID_COLS = ["ts"]                               # Timestamp (not fed to model)

CLASS_NAMES = ["down", "flat", "up"]

# Decision rules for serve.py
MARGIN_CUTOFF = 0.10               # Lower for 5m (vs 0.20 for daily)
PROB_GAP = 0.05                    # Tighter gap for higher frequency
SHORT_MIN_PROB = 0.65              # Higher precision for shorts

# Retraining schedule
RETRAIN_DAY = "Sunday"             # Weekly retrain (market closed)
RETRAIN_HOUR = 2                   # 2 AM ET
