# schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, RootModel  # <-- add RootModel

class InferRow(BaseModel):
    symbol: str
    asof_utc: str
    session: str
    # numeric
    close_d: Optional[float]=None
    ret_1d: Optional[float]=None
    ret_5d: Optional[float]=None
    high_d: Optional[float]=None
    low_d: Optional[float]=None
    beta_60d: Optional[float]=None
    mkt_ret_1d: Optional[float]=None
    sector_ret_1d: Optional[float]=None
    rel_sector_vs_mkt: Optional[float]=None
    news_cov_60m: Optional[float]=None
    news_sent_mean_60m: Optional[float]=None
    news_uniqueness_24h: Optional[float]=None
    news_sent_std_24h: Optional[float]=None
    news_age_min: Optional[float]=None
    rsi_14: Optional[float]=None
    macd_line: Optional[float]=None
    macd_signal: Optional[float]=None
    macd_hist: Optional[float]=None
    dist_vwap_bps: Optional[float]=None
    log_volume: Optional[float]=None
    chg_pct_d: Optional[float]=None
    dist_to_52w_high_pct: Optional[float]=None
    dist_to_52w_low_pct: Optional[float]=None
    ytd_ret_ratio: Optional[float]=None
    price_rel_spx_13w_pct: Optional[float]=None
    # categoricals
    mkt_etf: Optional[str]=None
    sector_etf: Optional[str]=None
    sector: Optional[str]=None
    industry: Optional[str]=None
    exchange: Optional[str]=None
    alertType: Optional[str]=None
    severity: Optional[str]=None

class InferJob(BaseModel):
    mode: Literal["infer"]
    schema_version: int
    rows: List[InferRow]

# ✅ Pydantic v2-compliant root model
class InferBatch(RootModel[List[InferJob]]):
    pass
