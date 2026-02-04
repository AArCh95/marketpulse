# n8n Integration Guide

This guide explains how to set up n8n workflows to orchestrate the MarketPulse trading system.

## Overview

n8n acts as the orchestration layer, connecting:
- Market data APIs (Finnhub, Polygon)
- ML inference APIs (CatBoost, Sentiment)
- Trading execution API (Trader server)

## Prerequisites

1. Install n8n (https://n8n.io)
2. Start n8n: `npx n8n`
3. Access UI: http://localhost:5678

## Required Credentials

Configure these in n8n → Settings → Credentials:

### 1. Finnhub API
- **Type**: Header Auth
- **Name**: `finnhub-header`
- **Header Name**: `X-Finnhub-Token`
- **Header Value**: `<your-finnhub-token>`

### 2. Polygon API
- **Type**: Header Auth
- **Name**: `polygon-header`
- **Header Name**: `Authorization`
- **Header Value**: `Bearer <your-polygon-key>`

### 3. CatBoost API
- **Type**: Header Auth
- **Name**: `catboost-header`
- **Header Name**: `X-API-Key`
- **Header Value**: `<CATBOOST_API_KEY from .env>`

### 4. Sentiment API
- **Type**: Header Auth
- **Name**: `sentiment-header`
- **Header Name**: `X-API-Key`
- **Header Value**: `<SENTIMENT_API_KEY from .env>`

### 5. Trader Webhook
- **Type**: Header Auth
- **Name**: `trader-webhook-header`
- **Header Name**: `X-Webhook-Token`
- **Header Value**: `<TRADER_WEBHOOK_TOKEN from .env>`

## Workflow Architecture

### Main Intraday Trading Workflow

**Trigger**: Schedule (every 5 minutes during market hours)

**Flow**:
1. **Finnhub News Node** → Fetch recent news articles
   - Endpoint: `GET /api/v1/company-news`
   - Params: `symbol`, `from` (now-1h), `to` (now)
   - Credential: `finnhub-header`

2. **Polygon Market Data Node** → Fetch 5-minute bars
   - Endpoint: `GET /v2/aggs/ticker/{symbol}/range/5/minute/{from}/{to}`
   - Params: `symbol`, `from`, `to`, `limit=50`
   - Credential: `polygon-header`

3. **Sentiment Analysis Node** → Analyze news sentiment
   - Endpoint: `POST http://localhost:8090/analyze`
   - Body: `{"articles": [{"headline": "...", "summary": "..."}]}`
   - Credential: `sentiment-header`

4. **CatBoost Inference Node** → Get price predictions
   - Endpoint: `POST http://localhost:8001/score_batch`
   - Body: `{"symbols": ["AAPL", "MSFT", ...], "as_of_utc": "2026-02-03T14:30:00Z"}`
   - Credential: `catboost-header`

5. **Merge Node** → Combine sentiment + price predictions
   - Join on `symbol` key

6. **Signal Filter Node** → Apply quality gates
   - Code: Filter by `p_margin > 0.15`, `pred_class != 1` (not flat)

7. **Trader Webhook Node** → Send signals to trader
   - Endpoint: `POST http://localhost:8600/webhook/signal`
   - Body: Signal payload with `symbol`, `pred_class`, `p_margin`, `sentiment_score`
   - Credential: `trader-webhook-header`

### Daily ETL Workflow

**Trigger**: Schedule (06:00 daily)

**Flow**:
1. **Execute Command Node** → Run daily ETL
   - Command: `powershell -File D:\AARCH\DBs\run_daily.ps1`

### Weekly Model Retrain Workflow

**Trigger**: Schedule (Sunday 02:00)

**Flow**:
1. **Execute Command Node** → Retrain intraday model
   - Command: `python D:\AARCH\models\catboost_intraday\retrain_weekly.py`

## Webhook Configuration

### Trader Server Webhook

The trader server expects signals in this format:

```json
{
  "signals": [
    {
      "symbol": "AAPL",
      "pred_class": 2,
      "p_up": 0.45,
      "p_down": 0.10,
      "p_flat": 0.45,
      "p_margin": 0.35,
      "sentiment_score": 0.75,
      "asof_utc": "2026-02-03T14:30:00Z",
      "dist_vwap_bps": 15.2
    }
  ]
}
```

**Authentication**: Include `X-Webhook-Token` header (from trader/.env)

## Error Handling

### Recommended n8n Settings

1. **Retry on Fail**: 3 attempts with exponential backoff
2. **Error Workflow**: Create separate workflow to log failures to DuckDB
3. **Timeout**: 30 seconds for inference APIs
4. **Rate Limiting**:
   - Finnhub: 30 calls/sec (free tier)
   - Polygon: 5 calls/min (free tier)

## Monitoring

### Health Check Workflow

**Trigger**: Schedule (every minute)

**Flow**:
1. **HTTP Request Nodes** → Ping each service
   - CatBoost: `GET http://localhost:8001/health`
   - Sentiment: `GET http://localhost:8090/health`
   - Trader: `GET http://localhost:8600/health`

2. **Alert Node** → Send email/Slack if any service down

## Testing

### Manual Trigger Test

1. Create test workflow with manual trigger
2. Hardcode 1-2 symbols: `["AAPL", "MSFT"]`
3. Run workflow step-by-step
4. Verify:
   - Sentiment API returns scores
   - CatBoost API returns predictions
   - Trader API logs signal (check trader.duckdb)

### Dry Run Mode

Set `ALPACA_PAPER=true` in trader/.env for paper trading.

## Symbol List

Default symbols (~250 stocks) are defined in `DBs/market_sync_yahoo.py:SYMBOLS`.

For n8n workflows, use a subset of liquid stocks:
```json
["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]
```

## Troubleshooting

### Issue: "API key invalid"
- **Solution**: Verify credential configuration in n8n UI
- Check header name matches API expectations

### Issue: "Connection refused"
- **Solution**: Ensure service is running (`ps aux | grep python`)
- Check port matches .env configuration

### Issue: "Stale signals rejected"
- **Solution**: Verify system time is synced (NTP)
- Check `asof_utc` timestamp is within 6 hours

### Issue: "VWAP gate blocked trade"
- **Solution**: Normal behavior during volatile periods
- Adjust `dist_vwap_bps` threshold in trader logic if needed

## Advanced: Cloudflare Tunnel

To expose local services to external n8n instance:

```bash
cloudflared tunnel --url http://localhost:8001  # CatBoost
cloudflared tunnel --url http://localhost:8090  # Sentiment
cloudflared tunnel --url http://localhost:8600  # Trader
```

Update n8n HTTP nodes to use tunnel URLs instead of localhost.

## Security Notes

- **Never commit** n8n workflow JSON with credentials embedded
- Use n8n's credential management system
- Rotate API keys quarterly
- Enable n8n authentication (default: none in local mode)
- Consider IP whitelisting for production deployments
