# MarketPulse n8n Workflow

This directory contains the n8n workflow configuration for orchestrating the MarketPulse trading system.

## Overview

The workflow automates the complete trading pipeline:
1. Market data ingestion (Finnhub, Polygon)
2. ML model inference (CatBoost)  
3. Sentiment analysis (Ollama LLM)
4. Signal generation and filtering
5. Trade execution (Alpaca API)

## File

- **MarketPulse_n8n_workflow_SANITIZED.json**: Sanitized workflow backup (all API keys removed)

## Setup Instructions

### 1. Import Workflow

1. Open your n8n instance
2. Go to Workflows → Import from File
3. Select `MarketPulse_n8n_workflow_SANITIZED.json`
4. The workflow will be imported with placeholder credentials

### 2. Configure Credentials

You need to create credentials for these services in n8n:

#### Finnhub API
- Type: HTTP Header Auth
- Header Name: `X-Finnhub-Token`
- Header Value: `<your-finnhub-token>`

#### Polygon API  
- Type: HTTP Header Auth
- Header Name: `Authorization`
- Header Value: `Bearer <your-polygon-api-key>`

#### Alpaca API
- Type: Header Auth (custom)
- Headers:
  - `APCA-API-KEY-ID`: `<your-alpaca-key-id>`
  - `APCA-API-SECRET-KEY`: `<your-alpaca-secret>`

#### Internal Services (No auth needed if running locally)
- CatBoost API: `http://localhost:8001`
- Sentiment API: `http://localhost:8090`
- Trader API: `http://localhost:8600`

### 3. Update Placeholders

After import, find and replace these placeholders:

- `YOUR_FINNHUB_TOKEN_HERE` → Your actual Finnhub token
- `YOUR_POLYGON_API_KEY_HERE` → Your actual Polygon API key

### 4. Update Service URLs

If your services are not on localhost, update the URLs in HTTP Request nodes:
- CatBoost service
- Sentiment service  
- Trader service

## Workflow Structure

```
┌─────────────┐
│   Schedule  │ (Every 5 minutes during market hours)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Fetch Market Data (Parallel)       │
│  • Finnhub: Real-time quotes        │
│  • Polygon: News articles           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Sentiment Analysis (Ollama)        │
│  • Parse news content               │
│  • Score sentiment (-1 to +1)       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  ML Prediction (CatBoost)           │
│  • Prepare features                 │
│  • Get price predictions            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Signal Generation                  │
│  • Merge predictions + sentiment    │
│  • Apply filters (VWAP, margin)     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Trade Execution (Trader API)       │
│  • Size positions                   │
│  • Execute via Alpaca               │
└─────────────────────────────────────┘
```

## Customization

### Adjust Trading Hours

Modify the Schedule trigger to match your trading hours:
- Default: Mon-Fri, 09:30-16:00 ET
- Interval: Every 5 minutes

### Change Symbol Universe

Edit the symbols list in the initial node to track different stocks.

### Modify Filters

Adjust signal filtering criteria:
- `p_margin` threshold (default: 0.15)
- VWAP distance gate (default: 50 bps)
- Staleness window (default: 6 hours)

## Troubleshooting

### "Connection refused" errors
- Ensure all services are running (CatBoost, Sentiment, Trader)
- Check service ports match configuration

### "Invalid API key"  
- Verify credentials are correctly set in n8n
- Check API keys are not expired

### "Stale signals rejected"
- Verify system time is synced (NTP)
- Check `asof_utc` timestamp generation

## Security Notes

⚠️ **Important**:
- Never commit the original `MarketPulse n8n structure.json` with real API keys
- Use n8n's credential manager (encrypted storage)
- Rotate API keys regularly
- Use environment variables for sensitive config

## Support

For detailed architecture and setup:
- See [../CLAUDE.md](../CLAUDE.md)
- See [../docs/n8n_integration.md](../docs/n8n_integration.md)

---

**Note**: This is a sanitized backup. Your actual n8n instance should use the credential manager, not hardcoded keys.
