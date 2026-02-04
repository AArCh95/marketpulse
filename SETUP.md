# Setup Guide

## Prerequisites
- Python 3.13
- DuckDB CLI
- Ollama (https://ollama.ai)
- n8n (https://n8n.io) - for orchestration
- Alpaca Paper Trading Account (https://alpaca.markets)

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/<your-username>/marketpulse.git
   cd marketpulse
   ```

2. Install dependencies:
   ```bash
   # Each service has its own venv
   cd models/catboost_intraday
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

   # Repeat for: trader/, dashboard/, models/Sent_an_ollama/
   ```

3. Configure environment:
   ```bash
   # Copy .env.example to .env for each service
   cp .env.example .env
   # Edit .env files with your credentials
   ```

4. Initialize databases:
   ```bash
   # Run daily ETL once to create tables
   cd DBs
   python market_sync_yahoo.py --duckdb market.duckdb --mode daily
   ```

5. Train models:
   ```bash
   cd models/catboost_intraday
   python train_intraday.py
   ```

6. Start services:
   ```bash
   # Terminal 1: CatBoost inference
   cd models/catboost_intraday
   .venv\Scripts\activate
   python serve_intraday.py

   # Terminal 2: Sentiment API
   cd models/Sent_an_ollama
   .venv\Scripts\activate
   python sentiment_api.py

   # Terminal 3: Trader API
   cd trader
   venv\Scripts\activate
   python -m uvicorn server:app --host 127.0.0.1 --port 8600

   # Terminal 4: Dashboard (optional)
   cd trader
   streamlit run trader_dashboard.py
   ```

7. Configure n8n:

   See [docs/n8n_integration.md](docs/n8n_integration.md)

## Windows Task Scheduler (Optional)

Schedule daily/intraday ETL:

- Daily: 06:00 CR time → `powershell -File DBs\run_daily.ps1`
- Intraday: 07:25 CR time → `powershell -File DBs\run_intraday.ps1`
