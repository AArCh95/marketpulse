# MarketPulse 📈

> **Production-grade algorithmic trading system** leveraging machine learning, large language models, and real-time market data for automated trading decisions.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

MarketPulse is a **sophisticated algorithmic trading platform** that combines cutting-edge machine learning techniques with natural language processing to execute data-driven trades. Built for scalability and reliability, it handles real-time market data ingestion, predictive modeling, sentiment analysis, and automated order execution—all while maintaining strict risk management protocols.

**Key Achievement**: Processes 50+ symbols with 1-minute granularity, generating ML predictions every 5 minutes during market hours, with full position tracking and PnL monitoring.

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- **Custom CatBoost Intraday Model**: Predicts 5-minute price movements using 30+ engineered features
- **Real-time Feature Engineering**: Rolling volatility, momentum indicators, VWAP distance calculations
- **GPU-Accelerated Training**: 10x speedup for model retraining on large datasets (780K+ bars)
- **Class Imbalance Handling**: Tilted loss functions to boost recall on minority classes

### 💬 Sentiment Analysis Engine
- **LLM-Powered News Analysis**: Local Ollama integration (llama3.2) for financial sentiment scoring
- **Multi-Source News Aggregation**: Finnhub, Polygon.io APIs for real-time market news
- **Relevance Filtering**: Confidence-based filtering to eliminate noise from unrelated articles
- **Sentiment Weighting**: Combines sentiment signals with technical predictions for final trade decisions

### 📊 Trading Execution System
- **Alpaca API Integration**: Paper and live trading support with fractional shares
- **Dynamic Position Sizing**: Confidence-based position sizing with configurable risk limits
- **Risk Management**: Automated take-profit (3%), stop-loss (2%), max position weight (0.4%)
- **Trade Orchestration**: n8n workflow automation for signal generation and execution
- **Cooldown Logic**: Per-symbol throttling to prevent overtrading
- **VWAP Gates**: Liquidity-based trade filtering to reduce slippage

### 💾 Data Infrastructure
- **Snapshot-Based Architecture**: Solves Windows file-locking issues with immutable DuckDB snapshots
- **Dual Database Strategy**: Separate DBs for market data and trading state
- **ETL Pipeline**: Automated daily and intraday data ingestion from Yahoo Finance
- **Historical Analysis**: Stores 60+ days of 1-minute bars for backtesting and retraining

### 📈 Monitoring & Visualization
- **Real-time Dashboards**: Streamlit interfaces for portfolio monitoring and signal inspection
- **Trade History Tracking**: Complete audit trail of all order decisions and executions
- **Performance Metrics**: Unrealized/realized PnL, position-level P&L attribution
- **Signal Quality Monitoring**: Confidence distributions, prediction margins, feature importance

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MarketPulse System                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Yahoo Finance│────▶│  ETL Pipeline │────▶│  DuckDB      │
│ Finnhub      │     │  (Python)     │     │  (Snapshots) │
│ Polygon.io   │     └──────────────┘     └──────┬───────┘
└──────────────┘                                   │
                                                   ▼
        ┌──────────────────────────────────────────────────┐
        │         Machine Learning Layer                   │
        ├──────────────────────────────────────────────────┤
        │  • CatBoost Intraday (5m predictions)           │
        │  • Feature Engineering (30+ indicators)          │
        │  • Sentiment Analysis (Ollama LLM)              │
        └──────────────────┬───────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Signal Generation (n8n)                  │
        │  • Merge predictions + sentiment                 │
        │  • Apply filters (VWAP, margin, staleness)      │
        └──────────────────┬───────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Trader Server (FastAPI)                  │
        │  • Position sizing                               │
        │  • Risk management                               │
        │  • Order execution (Alpaca)                      │
        └──────────────────┬───────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Monitoring (Streamlit)                   │
        │  • Portfolio dashboard                           │
        │  • Signal visualization                          │
        │  • Performance analytics                         │
        └──────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer              | Technologies                                      |
|--------------------|--------------------------------------------------|
| **Languages**      | Python 3.13                                      |
| **ML Framework**   | CatBoost (GPU-accelerated)                       |
| **LLM**            | Ollama (llama3.2)                               |
| **Database**       | DuckDB (OLAP with snapshot architecture)         |
| **APIs**           | FastAPI (async services)                         |
| **Trading**        | Alpaca API (paper/live trading)                  |
| **Orchestration**  | n8n (workflow automation)                        |
| **Dashboards**     | Streamlit (real-time monitoring)                 |
| **Data Sources**   | Yahoo Finance, Finnhub, Polygon.io              |
| **Scheduler**      | Windows Task Scheduler                           |
| **Version Control**| Git, GitHub                                      |

---

## 🚀 Performance Highlights

- **Latency**: Sub-200ms inference time for intraday model predictions
- **Throughput**: Handles 50+ symbols with 1-minute data updates
- **Model Training**: 30-60 minutes for 60 days of intraday data (780K bars)
- **Feature Engineering**: <10ms snapshot reads for feature hydration
- **Scalability**: Snapshot architecture eliminates database contention on Windows
- **Reliability**: Automated ETL with error handling and logging

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Installation guide, prerequisites, and deployment instructions |
| [CLAUDE.md](CLAUDE.md) | Comprehensive system architecture and design patterns |
| [docs/n8n_integration.md](docs/n8n_integration.md) | Workflow orchestration setup and configuration |

---

## 🎓 Technical Highlights for Resume

- **End-to-End ML Pipeline**: Data ingestion → feature engineering → model training → inference → execution
- **Production System Design**: Snapshot-based concurrency, error handling, monitoring, logging
- **API Development**: RESTful FastAPI services with authentication and async operations
- **LLM Integration**: Local inference with prompt engineering for financial sentiment analysis
- **Risk Management**: Multi-layered gates (VWAP, confidence, position limits, cooldowns)
- **Data Engineering**: ETL pipeline processing millions of bars, DuckDB optimization
- **DevOps**: Automated scheduling, environment management, configuration as code

---

## 🏃 Quick Start

### Prerequisites
- Python 3.13
- DuckDB CLI
- Ollama (https://ollama.ai)
- Alpaca Paper Trading Account
- n8n (for orchestration)

### Installation

```bash
git clone https://github.com/AArCh95/marketpulse-.git
cd marketpulse-

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for each service
cd models/catboost_intraday
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
cd ../../DBs
python market_sync_yahoo.py --duckdb market.duckdb --mode daily

# Train model
cd ../models/catboost_intraday
python train_intraday.py

# Start services
python serve_intraday.py  # Port 8001
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

## 📊 Project Structure

```
marketpulse/
├── DBs/                          # Database files and ETL scripts
│   ├── market_sync_yahoo.py      # Main ETL pipeline
│   ├── run_daily.ps1             # Daily data sync
│   └── run_intraday.ps1          # Intraday data sync
├── models/
│   ├── catboost_intraday/        # 5-minute intraday model
│   │   ├── train_intraday.py     # Model training
│   │   ├── serve_intraday.py     # Inference API
│   │   └── features_intraday.py  # Feature engineering
│   ├── catboost_core/            # Daily model (legacy)
│   └── Sent_an_ollama/           # Sentiment analysis API
├── trader/
│   ├── server.py                 # Trading execution service
│   └── trader_dashboard.py       # Portfolio monitoring UI
├── dashboard/
│   ├── app.py                    # Signal visualization
│   └── db_app.py                 # Database query interface
└── docs/
    └── n8n_integration.md        # Workflow setup guide
```

---

## 🔒 Security & Best Practices

- ✅ **Environment Variables**: All secrets managed via `.env` files (gitignored)
- ✅ **API Authentication**: API keys required for all service endpoints
- ✅ **Git Hygiene**: Comprehensive `.gitignore` prevents secret leaks
- ✅ **Paper Trading First**: Default configuration uses Alpaca paper trading
- ✅ **Input Validation**: Pydantic schemas for all API payloads
- ✅ **Error Handling**: Comprehensive exception handling and logging

---

## 📈 Future Enhancements

- [ ] Multi-timeframe predictions (1m, 5m, 15m, 1h)
- [ ] Portfolio optimization using Modern Portfolio Theory
- [ ] Reinforcement learning for dynamic risk adjustment
- [ ] Options trading strategies
- [ ] Real-time performance attribution analytics
- [ ] Backtesting framework with Monte Carlo simulations
- [ ] Docker containerization for simplified deployment

---

## 🤝 Contributing

This is a portfolio/educational project, but feedback and suggestions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share ideas for enhancements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Aaron Josue Arce Chacon**

---

## ⚠️ Disclaimer

**For educational and research purposes only.** This software is not financial advice. Algorithmic trading carries substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. The author is not responsible for any financial losses incurred from using this software.

**Always test strategies thoroughly in paper trading before risking real capital.**

---

## 📫 Contact

**Aaron Josue Arce Chacon**
- GitHub: [@AArCh95](https://github.com/AArCh95)
- Repository: [marketpulse-](https://github.com/AArCh95/marketpulse)
- Linkedin: [Aaron Arce](https://www.linkedin.com/in/aar%C3%B3n-arce-a71079277/)

---

<div align="center">

**⭐ If you find this project interesting, please consider starring it! ⭐**

*Built with passion for quantitative finance and machine learning.*

</div>
