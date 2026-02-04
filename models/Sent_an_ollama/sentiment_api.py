from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Any, Union
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(dotenv_path=(Path(__file__).resolve().parent / ".env"))

# Import the enhanced analyzer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_analyzer_enhanced import NewsAnalyzer

API_VERSION = "1.0.0"
REQUIRED_API_KEY = os.getenv("API_KEY")  # read from .env
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))

# Initialize analyzer (singleton)
analyzer = NewsAnalyzer(model=DEFAULT_MODEL)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Validate API key from environment variable."""
    if REQUIRED_API_KEY and x_api_key != REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class NewsItem(BaseModel):
    """Individual news article."""
    source: Optional[str] = None
    title: str = Field(..., description="Article headline")
    url: str = Field(..., description="Article URL")
    summary: Optional[str] = Field(None, description="Article summary or description")
    description: Optional[str] = None  # Alternative field name
    datetime: str = Field(..., description="Article datetime in ISO format")


class SymbolBlock(BaseModel):
    """News articles for a single stock symbol."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA)")
    asof_utc: Optional[str] = Field(None, description="Timestamp of data collection")
    news: List[NewsItem] = Field(..., description="List of news articles")


class PayloadEnvelope(BaseModel):
    """Wrapper for batch processing multiple symbols."""
    payload: List[SymbolBlock]


class AnalysisConfig(BaseModel):
    """Configuration options for sentiment analysis."""
    min_relevance_confidence: float = Field(
        default=DEFAULT_RELEVANCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for relevance filtering (0-1)"
    )
    include_filtered_articles: bool = Field(
        default=True,
        description="Include filtered articles in response"
    )


app = FastAPI(
    title="Sentiment Analysis API",
    description="Financial news sentiment analysis with relevance filtering using Ollama LLM",
    version=API_VERSION
)

# CORS configuration
allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "Sentiment Analysis API",
        "version": API_VERSION,
        "model": DEFAULT_MODEL,
        "default_relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD,
        "endpoints": {
            "health": "/healthz",
            "analyze": "/analyze (POST)",
            "analyze_single": "/analyze_single (POST)",
            "batch": "/batch (POST)"
        },
        "documentation": "/docs"
    }


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {
        "ok": True,
        "version": API_VERSION,
        "model": DEFAULT_MODEL,
        "status": "ready"
    }


def _asdict(model: BaseModel) -> dict:
    """Convert Pydantic model to dict (v1/v2 compatibility)."""
    return model.dict() if hasattr(model, "dict") else model.model_dump()


@app.post("/analyze")
def analyze_route(
    body: Union[PayloadEnvelope, List[SymbolBlock], Any],
    min_relevance_confidence: float = DEFAULT_RELEVANCE_THRESHOLD,
    _auth=Depends(require_api_key)
) -> Any:
    """
    Analyze sentiment for news articles with relevance filtering.
    
    Accepts:
    - [{"payload": [{"symbol": "AAPL", "news": [...]}]}] (Input.json format)
    - {"payload": [{"symbol": "AAPL", "news": [...]}]} (envelope)
    - [{"symbol": "AAPL", "news": [...]}] (direct array)
    """
    blocks_in = []
    
    # Handle Input.json format: [{"payload": [...]}]
    if isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict) and "payload" in body[0]:
        # Extract from first item's payload
        payload_data = body[0]["payload"]
        blocks_in = [SymbolBlock(**item) for item in payload_data]
    # Handle envelope format: {"payload": [...]}
    elif isinstance(body, dict) and "payload" in body:
        payload_data = body["payload"]
        blocks_in = [SymbolBlock(**item) for item in payload_data]
    # Handle direct array: [{"symbol": ..., "news": ...}]
    elif isinstance(body, list):
        blocks_in = [SymbolBlock(**item) if isinstance(item, dict) else item for item in body]
    # Handle Pydantic-parsed envelope
    elif isinstance(body, PayloadEnvelope):
        blocks_in = body.payload
    else:
        raise HTTPException(status_code=422, detail="Invalid payload format")
    
    # Convert to analyzer format
    input_data = [{
        "payload": [
            {
                "symbol": b.symbol,
                "asof_utc": b.asof_utc or datetime.now(timezone.utc).isoformat(),
                "news": [_asdict(n) for n in b.news]
            }
            for b in blocks_in
        ]
    }]
    
    # Run analysis
    try:
        results = analyzer.analyze_payload(
            input_data,
            min_relevance_confidence=min_relevance_confidence
        )
        
        return {
            "success": True,
            "results": results,
            "config": {
                "model": DEFAULT_MODEL,
                "relevance_threshold": min_relevance_confidence
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze_single")
def analyze_single_route(
    data: dict,
    _auth=Depends(require_api_key)
) -> Any:
    """
    Analyze sentiment for a single symbol.
    
    Body: {"symbol": "AAPL", "news": [...]}
    """
    symbol = data.get("symbol")
    news_list = data.get("news", [])
    
    if not symbol or not news_list:
        raise HTTPException(status_code=422, detail="symbol and news are required")
    
    news = [NewsItem(**n) if isinstance(n, dict) else n for n in news_list]
    block = SymbolBlock(symbol=symbol, news=news)
    return analyze_route(body=[block], _auth=_auth)


@app.post("/batch")
def batch_route(
    body: Union[PayloadEnvelope, List[SymbolBlock], Any],
    min_relevance_confidence: float = DEFAULT_RELEVANCE_THRESHOLD,
    _auth=Depends(require_api_key)
) -> Any:
    """Batch processing endpoint (alias for /analyze)."""
    return analyze_route(body=body, min_relevance_confidence=min_relevance_confidence, _auth=_auth)


@app.get("/config")
def get_config(_auth=Depends(require_api_key)):
    """Get current configuration."""
    return {
        "model": DEFAULT_MODEL,
        "default_relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD,
        "api_version": API_VERSION,
        "features": {
            "relevance_filtering": True,
            "sentiment_analysis": True,
            "batch_processing": True
        }
    }


@app.post("/config")
def update_config(
    model: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    _auth=Depends(require_api_key)
):
    """
    Update runtime configuration.
    
    Note: Model changes require reinitializing the analyzer.
    """
    global analyzer, DEFAULT_MODEL, DEFAULT_RELEVANCE_THRESHOLD
    
    if model and model != DEFAULT_MODEL:
        try:
            analyzer = NewsAnalyzer(model=model)
            DEFAULT_MODEL = model
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize model {model}: {str(e)}"
            )
    
    if relevance_threshold is not None:
        if 0 <= relevance_threshold <= 1:
            DEFAULT_RELEVANCE_THRESHOLD = relevance_threshold
        else:
            raise HTTPException(
                status_code=422,
                detail="Relevance threshold must be between 0 and 1"
            )
    
    return {
        "success": True,
        "model": DEFAULT_MODEL,
        "default_relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8090"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Sentiment Analysis API on {host}:{port}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Default relevance threshold: {DEFAULT_RELEVANCE_THRESHOLD}")
    print(f"API key required: {bool(REQUIRED_API_KEY)}")
    
    uvicorn.run(
        "sentiment_api:app",
        host=host,
        port=port,
        reload=True
    )

    