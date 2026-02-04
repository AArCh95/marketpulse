# Sentiment Analysis API

## Setup

```bash
# 1. Install dependencies
pip install fastapi uvicorn python-dotenv pydantic ollama

# 2. Configure
cp .env.example .env
# Edit .env and set API_KEY

# 3. Run
python sentiment_api.py
```

## API Usage

### POST /analyze

**Accepts any of these formats:**

```json
// Format 1: Input.json format (array with payload wrapper)
[{"payload": [{"symbol": "AAPL", "news": [...]}]}]

// Format 2: Direct payload envelope
{"payload": [{"symbol": "AAPL", "news": [...]}]}

// Format 3: Direct array
[{"symbol": "AAPL", "news": [...]}]
```

**curl example:**
```bash
curl -X POST http://localhost:8090/analyze \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d @Input.json
```

**Query parameter:**
- `?min_relevance_confidence=0.7` - Adjust filtering (0.5-0.7, default: 0.6)

## n8n Integration

**HTTP Request Node:**
- **Method:** POST
- **URL:** `http://localhost:8090/analyze` (or `https://sentiment-api.aarch.shop/analyze`)
- **Authentication:** None
- **Headers:**
  - Name: `X-API-Key`, Value: `{{your_api_key}}`
  - Name: `Content-Type`, Value: `application/json`
- **Body:** JSON
  - Use your news data in any of the formats above

**Response:**
```json
{
  "success": true,
  "results": [{
    "symbol": "AAPL",
    "sentiment": {"score": 0.75, "label": "positive", "confidence": 0.92},
    "recommended_action": "buy",
    "articles_analyzed": 5,
    "articles_filtered": 2,
    "filtered_articles": [...]
  }],
  "config": {"model": "llama3.2", "relevance_threshold": 0.6}
}
```

## Cloudflare Tunnel

**Setup:**
```bash
# Create tunnel
cloudflared tunnel create sentiment-api

# Edit C:\Users\Aaron\.cloudflared\config_sentiment.yml:
# - Set tunnel: YOUR_TUNNEL_ID
# - Set credentials-file: C:\Users\Aaron\.cloudflared\YOUR_TUNNEL_ID.json

# Route DNS  
cloudflared tunnel route dns sentiment-api sentiment-api.aarch.shop

# Run tunnel
cloudflared tunnel --config C:\Users\Aaron\.cloudflared\config_sentiment.yml run sentiment-api
```

**Public URL:** `https://sentiment-api.aarch.shop/analyze`

## Testing

```bash
python test_api.py
```