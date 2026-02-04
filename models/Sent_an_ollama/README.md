# Financial News Sentiment Analysis with Ollama

A sentiment analysis system that processes financial news articles and generates sentiment scores, confidence levels, and trading recommendations using Ollama's local LLM models.

## Features

- **Real-time Sentiment Analysis**: Analyzes news headlines and summaries using Ollama
- **Confidence Scoring**: Provides confidence levels for each sentiment prediction
- **Trading Recommendations**: Generates actionable recommendations (buy, accumulate, hold, reduce, sell)
- **Aggregate Metrics**: Calculates overall sentiment and various news metrics
- **Flexible Model Selection**: Works with any Ollama model (default: llama3.2)

## Prerequisites

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and install from [ollama.ai](https://ollama.ai/download)

### 2. Pull a Language Model

```bash
# Recommended: Llama 3.2 (3B parameters - fast and accurate)
ollama pull llama3.2

# Alternatives:
ollama pull llama3.1      # Larger model (8B parameters)
ollama pull mistral       # Alternative model
ollama pull phi3          # Smaller model (3.8B parameters)
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install ollama
```

## Usage

### Basic Usage

```bash
python sentiment_analyzer.py Input.json Output.json
```

### Using a Different Model

Edit the `sentiment_analyzer.py` file and change the model parameter:

```python
analyzer = NewsAnalyzer(model="llama3.1")  # or "mistral", "phi3", etc.
```

### Command Line Arguments

```bash
python sentiment_analyzer.py <input_file> [output_file]
```

- `input_file`: Path to your JSON input file (required)
- `output_file`: Path for the output file (optional, defaults to "sentiment_output.json")

## Input Format

The input JSON should follow this structure:

```json
[
  {
    "payload": [
      {
        "symbol": "AMZN",
        "asof_utc": "2025-11-02T16:44:18.512-06:00",
        "news": [
          {
            "source": "Yahoo",
            "title": "Article headline",
            "url": "https://example.com/article",
            "description": "Article summary or description",
            "datetime": "2025-11-02T17:00:00.000Z"
          }
        ]
      }
    ]
  }
]
```

## Output Format

The system generates output with the following structure:

```json
[
  {
    "symbol": "AMZN",
    "sentiment": {
      "score": 0.28,
      "label": "positive",
      "confidence": 0.944
    },
    "recommended_action": "accumulate",
    "top_news": {
      "headline": "Most recent article headline",
      "url": "https://...",
      "source": "Source name",
      "summary": "Article summary",
      "datetime": "2025-11-02T19:24:00Z",
      "datetime_iso": "2025-11-02T19:24:00Z",
      "datetime_epoch": 1762111440000
    },
    "articles_scored": [
      {
        "headline": "Article headline",
        "summary": "Article summary",
        "url": "https://...",
        "source": "Source",
        "datetime": "2025-11-02T19:24:00Z",
        "datetime_iso": "2025-11-02T19:24:00Z",
        "datetime_epoch": 1762111440000,
        "score": 1.0,
        "label": "positive",
        "confidence": 1.0
      }
    ],
    "news_cov_60m": 0,
    "news_sent_mean_60m": null,
    "news_uniqueness_24h": 1,
    "news_sent_std_24h": 0.734,
    "news_age_min": 200
  }
]
```

## Understanding the Output

### Sentiment Scores
- **score**: Range from -1 (very negative) to +1 (very positive)
- **label**: `positive`, `negative`, or `neutral`
- **confidence**: 0 to 1, indicating prediction confidence

### Recommended Actions
- **buy**: Strong positive sentiment (score > 0.5)
- **accumulate**: Moderate positive sentiment (0.15 < score ≤ 0.5)
- **hold**: Neutral sentiment (-0.15 ≤ score ≤ 0.15)
- **reduce**: Moderate negative sentiment (-0.5 ≤ score < -0.15)
- **sell**: Strong negative sentiment (score < -0.5)

### Metrics
- **news_cov_60m**: Number of articles in the last 60 minutes
- **news_sent_mean_60m**: Average sentiment in the last 60 minutes
- **news_uniqueness_24h**: Ratio of unique articles in 24 hours
- **news_sent_std_24h**: Standard deviation of sentiment in 24 hours
- **news_age_min**: Minutes since the most recent article

## Customization

### Adjusting Sentiment Thresholds

Edit the `determine_action()` method to customize trading recommendations:

```python
def determine_action(self, sentiment_score: float, sentiment_label: str) -> str:
    if sentiment_label == "positive" and sentiment_score > 0.6:  # Changed from 0.5
        return "buy"
    # ... adjust other thresholds
```

### Changing Model Temperature

Lower temperature = more consistent results
Higher temperature = more creative/varied results

```python
response = ollama.generate(
    model=self.model,
    prompt=prompt,
    options={
        "temperature": 0.3,  # Adjust between 0 and 1
        "top_p": 0.9,
    }
)
```

### Using a Custom Prompt

Modify the `sentiment_prompt_template` in the `__init__` method to customize how the model analyzes sentiment.

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.2
```

### Slow Performance
- Use a smaller model: `phi3` or `llama3.2`
- Reduce the number of articles processed
- Use GPU acceleration if available

### Connection Errors
Ensure Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

## Performance Tips

1. **Model Selection**:
   - `llama3.2`: Best balance of speed and accuracy (recommended)
   - `phi3`: Fastest, good for testing
   - `llama3.1`: Most accurate, slower

2. **Batch Processing**:
   The system processes articles sequentially. For large datasets, consider:
   - Processing during off-peak hours
   - Using a more powerful GPU
   - Implementing parallel processing (advanced)

3. **Caching**:
   Results are not cached by default. For repeated analyses, consider implementing a caching layer.

## Examples

### Analyze a Single File
```bash
python sentiment_analyzer.py news_data.json results.json
```

### Use with Default Output Name
```bash
python sentiment_analyzer.py Input.json
# Creates sentiment_output.json
```

### Process Multiple Symbols
The system automatically processes all symbols in the input payload.

## Integration

### As a Python Module

```python
from sentiment_analyzer import NewsAnalyzer

# Initialize
analyzer = NewsAnalyzer(model="llama3.2")

# Analyze
results = analyzer.analyze_payload(input_data)

# Access results
for result in results:
    print(f"{result['symbol']}: {result['sentiment']['label']}")
```

### API Integration

You can wrap this in a Flask or FastAPI application:

```python
from flask import Flask, request, jsonify
from sentiment_analyzer import NewsAnalyzer

app = Flask(__name__)
analyzer = NewsAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    results = analyzer.analyze_payload(data)
    return jsonify(results)
```

## License

This project is provided as-is for financial news sentiment analysis.

## Support

For issues related to:
- **Ollama**: Visit [ollama.ai](https://ollama.ai)
- **Models**: Check model documentation on [ollama.ai/library](https://ollama.ai/library)

## Version History

- **v1.0**: Initial release with Ollama integration
  - Support for multiple LLM models
  - Comprehensive sentiment analysis
  - Trading recommendations
  - News metrics calculation
