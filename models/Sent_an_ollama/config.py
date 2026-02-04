# Sentiment Analysis Configuration

# Model Settings
MODEL_NAME = "llama3.2"  # Options: llama3.2, llama3.1, mistral, phi3, or any Ollama model
MODEL_TEMPERATURE = 0.3  # Range: 0.0 (deterministic) to 1.0 (creative)
MODEL_TOP_P = 0.9        # Nucleus sampling parameter

# Sentiment Thresholds
POSITIVE_THRESHOLD = 0.15   # Scores above this are considered positive
NEGATIVE_THRESHOLD = -0.15  # Scores below this are considered negative

# Action Thresholds
BUY_THRESHOLD = 0.5         # Strong buy signal
ACCUMULATE_THRESHOLD = 0.15 # Moderate buy signal
REDUCE_THRESHOLD = -0.15    # Moderate sell signal
SELL_THRESHOLD = -0.5       # Strong sell signal

# Time Windows (in minutes)
SHORT_TERM_WINDOW = 60      # For news_cov_60m and news_sent_mean_60m
MEDIUM_TERM_WINDOW = 1440   # 24 hours for news_sent_std_24h

# Default confidence for error cases
DEFAULT_CONFIDENCE = 0.5

# Sentiment Labels
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_NEUTRAL = "neutral"

# Action Labels
ACTION_BUY = "buy"
ACTION_ACCUMULATE = "accumulate"
ACTION_HOLD = "hold"
ACTION_REDUCE = "reduce"
ACTION_SELL = "sell"

# Prompt Engineering
SENTIMENT_PROMPT = """Analyze the sentiment of this financial news article.

Headline: {headline}
Summary: {summary}

Rate the sentiment on a scale from -1 (very negative) to +1 (very positive).
Consider:
- Company performance indicators
- Financial metrics
- Market perception
- Future outlook

Respond with ONLY a JSON object in this exact format:
{{"score": <number between -1 and 1>, "confidence": <number between 0 and 1>}}

Example responses:
{{"score": 0.8, "confidence": 0.9}} for very positive news
{{"score": -0.6, "confidence": 0.85}} for negative news
{{"score": 0.1, "confidence": 0.7}} for slightly positive news"""

# Logging
VERBOSE_LOGGING = True      # Print detailed progress information
DEBUG_MODE = False          # Additional debug information
