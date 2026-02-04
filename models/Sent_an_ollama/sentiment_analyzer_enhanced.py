#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Model using Ollama
Processes financial news articles with relevance filtering and generates sentiment scores.
"""

import json
import ollama
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import statistics


class NewsAnalyzer:
    def __init__(self, model: str = "llama3.2"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model: Ollama model to use (default: llama3.2)
        """
        self.model = model
        
        # Relevance checking prompt
        self.relevance_prompt_template = """Determine if this financial news article is primarily about {symbol}.

Headline: {headline}
Summary: {summary}

The article should be considered relevant ONLY if:
- The company {symbol} is the primary subject of the article
- The article discusses {symbol}'s financial performance, products, or business operations
- The article provides specific information that would impact {symbol} investors

The article should be considered NOT relevant if:
- {symbol} is only mentioned in passing or in a list with other companies
- The article is primarily about another company, with {symbol} mentioned secondarily
- The article is about general market conditions or indices where {symbol} is mentioned incidentally

Respond with ONLY a JSON object in this exact format:
{{"relevant": <true or false>, "confidence": <number between 0 and 1>, "primary_subject": "<main company/topic discussed>"}}

Examples:
{{"relevant": true, "confidence": 0.95, "primary_subject": "{symbol}"}}
{{"relevant": false, "confidence": 0.9, "primary_subject": "AAPL"}}
{{"relevant": false, "confidence": 0.85, "primary_subject": "Market Overview"}}"""

        # Sentiment analysis prompt
        self.sentiment_prompt_template = """Analyze the sentiment of this financial news article about {symbol}.

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

    def check_relevance(self, symbol: str, headline: str, summary: str) -> Dict:
        """
        Check if an article is relevant to the target symbol.
        
        Args:
            symbol: Stock symbol being analyzed
            headline: Article headline
            summary: Article summary/description
            
        Returns:
            Dictionary with relevance, confidence, and primary_subject
        """
        prompt = self.relevance_prompt_template.format(
            symbol=symbol,
            headline=headline,
            summary=summary
        )
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.2,  # Lower temperature for more consistent relevance checks
                    "top_p": 0.9,
                }
            )
            
            # Extract JSON from response
            response_text = response['response'].strip()
            
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                relevant = bool(result.get('relevant', False))
                confidence = float(result.get('confidence', 0.5))
                primary_subject = result.get('primary_subject', 'Unknown')
                
                # Ensure confidence is in valid range
                confidence = max(0, min(1, confidence))
                
                return {
                    "relevant": relevant,
                    "confidence": round(confidence, 3),
                    "primary_subject": primary_subject
                }
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"  Error checking relevance: {e}")
            # On error, assume relevant but with low confidence
            return {
                "relevant": True,
                "confidence": 0.5,
                "primary_subject": "Unknown"
            }

    def analyze_article_sentiment(self, symbol: str, headline: str, summary: str) -> Dict:
        """
        Analyze sentiment of a single article using Ollama.
        
        Args:
            symbol: Stock symbol being analyzed
            headline: Article headline
            summary: Article summary/description
            
        Returns:
            Dictionary with score, label, and confidence
        """
        prompt = self.sentiment_prompt_template.format(
            symbol=symbol,
            headline=headline,
            summary=summary
        )
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent results
                    "top_p": 0.9,
                }
            )
            
            # Extract JSON from response
            response_text = response['response'].strip()
            
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                score = float(result.get('score', 0))
                confidence = float(result.get('confidence', 0.5))
                
                # Ensure values are in valid ranges
                score = max(-1, min(1, score))
                confidence = max(0, min(1, confidence))
                
                # Determine label
                if score > 0.15:
                    label = "positive"
                elif score < -0.15:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {
                    "score": round(score, 3),
                    "label": label,
                    "confidence": round(confidence, 3)
                }
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"  Error analyzing article: {e}")
            # Return neutral sentiment on error
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.5
            }

    def process_datetime(self, dt_string: str) -> Dict:
        """
        Process datetime string and return multiple formats.
        
        Args:
            dt_string: Datetime string in ISO format
            
        Returns:
            Dictionary with datetime_iso and datetime_epoch
        """
        try:
            # Parse the datetime string
            if dt_string.endswith('Z'):
                dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(dt_string)
            
            # Convert to epoch milliseconds
            epoch_ms = int(dt.timestamp() * 1000)
            
            return {
                "datetime_iso": dt_string,
                "datetime_epoch": epoch_ms
            }
        except Exception as e:
            print(f"  Error processing datetime {dt_string}: {e}")
            return {
                "datetime_iso": dt_string,
                "datetime_epoch": 0
            }

    def calculate_aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Calculate aggregate sentiment from scored articles.
        
        Args:
            articles: List of scored articles
            
        Returns:
            Dictionary with aggregate score, label, and confidence
        """
        if not articles:
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.5
            }
        
        # Calculate weighted average score (weight by confidence)
        total_weighted_score = sum(a['score'] * a['confidence'] for a in articles)
        total_confidence = sum(a['confidence'] for a in articles)
        
        avg_score = total_weighted_score / total_confidence if total_confidence > 0 else 0
        avg_confidence = total_confidence / len(articles)
        
        # Determine label
        if avg_score > 0.15:
            label = "positive"
        elif avg_score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": round(avg_score, 3),
            "label": label,
            "confidence": round(avg_confidence, 3)
        }

    def determine_action(self, sentiment_score: float, sentiment_label: str) -> str:
        """
        Determine recommended trading action based on sentiment.
        
        Args:
            sentiment_score: Aggregate sentiment score
            sentiment_label: Sentiment label (positive/negative/neutral)
            
        Returns:
            Recommended action string
        """
        if sentiment_label == "positive" and sentiment_score > 0.5:
            return "buy"
        elif sentiment_label == "positive" and sentiment_score > 0.15:
            return "accumulate"
        elif sentiment_label == "negative" and sentiment_score < -0.5:
            return "sell"
        elif sentiment_label == "negative" and sentiment_score < -0.15:
            return "reduce"
        else:
            return "hold"

    def calculate_metrics(self, articles: List[Dict], current_time: datetime) -> Dict:
        """
        Calculate additional news metrics.
        
        Args:
            articles: List of articles with datetime information
            current_time: Current timestamp for calculations
            
        Returns:
            Dictionary with various metrics
        """
        if not articles:
            return {
                "news_cov_60m": 0,
                "news_sent_mean_60m": None,
                "news_uniqueness_24h": None,
                "news_sent_std_24h": 0,
                "news_age_min": 0
            }
        
        # Parse article times
        article_times = []
        scores_60m = []
        scores_24h = []
        
        for article in articles:
            try:
                dt_str = article['datetime']
                if dt_str.endswith('Z'):
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(dt_str)
                
                article_times.append(dt)
                
                # Check if within 60 minutes
                time_diff_minutes = (current_time - dt).total_seconds() / 60
                if time_diff_minutes <= 60:
                    scores_60m.append(article['score'])
                
                # Check if within 24 hours
                if time_diff_minutes <= 1440:  # 24 hours
                    scores_24h.append(article['score'])
                    
            except Exception as e:
                print(f"  Error parsing datetime: {e}")
                continue
        
        # Calculate metrics
        news_cov_60m = len(scores_60m)
        news_sent_mean_60m = statistics.mean(scores_60m) if scores_60m else None
        
        # Calculate uniqueness (ratio of unique articles to total in 24h)
        news_uniqueness_24h = 1.0 if len(articles) > 0 else None
        
        # Calculate standard deviation of sentiment in 24h
        news_sent_std_24h = round(statistics.stdev(scores_24h), 3) if len(scores_24h) > 1 else 0
        
        # Calculate age of newest article in minutes
        if article_times:
            newest_article = max(article_times)
            news_age_min = int((current_time - newest_article).total_seconds() / 60)
        else:
            news_age_min = 0
        
        return {
            "news_cov_60m": news_cov_60m,
            "news_sent_mean_60m": round(news_sent_mean_60m, 3) if news_sent_mean_60m is not None else None,
            "news_uniqueness_24h": news_uniqueness_24h,
            "news_sent_std_24h": news_sent_std_24h,
            "news_age_min": news_age_min
        }

    def process_symbol(self, symbol_data: Dict, current_time: datetime, 
                      min_relevance_confidence: float = 0.6) -> Dict:
        """
        Process all news for a single symbol with relevance filtering.
        
        Args:
            symbol_data: Dictionary containing symbol and news articles
            current_time: Current timestamp
            min_relevance_confidence: Minimum confidence threshold for relevance (default: 0.6)
            
        Returns:
            Dictionary with analysis results
        """
        symbol = symbol_data['symbol']
        news_articles = symbol_data.get('news', [])
        
        print(f"\n{'='*60}")
        print(f"Processing {symbol}...")
        print(f"{'='*60}")
        print(f"Total articles found: {len(news_articles)}")
        
        # First pass: Check relevance
        relevant_articles = []
        filtered_articles = []
        
        for idx, article in enumerate(news_articles, 1):
            headline = article.get('title', '')
            # Use description as summary (description is the primary field from news APIs)
            summary = article.get('description') or article.get('summary') or ''

            print(f"\n[{idx}/{len(news_articles)}] Checking relevance...")
            print(f"  Headline: {headline[:80]}...")
            
            # Check relevance
            relevance = self.check_relevance(symbol, headline, summary)
            
            article_with_relevance = {
                **article,
                'relevance': relevance
            }
            
            if relevance['relevant'] and relevance['confidence'] >= min_relevance_confidence:
                relevant_articles.append(article_with_relevance)
                print(f"  ✓ RELEVANT (confidence: {relevance['confidence']}, subject: {relevance['primary_subject']})")
            else:
                filtered_articles.append(article_with_relevance)
                print(f"  ✗ FILTERED OUT (confidence: {relevance['confidence']}, subject: {relevance['primary_subject']})")
        
        print(f"\n{'-'*60}")
        print(f"Relevance filtering complete:")
        print(f"  Relevant articles: {len(relevant_articles)}")
        print(f"  Filtered out: {len(filtered_articles)}")
        print(f"{'-'*60}")
        
        # Second pass: Analyze sentiment for relevant articles only
        scored_articles = []
        for idx, article in enumerate(relevant_articles, 1):
            headline = article.get('title', '')
            # Use description as summary (description is the primary field from news APIs)
            summary = article.get('description') or article.get('summary') or ''

            print(f"\nAnalyzing sentiment [{idx}/{len(relevant_articles)}]...")

            # Get sentiment
            sentiment = self.analyze_article_sentiment(symbol, headline, summary)
            
            # Process datetime
            dt_info = self.process_datetime(article['datetime'])
            
            # Create scored article
            scored_article = {
                "headline": headline,
                "summary": summary,
                "url": article['url'],
                "source": article['source'],
                "datetime": article['datetime'],
                "datetime_iso": dt_info['datetime_iso'],
                "datetime_epoch": dt_info['datetime_epoch'],
                "score": sentiment['score'],
                "label": sentiment['label'],
                "confidence": sentiment['confidence'],
                "relevance_check": article['relevance']
            }
            scored_articles.append(scored_article)
            print(f"  Sentiment: {sentiment['label']} (score: {sentiment['score']}, confidence: {sentiment['confidence']})")
        
        print(f"\n{'='*60}")
        print(f"Completed sentiment analysis for {len(scored_articles)} relevant articles")
        print(f"{'='*60}")
        
        # Calculate aggregate sentiment (only from relevant articles)
        aggregate_sentiment = self.calculate_aggregate_sentiment(scored_articles)
        
        # Determine recommended action
        action = self.determine_action(
            aggregate_sentiment['score'],
            aggregate_sentiment['label']
        )
        
        # Find top news (most recent relevant article)
        top_news = scored_articles[0] if scored_articles else None
        if top_news:
            top_news_output = {
                "headline": top_news["headline"],
                "url": top_news["url"],
                "source": top_news["source"],
                "summary": top_news["summary"],  # Summary is guaranteed to be set from description
                "datetime": top_news["datetime"],
                "datetime_iso": top_news["datetime_iso"],
                "datetime_epoch": top_news["datetime_epoch"]
            }
        else:
            top_news_output = None
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(scored_articles, current_time)
        
        # Build filtered articles summary (without full content)
        filtered_summary = [
            {
                "headline": art.get('title', ''),
                "source": art['source'],
                "datetime": art['datetime'],
                "filtered_reason": f"Primary subject: {art['relevance']['primary_subject']}",
                "relevance_confidence": art['relevance']['confidence']
            }
            for art in filtered_articles
        ]
        
        # Build output
        result = {
            "symbol": symbol,
            "sentiment": aggregate_sentiment,
            "recommended_action": action,
            "top_news": top_news_output,
            "articles_analyzed": len(scored_articles),
            "articles_filtered": len(filtered_articles),
            "articles_scored": scored_articles,
            "filtered_articles": filtered_summary,
            **metrics
        }
        
        return result

    def analyze_payload(self, input_data: List[Dict], 
                       min_relevance_confidence: float = 0.6) -> List[Dict]:
        """
        Analyze the complete input payload.
        
        Args:
            input_data: List containing payload dictionaries
            min_relevance_confidence: Minimum confidence for relevance filtering
            
        Returns:
            List of analysis results
        """
        results = []
        current_time = datetime.now(timezone.utc)
        
        for item in input_data:
            payload = item.get('payload', [])
            
            for symbol_data in payload:
                result = self.process_symbol(
                    symbol_data, 
                    current_time,
                    min_relevance_confidence
                )
                results.append(result)
        
        return results


def main():
    """Main function to run the sentiment analysis."""
    import sys
    
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer_enhanced.py <input_file.json> [output_file.json] [min_relevance]")
        print("\nArguments:")
        print("  input_file.json   : Input JSON file with news data")
        print("  output_file.json  : Output JSON file (default: sentiment_output_enhanced.json)")
        print("  min_relevance     : Minimum relevance confidence 0-1 (default: 0.6)")
        print("\nExample:")
        print("  python sentiment_analyzer_enhanced.py Input.json Output.json 0.7")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sentiment_output_enhanced.json"
    min_relevance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    
    print("="*60)
    print("ENHANCED SENTIMENT ANALYZER WITH RELEVANCE FILTERING")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Minimum relevance confidence: {min_relevance}")
    print("="*60)
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Initialize analyzer
    print("\nInitializing sentiment analyzer with Ollama (llama3.2)...")
    analyzer = NewsAnalyzer(model="llama3.2")
    
    # Process the data
    print("\nStarting analysis with relevance filtering...\n")
    results = analyzer.analyze_payload(input_data, min_relevance)
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nProcessed {len(results)} symbols:")
    for result in results:
        print(f"\n{result['symbol']}:")
        print(f"  Articles analyzed: {result['articles_analyzed']}")
        print(f"  Articles filtered: {result['articles_filtered']}")
        print(f"  Sentiment: {result['sentiment']['label']} (score: {result['sentiment']['score']})")
        print(f"  Recommended action: {result['recommended_action']}")
        if result['articles_filtered'] > 0:
            print(f"  Filtered articles:")
            for filtered in result['filtered_articles']:
                print(f"    - {filtered['headline'][:60]}...")
                print(f"      Reason: {filtered['filtered_reason']}")


if __name__ == "__main__":
    main()
