#!/usr/bin/env python3
"""
Test script for Sentiment Analysis API
Tests all endpoints with various payloads
"""

import requests
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv()

API_KEY = os.getenv("API_KEY", "test_key")
BASE_URL = os.getenv("API_URL", "http://localhost:8090")

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def print_separator(title=""):
    """Print a nice separator."""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)


def test_health():
    """Test health check endpoint."""
    print_separator("Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200 and response.json().get("ok"):
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_root():
    """Test root endpoint."""
    print_separator("Testing Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            return True
        else:
            print("❌ Root endpoint failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_analyze_with_file():
    """Test analyze endpoint with Input.json file."""
    print_separator("Testing Analyze Endpoint with Input.json")
    
    input_file = Path("Input.json")
    if not input_file.exists():
        print(f"❌ Input.json not found in current directory")
        return False
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Loaded {len(data[0]['payload'])} symbols from Input.json")
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            headers=headers,
            json=data,
            timeout=120  # Allow time for LLM processing
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Symbols analyzed: {len(result['results'])}")
            
            for r in result['results']:
                print(f"\n📊 {r['symbol']}:")
                print(f"   Sentiment: {r['sentiment']['label'].upper()} ({r['sentiment']['score']})")
                print(f"   Confidence: {r['sentiment']['confidence']}")
                print(f"   Action: {r['recommended_action'].upper()}")
                print(f"   Articles analyzed: {r['articles_analyzed']}")
                print(f"   Articles filtered: {r['articles_filtered']}")
                
                if r.get('filtered_articles'):
                    print(f"   Filtered:")
                    for fa in r['filtered_articles'][:3]:  # Show first 3
                        print(f"     - {fa['headline'][:60]}...")
            
            print("\n✅ Analyze with Input.json passed")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_simple():
    """Test analyze endpoint with simple payload."""
    print_separator("Testing Analyze Endpoint with Simple Payload")
    
    data = {
        "payload": [
            {
                "symbol": "AAPL",
                "news": [
                    {
                        "title": "Apple Reports Record Q4 Earnings",
                        "url": "https://example.com/apple-earnings",
                        "datetime": "2025-11-02T10:00:00Z",
                        "summary": "Apple Inc. reported record quarterly revenue of $125 billion, exceeding analyst expectations."
                    },
                    {
                        "title": "Apple Stock Surges on Strong iPhone Sales",
                        "url": "https://example.com/apple-iphone",
                        "datetime": "2025-11-02T11:00:00Z",
                        "summary": "Apple shares jumped 5% following reports of strong iPhone 16 sales in international markets."
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            symbol_result = result['results'][0]
            
            print(f"\n📊 {symbol_result['symbol']}:")
            print(f"   Sentiment: {symbol_result['sentiment']['label'].upper()}")
            print(f"   Score: {symbol_result['sentiment']['score']}")
            print(f"   Action: {symbol_result['recommended_action'].upper()}")
            
            print("\n✅ Simple analyze test passed")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_analyze_single():
    """Test analyze_single endpoint."""
    print_separator("Testing Analyze Single Symbol Endpoint")
    
    data = {
        "symbol": "TSLA",
        "news": [
            {
                "title": "Tesla Misses Q3 Delivery Targets",
                "url": "https://example.com/tesla",
                "datetime": "2025-11-02T09:00:00Z",
                "summary": "Tesla reported lower than expected vehicle deliveries for Q3, citing supply chain challenges."
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze_single",
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            symbol_result = result['results'][0]
            
            print(f"\n📊 {symbol_result['symbol']}:")
            print(f"   Sentiment: {symbol_result['sentiment']['label'].upper()}")
            print(f"   Score: {symbol_result['sentiment']['score']}")
            
            print("\n✅ Single symbol test passed")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_config():
    """Test config endpoints."""
    print_separator("Testing Config Endpoints")
    
    try:
        # Get config
        response = requests.get(
            f"{BASE_URL}/config",
            headers=headers,
            timeout=5
        )
        
        print(f"GET /config - Status Code: {response.status_code}")
        
        if response.status_code == 200:
            config = response.json()
            print(f"Current config: {json.dumps(config, indent=2)}")
            print("✅ Get config passed")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_auth_failure():
    """Test that authentication is required."""
    print_separator("Testing Authentication (Should Fail)")
    
    try:
        # Try without API key
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"payload": []},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("✅ Authentication properly rejected unauthorized request")
            return True
        else:
            print(f"⚠️  Expected 401, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("  SENTIMENT ANALYSIS API - TEST SUITE")
    print("="*70)
    print(f"\nAPI URL: {BASE_URL}")
    print(f"API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Authentication Check", test_auth_failure),
        ("Config Endpoint", test_config),
        ("Simple Analyze", test_analyze_simple),
        ("Single Symbol", test_analyze_single),
        ("Full Input.json", test_analyze_with_file),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print_separator("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_status in results:
        status = "✅ PASS" if passed_status else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return passed == total


if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=2)
        if response.status_code != 200:
            print(f"⚠️  Warning: API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to API at {BASE_URL}")
        print("\nMake sure the API is running:")
        print("  python sentiment_api.py")
        print("\nOr set API_URL environment variable:")
        print("  export API_URL=https://sentiment-api.aarch.shop")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        sys.exit(1)
    
    # Run tests
    all_passed = run_all_tests()
    
    sys.exit(0 if all_passed else 1)
