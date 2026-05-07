
import logging
import json
from datetime import datetime, timedelta, timezone
from src.pipeline.pass_a_ingest import fetch_gdelt_articles, build_gdelt_queries

logging.basicConfig(level=logging.INFO)

def test_gdelt():
    queries = build_gdelt_queries()
    print(f"Testing {len(queries)} GDELT queries...")
    
    for i, q in enumerate(queries[:3]): # Test first 3 queries
        print(f"\n--- Testing Query {i+1} ---")
        print(f"Query: {q['query']}")
        print(f"Countries: {q.get('countries')}")
        print(f"Tone: {q.get('tone')}")
        
        try:
            items = fetch_gdelt_articles(
                query=q["query"],
                max_age_days=3,
                tone=q.get("tone"),
                source_countries=q.get("countries")
            )
            print(f"Results: {len(items)}")
            if items:
                for item in items[:2]:
                    print(f"  - {item['title']} ({item['link'][:50]}...)")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_gdelt()
