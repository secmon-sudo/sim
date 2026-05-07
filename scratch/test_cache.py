import os
import psycopg
from src.services.supabase_client import get_connection, put_connection
from streamlit_app.services.cache import get_recent_events

def test():
    conn = get_connection()
    try:
        events = get_recent_events(conn, limit=5)
        print(f"Fetched {len(events)} events")
        if events:
            print(f"Keys: {list(events[0].keys())}")
            print(f"First event alert_tier: {events[0].get('alert_tier')}")
            print(f"First event status: {events[0].get('status')}")
    finally:
        put_connection(conn)

if __name__ == "__main__":
    test()
