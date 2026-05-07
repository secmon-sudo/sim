"""
SIM — Streamlit Dashboard Entry Point
Blueprint V20.1 §5

Security Incident Monitor — Aviation OSINT Intelligence Dashboard.
"""

import json
import sys
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.supabase_client import get_connection, put_connection
from streamlit_app.components.alert_feed import render_alert_feed
from streamlit_app.components.anchor_lookup import render_anchor_lookup
from streamlit_app.components.event_table import render_event_table
from streamlit_app.components.map_view import render_map
from streamlit_app.components.storyline_graph import render_storyline_graph
from streamlit_app.components.telemetry_dashboard import render_telemetry
from streamlit_app.services.cache import (
    get_alert_events,
    get_pipeline_stats,
    get_recent_events,
    get_storyline_graph_data,
)

# Load UI config
_UI_CONFIG = json.loads(
    (Path(__file__).parent / "config" / "ui_settings.json").read_text()
)

# --- Page Config ---
st.set_page_config(
    page_title=_UI_CONFIG["app"]["title"],
    page_icon=_UI_CONFIG["app"]["page_icon"],
    layout=_UI_CONFIG["app"]["layout"],
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {_UI_CONFIG["theme"]["background"]};
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        max-width: 100%;
    }}

    /* Metric cards */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {_UI_CONFIG["theme"]["text_primary"]};
    }}

    [data-testid="stMetricLabel"] {{
        color: {_UI_CONFIG["theme"]["text_secondary"]};
        font-weight: 500;
    }}

    /* Divider */
    hr {{
        border-color: {_UI_CONFIG["theme"]["border"]};
        opacity: 0.3;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {_UI_CONFIG["theme"]["surface"]};
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: {_UI_CONFIG["theme"]["text_secondary"]};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {_UI_CONFIG["theme"]["surface_light"]};
        color: {_UI_CONFIG["theme"]["primary"]};
        font-weight: 600;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {_UI_CONFIG["theme"]["surface"]};
        border-radius: 8px;
    }}

    /* Header gradient */
    .sim-header {{
        background: linear-gradient(135deg, {_UI_CONFIG["theme"]["primary"]}, {_UI_CONFIG["theme"]["secondary"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0;
    }}

    .sim-subtitle {{
        color: {_UI_CONFIG["theme"]["text_secondary"]};
        font-size: 1rem;
        margin-top: -8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    f'<p class="sim-header">{_UI_CONFIG["app"]["page_icon"]} {_UI_CONFIG["app"]["title"]}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="sim-subtitle">{_UI_CONFIG["app"]["subtitle"]}</p>',
    unsafe_allow_html=True,
)

st.divider()

# --- Database Connection ---
db_conn = None
try:
    db_conn = get_connection()
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.info("Configure DATABASE_URL or SUPABASE_* environment variables")
    st.stop()

# --- Auto Refresh (state-preserving) ---
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    auto_refresh = st.toggle("Auto Refresh (60s)", value=True)
    if auto_refresh:
        st_autorefresh(interval=_UI_CONFIG["app"]["auto_refresh_seconds"] * 1000, limit=None, key="auto_refresh")

    st.divider()
    st.markdown("### 📊 Quick Stats")
    try:
        stats = get_pipeline_stats(db_conn)
        st.metric("Events (24h)", stats.get("llm_calls_24h", 0))
        st.metric("Last Run", stats.get("last_run_at", "—")[:19] if stats.get("last_run_at") else "—")
    except Exception:
        st.warning("Could not load stats")

    st.divider()
    st.markdown(
        """
        <div style="color: #64748B; font-size: 0.75em; text-align: center;">
            SIM V20.1 — Multi-Provider Production Fortress<br/>
            Powered by Groq + OpenRouter
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Main Content Tabs ---
tab_map, tab_events, tab_alerts, tab_storylines, tab_telemetry, tab_anchors = st.tabs([
    "🗺️ Incident Map",
    "📋 Event Table",
    "🚨 Alert Feed",
    "🔗 Storylines",
    "📈 Telemetry",
    "✈️ Anchors",
])

# Tab 1: Map
with tab_map:
    try:
        events = get_recent_events(db_conn)
        render_map(events)
    except Exception as e:
        st.error(f"Map error: {e}")

# Tab 2: Event Table
with tab_events:
    try:
        events = get_recent_events(db_conn)
        render_event_table(events)
    except Exception as e:
        st.error(f"Table error: {e}")

# Tab 3: Alert Feed
with tab_alerts:
    try:
        alerts = get_alert_events(db_conn)
        render_alert_feed(alerts)
    except Exception as e:
        st.error(f"Alert feed error: {e}")

# Tab 4: Storylines
with tab_storylines:
    try:
        graph_data = get_storyline_graph_data(db_conn)
        render_storyline_graph(graph_data)
    except Exception as e:
        st.error(f"Storyline error: {e}")

# Tab 5: Telemetry
with tab_telemetry:
    try:
        stats = get_pipeline_stats(db_conn)
        render_telemetry(stats)
    except Exception as e:
        st.error(f"Telemetry error: {e}")

# Tab 6: Anchors
with tab_anchors:
    try:
        render_anchor_lookup(db_conn)
    except Exception as e:
        st.error(f"Anchor lookup error: {e}")

# --- Cleanup ---
if db_conn is not None:
    try:
        # Commit to close any implicit transaction from SELECT queries
        # so the pool doesn't log a rollback warning
        db_conn.commit()
    except Exception:
        pass
    try:
        put_connection(db_conn)
    except Exception:
        pass
