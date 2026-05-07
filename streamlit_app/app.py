"""
SIM — Streamlit Dashboard Entry Point
Blueprint V20.1 §5

Security Incident Monitor — Aviation OSINT Intelligence Dashboard.
Modern dark theme with glassmorphism, rich sidebar, and tabbed navigation.
"""

import importlib
import json
import sys
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Path Setup ──
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
importlib.invalidate_caches()  # Clear stale module caches on reload

# ── Lazy Imports ──
# Import after path setup to avoid caching issues.
from src.services.supabase_client import get_connection, put_connection
from components.alert_feed import render_alert_feed
from components.anchor_lookup import render_anchor_lookup
from components.czib_dashboard import render_czib_dashboard
from components.event_table import render_event_table
from components.map_view import render_map
from components.storyline_graph import render_storyline_graph
from components.telemetry_dashboard import render_telemetry
from services.cache import (
    get_alert_events,
    get_czib_stats,
    get_czib_zones,
    get_geo_summary,
    get_pipeline_stats,
    get_recent_events,
    get_storyline_graph_data,
)

_UI_CONFIG = json.loads(
    (Path(__file__).parent / "config" / "ui_settings.json").read_text()
)
_THEME = _UI_CONFIG["theme"]
_TIERS = _UI_CONFIG["tiers"]

# ── Page Config ──
st.set_page_config(
    page_title=_UI_CONFIG["app"]["title"],
    page_icon=_UI_CONFIG["app"]["page_icon"],
    layout=_UI_CONFIG["app"]["layout"],
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
    }}

    .stApp {{
      background: linear-gradient(180deg, {_THEME["background"]} 0%, #0D1525 100%);
    }}

    .main .block-container {{
      padding-top: 1rem;
      padding-bottom: 2rem;
      max-width: 100%;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {_THEME["surface"]} 0%, #111827 100%) !important;
      border-right: 1px solid {_THEME["border"]};
    }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
      color: {_THEME["text_primary"]} !important;
      font-size: 0.9em;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-top: 1.2em;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
      background: {_THEME["surface"]};
      border: 1px solid {_THEME["border"]};
      border-radius: 10px;
      padding: 12px;
    }}
    [data-testid="stMetricValue"] {{
      font-size: 1.6rem;
      font-weight: 800;
      color: {_THEME["text_primary"]};
    }}
    [data-testid="stMetricLabel"] {{
      color: {_THEME["text_secondary"]};
      font-weight: 500;
      font-size: 0.75em;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
      gap: 6px;
      padding: 4px;
      background: {_THEME["surface"]};
      border-radius: 12px;
      border: 1px solid {_THEME["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 8px;
      padding: 8px 18px;
      color: {_THEME["text_secondary"]};
      font-weight: 500;
      font-size: 0.85em;
      transition: all 0.2s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
      background: {_THEME["surface_hover"]};
      color: {_THEME["text_primary"]};
    }}
    .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg, {_THEME["primary"]}20, {_THEME["secondary"]}20);
      color: {_THEME["primary"]};
      font-weight: 700;
      border: 1px solid {_THEME["primary"]}40;
    }}

    /* Buttons */
    .stButton > button {{
      border-radius: 8px;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .stButton > button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px {_THEME["primary"]}30;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {{
      background: {_THEME["surface"]} !important;
      border: 1px solid {_THEME["border"]} !important;
      border-radius: 8px !important;
      color: {_THEME["text_primary"]} !important;
    }}
    .stSlider > div > div {{
      background: {_THEME["surface"]} !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
      background: {_THEME["surface"]};
      border: 1px solid {_THEME["border"]};
      border-radius: 8px;
      font-size: 0.85em;
      font-weight: 600;
    }}
    .streamlit-expanderContent {{
      background: {_THEME["background"]};
      border: 1px solid {_THEME["border"]};
      border-top: none;
      border-radius: 0 0 8px 8px;
    }}

    /* Dataframes */
    .stDataFrame {{
      border: 1px solid {_THEME["border"]};
      border-radius: 10px;
      overflow: hidden;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    ::-webkit-scrollbar-track {{
      background: {_THEME["background"]};
    }}
    ::-webkit-scrollbar-thumb {{
      background: {_THEME["border"]};
      border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: {_THEME["text_muted"]};
    }}

    /* Header */
    .sim-header {{
      background: linear-gradient(135deg, {_THEME["primary"]}, {_THEME["secondary"]}, {_THEME["accent"]});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 2.2rem;
      font-weight: 800;
      margin-bottom: 0;
      letter-spacing: -0.02em;
    }}
    .sim-subtitle {{
      color: {_THEME["text_secondary"]};
      font-size: 0.95em;
      margin-top: -6px;
      font-weight: 400;
    }}
    .sim-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.65em;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {_THEME["accent"]};
      background: {_THEME["accent"]}15;
      border: 1px solid {_THEME["accent"]}30;
      margin-left: 10px;
      vertical-align: middle;
    }}

    /* Alert pulse animation for critical */
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 {_TIERS["CRITICAL"]["color"]}40; }}
      70% {{ box-shadow: 0 0 0 8px {_TIERS["CRITICAL"]["color"]}00; }}
      100% {{ box-shadow: 0 0 0 0 {_TIERS["CRITICAL"]["color"]}00; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──
col_h1, col_h2 = st.columns([6, 1])
with col_h1:
    st.markdown(
        f'<p class="sim-header">{_UI_CONFIG["app"]["page_icon"]} {_UI_CONFIG["app"]["title"]}'
        f'<span class="sim-badge">v20.1</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sim-subtitle">{_UI_CONFIG["app"]["subtitle"]}</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ── Database Connection ──
db_conn = None
try:
    db_conn = get_connection()
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.info("Configure DATABASE_URL or SUPABASE_* environment variables")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    # Brand
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:16px;">
          <div style="font-size:2em;margin-bottom:4px;">{_UI_CONFIG["app"]["page_icon"]}</div>
          <div style="font-weight:800;color:#F8FAFC;font-size:1.1em;">{_UI_CONFIG["app"]["title"]}</div>
          <div style="font-size:0.75em;color:#64748B;">{_UI_CONFIG["app"]["subtitle"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Controls
    st.markdown("### ⚙️ Controls")
    auto_refresh = st.toggle("Auto Refresh (60s)", value=True, key="auto_ref")
    if auto_refresh:
        st_autorefresh(
            interval=_UI_CONFIG["app"]["auto_refresh_seconds"] * 1000,
            limit=None,
            key="auto_refresh",
        )

    # Quick stats
    st.divider()
    st.markdown("### 📊 Quick Stats")
    try:
        stats = get_pipeline_stats(db_conn)
        events_24h = stats.get("events_24h", 0)
        alert_counts = stats.get("alert_counts", {})
        last_run = stats.get("last_run_at", "—")

        s1, s2 = st.columns(2)
        s1.metric("Events 24h", events_24h)
        s2.metric("Critical", alert_counts.get("CRITICAL", 0))

        s3, s4 = st.columns(2)
        s3.metric("Alert", alert_counts.get("ALERT", 0))
        s4.metric("Watch", alert_counts.get("WATCH", 0))

        st.caption(f"Last run: {last_run[:19] if last_run else '—'}")
    except Exception:
        st.warning("Could not load stats")

    # CZIB mini stats
    st.divider()
    st.markdown("### 🛡️ EASA CZIB")
    try:
        czib_stats = get_czib_stats(db_conn)
        cz1, cz2, cz3 = st.columns(3)
        cz1.metric("🟢 Active", czib_stats.get("active", 0))
        cz2.metric("🟡 Susp.", czib_stats.get("suspended", 0))
        cz3.metric("🌍 Countries", czib_stats.get("countries", 0))
    except Exception:
        st.caption("CZIB data not available")

    # Geo summary mini table
    st.divider()
    st.markdown("### 🌍 Top Countries")
    try:
        geo = get_geo_summary(db_conn)
        for g in geo[:5]:
            total = g["total"]
            crit = g.get("critical", 0)
            flag = chr(0x1F1E6 + ord(g["country_iso"][0].upper()) - 65) + chr(0x1F1E6 + ord(g["country_iso"][1].upper()) - 65) if g.get("country_iso") else "🌐"
            bar_w = min(100, total * 3)
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:0.8em;">
                  <span style="min-width:28px;">{flag}</span>
                  <span style="color:#94A3B8;min-width:24px;">{g['country_iso']}</span>
                  <div style="flex:1;height:6px;background:#1E293B;border-radius:3px;overflow:hidden;">
                    <div style="width:{bar_w}px;height:100%;background:{'#EF4444' if crit > 0 else '#6366F1'};border-radius:3px;"></div>
                  </div>
                  <span style="color:#F8FAFC;font-weight:600;min-width:28px;text-align:right;">{total}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="color: #475569; font-size: 0.7em; text-align: center; line-height: 1.6;">
            SIM V20.1 — Multi-Provider Production Fortress<br/>
            <span style="color:#6366F1;">Groq</span> + <span style="color:#8B5CF6;">OpenRouter</span><br/>
            OSINT Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Tabs ──
tab_events, tab_alerts, tab_map, tab_czib, tab_storylines, tab_telemetry, tab_anchors = st.tabs([
    "📋 Events",
    "🚨 Alerts",
    "🗺️ Map",
    "🛡️ CZIB",
    "🔗 Storylines",
    "📈 Telemetry",
    "✈️ Anchors",
])

# Tab 1: Events
with tab_events:
    try:
        events = get_recent_events(db_conn)
        render_event_table(events)
    except Exception as e:
        st.error(f"Events error: {e}")

# Tab 2: Alerts
with tab_alerts:
    try:
        alerts = get_alert_events(db_conn)
        render_alert_feed(alerts)
    except Exception as e:
        st.error(f"Alert feed error: {e}")

# Tab 3: Map
with tab_map:
    try:
        events = get_recent_events(db_conn)
        czib_zones = get_czib_zones(db_conn, only_active=False)
        render_map(events, czib_data=czib_zones)
    except Exception as e:
        st.error(f"Map error: {e}")

# Tab 4: CZIB Dashboard
with tab_czib:
    try:
        render_czib_dashboard(db_conn)
    except Exception as e:
        st.error(f"CZIB error: {e}")

# Tab 5: Storylines
with tab_storylines:
    try:
        graph_data = get_storyline_graph_data(db_conn)
        render_storyline_graph(graph_data)
    except Exception as e:
        st.error(f"Storyline error: {e}")

# Tab 6: Telemetry
with tab_telemetry:
    try:
        stats = get_pipeline_stats(db_conn)
        render_telemetry(stats)
    except Exception as e:
        st.error(f"Telemetry error: {e}")

# Tab 7: Anchors
with tab_anchors:
    try:
        render_anchor_lookup(db_conn)
    except Exception as e:
        st.error(f"Anchor lookup error: {e}")

# ── Cleanup ──
if db_conn is not None:
    try:
        db_conn.commit()
    except Exception:
        pass
    try:
        put_connection(db_conn)
    except Exception:
        pass
