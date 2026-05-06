"""
SIM — Map View Component
Blueprint V20.1 §5.1

PyDeck ScatterplotLayer with severity-based coloring and rich tooltips.
"""

import json
from pathlib import Path

import pydeck as pdk
import streamlit as st

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())
_TIER_COLORS = _UI_CONFIG["tiers"]

# Color mapping: alert_tier → RGB
TIER_COLOR_MAP = {
    "CRITICAL": [220, 38, 38, 200],
    "ALERT":    [234, 88, 12, 200],
    "WATCH":    [202, 138, 4, 200],
    None:       [100, 116, 139, 150],
}


def render_map(events: list[dict]):
    """Render PyDeck scatter map of events with severity-based sizing."""
    # Filter events with valid coordinates
    map_data = []
    for e in events:
        if e.get("latitude") and e.get("longitude"):
            tier = e.get("alert_tier")
            color = TIER_COLOR_MAP.get(tier, TIER_COLOR_MAP[None])
            severity = e.get("severity_score", 20)
            map_data.append({
                "lat": float(e["latitude"]),
                "lon": float(e["longitude"]),
                "color": color,
                "radius": max(severity * 300, 5000),
                "title": (e.get("source_title") or "Untitled")[:80],
                "type": e.get("event_type", "unknown"),
                "tier": tier or "—",
                "severity": severity,
                "anchor": e.get("anchor_name_norm") or "—",
                "country": e.get("country_iso") or "—",
            })

    if not map_data:
        st.info("📍 No geolocated events to display")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        opacity=0.7,
    )

    view = pdk.ViewState(
        latitude=_UI_CONFIG["map"]["initial_latitude"],
        longitude=_UI_CONFIG["map"]["initial_longitude"],
        zoom=_UI_CONFIG["map"]["initial_zoom"],
        pitch=0,
    )

    tooltip = {
        "html": """
        <div style="font-family: Inter, sans-serif; padding: 8px;">
            <b>{title}</b><br/>
            <span style="color: #94A3B8;">Type:</span> {type}<br/>
            <span style="color: #94A3B8;">Tier:</span> {tier} &nbsp;|&nbsp;
            <span style="color: #94A3B8;">Severity:</span> {severity}<br/>
            <span style="color: #94A3B8;">Anchor:</span> {anchor} ({country})
        </div>
        """,
        "style": {
            "backgroundColor": "#1E293B",
            "color": "#F1F5F9",
            "border": "1px solid #475569",
            "border-radius": "8px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v11",
        ),
        use_container_width=True,
    )

    # Legend
    cols = st.columns(4)
    for i, (tier, info) in enumerate(_TIER_COLORS.items()):
        cols[i].markdown(
            f"{info['icon']} **{info['label']}**",
        )
    cols[3].markdown("⚪ **No Alert**")
