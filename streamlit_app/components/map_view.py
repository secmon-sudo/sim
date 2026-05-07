"""
SIM — Map View Component
Blueprint V20.1 §5.1

PyDeck ScatterplotLayer with severity-based sizing, tier coloring,
rich HTML tooltips, and a floating legend.
"""

import json
from pathlib import Path

import pydeck as pdk
import streamlit as st

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())
_TIER_COLORS = _UI_CONFIG["tiers"]

TIER_COLOR_MAP = {
    "CRITICAL": [239, 68, 68, 220],
    "ALERT":    [249, 115, 22, 200],
    "WATCH":    [234, 179, 8, 200],
    None:       [100, 116, 139, 120],
}

TIER_GLOW_MAP = {
    "CRITICAL": [239, 68, 68, 60],
    "ALERT":    [249, 115, 22, 50],
    "WATCH":    [234, 179, 8, 40],
    None:       [100, 116, 139, 30],
}


def _country_flag(iso: str | None) -> str:
    if not iso or len(iso) != 2:
        return ""
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def render_map(events: list[dict]):
    """Render PyDeck scatter map of events with severity-based sizing and glow."""
    map_data = []
    for e in events:
        if e.get("latitude") and e.get("longitude"):
            tier = e.get("alert_tier")
            color = TIER_COLOR_MAP.get(tier, TIER_COLOR_MAP[None])
            glow = TIER_GLOW_MAP.get(tier, TIER_GLOW_MAP[None])
            severity = e.get("severity_score", 20)
            # Size based on severity, with minimum visibility
            radius = max(severity * 250, 3000)
            glow_radius = radius * 2.5

            title = (e.get("source_title") or "Untitled")[:90]
            flag = _country_flag(e.get("country_iso"))
            map_data.append({
                "lat": float(e["latitude"]),
                "lon": float(e["longitude"]),
                "color": color,
                "glow_color": glow,
                "radius": radius,
                "glow_radius": glow_radius,
                "title": title,
                "type": (e.get("event_type") or "unknown").replace("_", " ").title(),
                "tier": tier or "—",
                "severity": severity,
                "anchor": e.get("anchor_name_norm") or "—",
                "country": f"{flag} {e.get('country_iso') or '—'}",
                "confidence": float(e.get("system_confidence") or 0),
            })

    if not map_data:
        st.info("📍 No geolocated events to display")
        return

    # Main scatter layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        opacity=0.85,
        stroked=True,
        get_line_color=[255, 255, 255, 80],
        get_line_width=2,
    )

    # Glow layer (larger, more transparent)
    glow_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_fill_color="glow_color",
        get_radius="glow_radius",
        pickable=False,
        auto_highlight=False,
        opacity=0.4,
    )

    view = pdk.ViewState(
        latitude=_UI_CONFIG["map"]["initial_latitude"],
        longitude=_UI_CONFIG["map"]["initial_longitude"],
        zoom=_UI_CONFIG["map"]["initial_zoom"],
        pitch=0,
    )

    tooltip = {
        "html": """
        <div style="font-family: Inter, sans-serif; padding: 10px 12px; max-width: 320px;">
          <div style="font-weight:700; color:#F8FAFC; font-size:0.95em; margin-bottom:6px; line-height:1.3;">
            {title}
          </div>
          <div style="color:#94A3B8; font-size:0.82em; line-height:1.6;">
            <span style="color:#64748B;">Type:</span> {type}<br/>
            <span style="color:#64748B;">Tier:</span> {tier}<br/>
            <span style="color:#64748B;">Severity:</span> {severity}<br/>
            <span style="color:#64748B;">Confidence:</span> {confidence:.0%}<br/>
            <span style="color:#64748B;">Anchor:</span> {anchor}<br/>
            <span style="color:#64748B;">Country:</span> {country}
          </div>
        </div>
        """,
        "style": {
            "backgroundColor": "#151E32",
            "color": "#F8FAFC",
            "border": "1px solid #334155",
            "borderRadius": "10px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.5)",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[glow_layer, scatter_layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v11",
        ),
        use_container_width=True,
        height=550,
    )

    # Floating legend
    st.markdown(
        """
        <div style="display:flex;gap:16px;justify-content:center;margin-top:8px;">
          <span style="display:flex;align-items:center;gap:6px;font-size:0.8em;color:#94A3B8;">
            <span style="width:10px;height:10px;border-radius:50%;background:#EF4444;display:inline-block;"></span> Critical
          </span>
          <span style="display:flex;align-items:center;gap:6px;font-size:0.8em;color:#94A3B8;">
            <span style="width:10px;height:10px;border-radius:50%;background:#F97316;display:inline-block;"></span> Alert
          </span>
          <span style="display:flex;align-items:center;gap:6px;font-size:0.8em;color:#94A3B8;">
            <span style="width:10px;height:10px;border-radius:50%;background:#EAB308;display:inline-block;"></span> Watch
          </span>
          <span style="display:flex;align-items:center;gap:6px;font-size:0.8em;color:#94A3B8;">
            <span style="width:10px;height:10px;border-radius:50%;background:#64748B;display:inline-block;"></span> No Alert
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
