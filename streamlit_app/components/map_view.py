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


def render_map(events: list[dict], czib_data: list[dict] | None = None):
    """Render PyDeck scatter map of events with severity-based sizing, glow, and CZIB overlay."""
    map_data = []
    for e in events:
        if e.get("latitude") and e.get("longitude"):
            tier = e.get("alert_tier")
            color = TIER_COLOR_MAP.get(tier, TIER_COLOR_MAP[None])
            glow = TIER_GLOW_MAP.get(tier, TIER_GLOW_MAP[None])
            severity = e.get("severity_score", 20)
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

    # Build CZIB layer data
    czib_map_data = []
    if czib_data:
        for z in czib_data:
            coords = z.get("coordinates", "")
            if coords and "," in coords:
                try:
                    parts = coords.split(",")
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    czib_map_data.append({
                        "lat": lat,
                        "lon": lon,
                        "name": z.get("name", "CZIB Zone"),
                        "status": z.get("status", "Active"),
                        "countries": z.get("country_names", "—") or "—",
                        "valid_until": z.get("valid_until", "—") or "—",
                    })
                except (ValueError, IndexError):
                    continue

    if not map_data and not czib_map_data:
        st.info("📍 No geolocated events or CZIB zones to display")
        return

    layers = []

    # CZIB layer (if available) — red cross markers
    if czib_map_data:
        czib_layer = pdk.Layer(
            "ScatterplotLayer",
            data=czib_map_data,
            get_position=["lon", "lat"],
            get_fill_color=[239, 68, 68, 200],
            get_radius=15000,
            pickable=True,
            opacity=0.7,
            stroked=True,
            get_line_color=[255, 255, 255, 100],
            get_line_width=3,
        )
        layers.append(czib_layer)

    # Event glow layer
    if map_data:
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
        layers.append(glow_layer)

        # Event main layer
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
        layers.append(scatter_layer)

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
            {name}
          </div>
          <div style="color:#94A3B8; font-size:0.82em; line-height:1.6;">
            <span style="color:#64748B;">{type_or_status}</span> {value}<br/>
            <span style="color:#64748B;">{extra_label}</span> {extra_value}
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
            layers=layers,
            initial_view_state=view,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v11",
        ),
        width="stretch",
        height=550,
    )

    # Combined legend
    legend_items = [
        ("#EF4444", "Critical"),
        ("#F97316", "Alert"),
        ("#EAB308", "Watch"),
        ("#64748B", "No Alert"),
    ]
    if czib_map_data:
        legend_items.append(("#EF4444", "CZIB Zone ⚠️"))

    cols = st.columns(len(legend_items))
    for i, (color, label) in enumerate(legend_items):
        cols[i].markdown(
            f"<div style='text-align:center;font-size:0.75em;color:#94A3B8;'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:4px;'></span>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )
