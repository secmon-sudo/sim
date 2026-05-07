"""
SIM — CZIB Map Overlay Component
Adds EASA Conflict Zone markers to the PyDeck map.
"""

import streamlit as st


def _parse_coords(coord_str: str) -> tuple[float, float] | None:
    """Parse 'lat, lon' string to (lat, lon)."""
    if not coord_str or "," not in coord_str:
        return None
    try:
        parts = coord_str.split(",")
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return (lat, lon)
    except (ValueError, IndexError):
        return None


def get_czib_map_data(db_conn) -> list[dict]:
    """Fetch active CZIB zones with coordinates for map overlay."""
    try:
        rows = db_conn.execute(
            """SELECT name, coordinates, status, country_names, valid_until
               FROM czib_zones
               WHERE status = 'Active'
                 AND coordinates <> ''
               ORDER BY updated_at DESC"""
        ).fetchall()
    except Exception:
        return []

    data = []
    for row in rows:
        name, coords, status, countries, valid = row
        parsed = _parse_coords(coords)
        if parsed:
            data.append({
                "lat": parsed[0],
                "lon": parsed[1],
                "name": name,
                "status": status,
                "countries": countries or "—",
                "valid_until": valid or "—",
            })
    return data


def render_czib_layer(czib_data: list[dict]):
    """Render CZIB markers as HTML overlay (since PyDeck layer merging is complex)."""
    if not czib_data:
        return

    st.markdown("##### 🛡️ EASA Active Conflict Zones")
    for zone in czib_data[:10]:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding:6px 10px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:6px;margin-bottom:4px;font-size:0.8em;">
              <span style="color:#EF4444;font-size:1.1em;">⚠️</span>
              <span style="color:#F8FAFC;font-weight:600;">{zone['name']}</span>
              <span style="color:#64748B;">({zone['countries']})</span>
              <span style="color:#94A3B8;margin-left:auto;">Valid: {zone['valid_until']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if len(czib_data) > 10:
        st.caption(f"... and {len(czib_data) - 10} more active zones")
