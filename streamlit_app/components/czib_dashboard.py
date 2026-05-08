"""
SIM — CZIB (Conflict Zone Information Bulletin) Dashboard
EASA CZIB data viewer with status cards, country breakdown, and sync controls.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st


def _country_flag(iso: str | None) -> str:
    if not isinstance(iso, str) or len(iso) != 2:
        return "🌐"
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def _status_color(status: str) -> str:
    return {
        "Active": "#10B981",
        "Suspended": "#F59E0B",
        "Withdrawn": "#64748B",
    }.get(status, "#64748B")


def render_czib_dashboard(db_conn):
    """Render CZIB dashboard with zones list, stats, and sync button."""
    from services.cache import get_conn_key, get_czib_stats, get_czib_zones

    ck = get_conn_key(db_conn)

    st.markdown("#### ⚠️ EASA Conflict Zone Information Bulletins")
    st.caption("Data sourced from [EASA CZIB](https://www.easa.europa.eu/en/domains/air-operations/czibs). Updated automatically.")

    # Stats row
    try:
        stats = get_czib_stats(ck, db_conn)
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 Active Zones", stats.get("active", 0))
        s2.metric("🟡 Suspended", stats.get("suspended", 0))
        s3.metric("🌍 Affected Countries", stats.get("countries", 0))
    except Exception:
        st.warning("Could not load CZIB stats")

    st.divider()

    # Filters + Sync
    f1, f2 = st.columns([3, 1])
    with f1:
        show_status = st.segmented_control(
            "Filter by status",
            options=["Active", "Suspended", "Withdrawn", "All"],
            default="Active",
            key="czib_status_filter",
        )
    with f2:
        if st.button("🔄 Sync CZIB Data", width="stretch"):
            with st.spinner("Fetching from EASA..."):
                try:
                    from src.services.czib_client import sync_czib_to_db
                    result = sync_czib_to_db(db_conn)
                    st.success(f"Synced: {result['fetched']} fetched, {result['updated']} updated")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync failed: {e}")

    # Load zones
    try:
        zones = get_czib_zones(ck, db_conn, only_active=False)
    except Exception:
        st.error("Could not load CZIB zones")
        return

    if not zones:
        st.info("No CZIB zones in database. Click 'Sync CZIB Data' to fetch from EASA.")
        return

    if show_status and show_status != "All":
        zones = [z for z in zones if z.get("status") == show_status]

    st.caption(f"Showing {len(zones)} zone{'s' if len(zones) > 1 else ''}")

    from components.map_view import render_map
    render_map([], czib_data=zones)

    for zone in zones:
        _render_zone_card_native(zone)


def _render_zone_card_native(zone: dict):
    """Render a single CZIB zone card with native Streamlit components."""
    status = zone.get("status", "Unknown")
    name = zone.get("name", "Untitled")
    countries = zone.get("countries", []) or []
    country_names = zone.get("country_names", "")
    valid = zone.get("valid_until", "—")
    issued = zone.get("issued_date")
    coords = zone.get("coordinates", "")
    updated = zone.get("updated_at")

    issued_str = ""
    if issued and isinstance(issued, datetime):
        issued_str = issued.strftime("%Y-%m-%d")

    status_color = _status_color(status)
    flags = " ".join(_country_flag(c) for c in countries[:12])
    if len(countries) > 12:
        flags += f" +{len(countries) - 12}"

    with st.container(border=True):
        # Header row
        h1, h2 = st.columns([4, 1])
        with h1:
            st.markdown(
                f"<span style='display:inline-block;padding:3px 10px;border-radius:999px;font-size:0.75em;font-weight:700;color:{status_color};background:{status_color}15;border:1px solid {status_color}40;text-transform:uppercase;'>{status}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{name}**")
        with h2:
            st.caption("Valid until")
            st.write(f"**{valid or '—'}**")

        st.caption(f"{flags}")

        with st.expander("Details"):
            d1, d2 = st.columns(2)
            with d1:
                st.write(f"**Countries:** {country_names or '—'}")
                st.write(f"**Issued:** {issued_str or '—'}")
                st.write(f"**Coordinates:** {coords or '—'}")
            with d2:
                st.write(f"**Status:** {status}")
                st.write(f"**Valid Until:** {valid or '—'}")
                if updated and isinstance(updated, datetime):
                    st.write(f"**Last Updated:** {updated.strftime('%Y-%m-%d %H:%M UTC')}")
