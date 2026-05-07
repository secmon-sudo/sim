"""
SIM — CZIB (Conflict Zone Information Bulletin) Dashboard
EASA CZIB data viewer with status cards, country breakdown, and sync controls.
"""

import html
from datetime import datetime
from pathlib import Path

import streamlit as st


def _country_flag(iso: str | None) -> str:
    if not iso or len(iso) != 2:
        return "🌐"
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def _status_badge(status: str) -> str:
    colors = {
        "Active": ("#10B981", "rgba(16,185,129,0.15)"),
        "Suspended": ("#F59E0B", "rgba(245,158,11,0.15)"),
        "Withdrawn": ("#64748B", "rgba(100,116,139,0.15)"),
    }
    c, bg = colors.get(status, ("#64748B", "rgba(100,116,139,0.15)"))
    return f"""
    <span style="display:inline-block;padding:3px 10px;border-radius:999px;font-size:0.7em;font-weight:700;color:{c};background:{bg};border:1px solid {c}40;text-transform:uppercase;">
      {status}
    </span>
    """


def render_czib_dashboard(db_conn):
    """Render CZIB dashboard with zones list, stats, and sync button."""
    from streamlit_app.services.cache import get_czib_stats, get_czib_zones

    st.markdown("#### ⚠️ EASA Conflict Zone Information Bulletins")
    st.caption("Data sourced from [EASA CZIB](https://www.easa.europa.eu/en/domains/air-operations/czibs). Updated automatically.")

    # Stats row
    try:
        stats = get_czib_stats(db_conn)
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 Active Zones", stats.get("active", 0))
        s2.metric("🟡 Suspended", stats.get("suspended", 0))
        s3.metric("🌍 Affected Countries", stats.get("countries", 0))
    except Exception:
        st.warning("Could not load CZIB stats")

    st.divider()

    # Filters
    f1, f2 = st.columns([3, 1])
    with f1:
        show_status = st.segmented_control(
            "Filter by status",
            options=["Active", "Suspended", "Withdrawn", "All"],
            default="Active",
            key="czib_status_filter",
        )
    with f2:
        if st.button("🔄 Sync CZIB Data", use_container_width=True):
            with st.spinner("Fetching from EASA..."):
                try:
                    from src.services.czib_client import sync_czib_to_db
                    result = sync_czib_to_db(db_conn)
                    st.success(f"Synced: {result['fetched']} fetched, {result['updated']} updated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync failed: {e}")

    # Load zones
    try:
        zones = get_czib_zones(db_conn, only_active=False)
    except Exception:
        st.error("Could not load CZIB zones")
        return

    if not zones:
        st.info("No CZIB zones in database. Click 'Sync CZIB Data' to fetch from EASA.")
        return

    # Filter
    if show_status and show_status != "All":
        zones = [z for z in zones if z.get("status") == show_status]

    st.caption(f"Showing {len(zones)} zone{'s' if len(zones) > 1 else ''}")

    # Zone cards
    for zone in zones:
        _render_zone_card(zone)


def _render_zone_card(zone: dict):
    """Render a single CZIB zone card."""
    status = zone.get("status", "Unknown")
    name = zone.get("name", "Untitled")
    countries = zone.get("countries", []) or []
    country_names = zone.get("country_names", "")
    valid = zone.get("valid_until", "—")
    issued = zone.get("issued_date")
    coords = zone.get("coordinates", "")
    updated = zone.get("updated_at")

    # Format dates
    issued_str = ""
    if issued and isinstance(issued, datetime):
        issued_str = issued.strftime("%Y-%m-%d")

    # Country flags
    flags = " ".join(_country_flag(c) for c in countries[:12])
    if len(countries) > 12:
        flags += f" +{len(countries) - 12}"

    st.markdown(
        f"""
        <div style="
            background: #151E32;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
        ">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;min-width:0;">
              <div style="margin-bottom:6px;">{_status_badge(status)}</div>
              <div style="font-weight:700;color:#F8FAFC;font-size:0.95em;line-height:1.3;word-break:break-word;">
                {html.escape(name)}
              </div>
              <div style="margin-top:6px;font-size:0.8em;color:#94A3B8;">
                {flags}
              </div>
            </div>
            <div style="text-align:right;flex-shrink:0;min-width:100px;">
              <div style="font-size:0.75em;color:#64748B;">Valid until</div>
              <div style="font-size:0.85em;color:#F8FAFC;font-weight:600;">{html.escape(valid) or '—'}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
