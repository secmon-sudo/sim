"""
SIM — Event Table Component
Blueprint V20.1 §5.1

Modern card-based event list with readable dates, status badges,
severity bars, expandable details, and rich filters.
"""

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import json

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())
_TIERS = _UI_CONFIG["tiers"]
_STATUS_CFG = _UI_CONFIG["status"]


def _relative_time(dt: datetime | None) -> str:
    """Human-readable relative time."""
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 604800:
        return f"{seconds // 86400}d ago"
    return dt.strftime("%b %d, %Y")


def _format_datetime(dt: datetime | None) -> str:
    """Format datetime for tooltip."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _severity_bar(score: int, width: int = 80) -> str:
    """HTML mini severity bar."""
    if score >= 80:
        color = "#EF4444"
    elif score >= 65:
        color = "#F97316"
    elif score >= 45:
        color = "#EAB308"
    elif score >= 30:
        color = "#3B82F6"
    else:
        color = "#64748B"
    return f"""
    <div style="display:flex;align-items:center;gap:6px;">
      <div style="width:{width}px;height:6px;background:#1E293B;border-radius:3px;overflow:hidden;">
        <div style="width:{score}%;height:100%;background:{color};border-radius:3px;"></div>
      </div>
      <span style="font-size:0.75em;color:#94A3B8;font-weight:600;">{score}</span>
    </div>
    """


def _badge(text: str, color: str, bg: str) -> str:
    """HTML badge pill."""
    return f"""
    <span style="
        display:inline-block;
        padding:2px 8px;
        border-radius:999px;
        font-size:0.7em;
        font-weight:700;
        letter-spacing:0.02em;
        color:{color};
        background:{bg};
        border:1px solid {color}40;
        text-transform:uppercase;
    ">{html.escape(text)}</span>
    """


def _tier_badge(tier: str | None) -> str:
    if not tier or tier == "None":
        return _badge("none", "#64748B", "rgba(100,116,139,0.12)")
    cfg = _TIERS.get(tier, _TIERS["WATCH"])
    return _badge(cfg["label"], cfg["color"], cfg["bg"])


def _status_badge(status: str | None) -> str:
    if not status:
        return _badge("unknown", "#64748B", "rgba(100,116,139,0.12)")
    cfg = _STATUS_CFG.get(status, _STATUS_CFG["raw"])
    return _badge(cfg["label"], cfg["color"], cfg["bg"])


def _event_type_label(et: str | None) -> str:
    if not et:
        return "other"
    return et.replace("_", " ").title()


def _country_flag(iso: str | None) -> str:
    """Convert ISO2 to emoji flag."""
    if not iso or len(iso) != 2:
        return "🌐"
    # Regional indicator symbols: A=🇦 (0x1F1E6), offset from ASCII 'A' (0x41)
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def render_event_table(events: list[dict]):
    """Render a modern, filterable event list with cards and expandable details."""
    if not events:
        st.info("📋 No events to display")
        return

    df = pd.DataFrame(events)

    # ── Filters Row ──
    st.markdown("#### 🔍 Filters")
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
    with f1:
        search_text = st.text_input("Search title", "", key="evt_search", placeholder="e.g. airport attack")
    with f2:
        types = ["All"] + sorted([t for t in df["event_type"].dropna().unique() if t])
        sel_type = st.selectbox("Event Type", types, key="evt_type")
    with f3:
        tiers = ["All", "CRITICAL", "ALERT", "WATCH", "None"]
        sel_tier = st.selectbox("Alert Tier", tiers, key="evt_tier")
    with f4:
        countries = ["All"] + sorted([c for c in df["country_iso"].dropna().unique() if c])
        sel_country = st.selectbox("Country", countries, key="evt_country")
    with f5:
        sev_range = st.select_slider(
            "Severity",
            options=[0, 30, 45, 65, 80, 100],
            value=(0, 100),
            key="evt_sev",
        )

    # Apply filters
    filtered = df.copy()
    if search_text.strip():
        q = search_text.lower()
        mask = filtered["source_title"].fillna("").str.lower().str.contains(q)
        if "canonical_text" in filtered.columns:
            mask = mask | filtered["canonical_text"].fillna("").str.lower().str.contains(q)
        filtered = filtered[mask]
    if sel_type != "All":
        filtered = filtered[filtered["event_type"] == sel_type]
    if sel_tier != "All":
        if sel_tier == "None":
            filtered = filtered[filtered["alert_tier"].isna()]
        else:
            filtered = filtered[filtered["alert_tier"] == sel_tier]
    if sel_country != "All":
        filtered = filtered[filtered["country_iso"] == sel_country]
    filtered = filtered[
        (filtered["severity_score"].fillna(0) >= sev_range[0]) &
        (filtered["severity_score"].fillna(0) <= sev_range[1])
    ]

    # Summary bar
    total_all = len(df)
    total_filt = len(filtered)
    crit_n = len(filtered[filtered["alert_tier"] == "CRITICAL"])
    alert_n = len(filtered[filtered["alert_tier"] == "ALERT"])
    watch_n = len(filtered[filtered["alert_tier"] == "WATCH"])

    st.markdown(
        f"""
        <div style="display:flex;gap:16px;align-items:center;margin:8px 0 16px 0;">
          <span style="color:#94A3B8;font-size:0.85em;">
            Showing <b style="color:#F8FAFC;">{total_filt}</b> of <b style="color:#F8FAFC;">{total_all}</b> events
          </span>
          <span style="color:#EF4444;font-size:0.8em;font-weight:600;">🔴 {crit_n}</span>
          <span style="color:#F97316;font-size:0.8em;font-weight:600;">🟠 {alert_n}</span>
          <span style="color:#EAB308;font-size:0.8em;font-weight:600;">🟡 {watch_n}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Event Cards ──
    for _, row in filtered.iterrows():
        _render_event_card(row)

    if total_filt == 0:
        st.info("No events match the selected filters.")


def _render_event_card(row: pd.Series):
    """Render a single event card."""
    eid = str(row.get("id", "?"))[:8]
    title = str(row.get("source_title") or "Untitled")
    etype = _event_type_label(row.get("event_type"))
    tier = row.get("alert_tier")
    status = row.get("status", "raw")
    severity = int(row.get("severity_score") or 0)
    confidence = float(row.get("system_confidence") or 0)
    anchor = str(row.get("anchor_name_norm") or "—")
    country = str(row.get("country_iso") or "—")
    flag = _country_flag(row.get("country_iso"))
    ingested = row.get("ingested_at")
    occurred = row.get("occurred_at_est")
    domain = str(row.get("source_domain") or "—")
    url = str(row.get("source_url") or "")
    provider = str(row.get("llm_provider") or "—")
    model = str(row.get("llm_model") or "—")

    tier_badge = _tier_badge(tier)
    status_badge = _status_badge(status)
    sev_bar = _severity_bar(severity)
    rel_time = _relative_time(ingested)
    full_time = _format_datetime(ingested)

    # Card header
    st.markdown(
        f"""
        <div style="
            background: #151E32;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            transition: all 0.2s;
        ">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;min-width:0;">
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
                {tier_badge}
                {status_badge}
                <span style="font-size:0.75em;color:#64748B;">{etype}</span>
              </div>
              <div style="font-weight:600;color:#F8FAFC;font-size:0.95em;line-height:1.4;word-break:break-word;">
                {html.escape(title[:140])}{"..." if len(title) > 140 else ""}
              </div>
            </div>
            <div style="text-align:right;flex-shrink:0;">
              <div style="font-size:0.8em;color:#64748B;white-space:nowrap;" title="{full_time}">
                {rel_time}
              </div>
              <div style="margin-top:6px;">{sev_bar}</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:12px;margin-top:10px;font-size:0.78em;color:#64748B;flex-wrap:wrap;">
            <span>{flag} {html.escape(country)}</span>
            <span>📍 {html.escape(anchor)}</span>
            <span>🔮 {confidence:.0%}</span>
            <span>🌐 {html.escape(domain)}</span>
            <span title="ID: {eid}">#{eid}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable details
    with st.expander("Details", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.caption("Timing")
            st.write(f"**Ingested:** {_format_datetime(ingested)}")
            st.write(f"**Occurred (est.):** {_format_datetime(occurred)}")
            st.write(f"**Time certainty:** {row.get('time_certainty') or '—'}")
        with d2:
            st.caption("ML / Source")
            st.write(f"**Provider:** {provider}")
            st.write(f"**Model:** {model}")
            st.write(f"**Confidence:** {confidence:.2%}")
            if url:
                st.markdown(f"[🔗 Open Source]({url})")

        # Raw text preview
        raw = str(row.get("canonical_text") or row.get("raw_text") or "")
        if raw:
            st.caption("Content Preview")
            st.markdown(
                f"""<div style="
                    background:#0B1120;
                    border:1px solid #1E293B;
                    border-radius:6px;
                    padding:10px 12px;
                    font-size:0.82em;
                    color:#94A3B8;
                    max-height:140px;
                    overflow-y:auto;
                    line-height:1.5;
                ">{html.escape(raw[:600])}{"..." if len(raw) > 600 else ""}</div>""",
                unsafe_allow_html=True,
            )
