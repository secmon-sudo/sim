"""
SIM — Alert Feed Component
Blueprint V20.1 §5.1

Real-time alert stream with severity bars, relative time, grouped by tier,
and expandable detail cards.
"""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())
_TIER_CFG = _UI_CONFIG["tiers"]


def _relative_time(dt: datetime | None) -> str:
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
    return dt.strftime("%b %d")


def _severity_bar_html(score: int, max_w: int = 120) -> str:
    if score >= 80:
        color = "#EF4444"
    elif score >= 65:
        color = "#F97316"
    elif score >= 45:
        color = "#EAB308"
    else:
        color = "#64748B"
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:{max_w}px;height:8px;background:#0B1120;border-radius:4px;overflow:hidden;border:1px solid #1E293B;">
        <div style="width:{score}%;height:100%;background:{color};border-radius:4px;box-shadow:0 0 8px {color}60;"></div>
      </div>
      <span style="font-size:0.8em;color:{color};font-weight:700;min-width:28px;">{score}</span>
    </div>
    """


def _country_flag(iso: str | None) -> str:
    if not iso or len(iso) != 2:
        return "🌐"
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def render_alert_feed(alerts: list[dict]):
    """Render grouped alert feed with visual severity bars."""
    if not alerts:
        st.success("✅ No active alerts in the last 24 hours")
        return

    # ── Summary KPI cards ──
    tier_counts = {"CRITICAL": 0, "ALERT": 0, "WATCH": 0}
    for a in alerts:
        t = a.get("alert_tier", "WATCH")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    st.markdown("#### 🚨 Alert Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔴 Critical", tier_counts.get("CRITICAL", 0))
    with c2:
        st.metric("🟠 Alert", tier_counts.get("ALERT", 0))
    with c3:
        st.metric("🟡 Watch", tier_counts.get("WATCH", 0))
    with c4:
        st.metric("Total", len(alerts))

    st.divider()

    # ── Group by tier ──
    groups = {"CRITICAL": [], "ALERT": [], "WATCH": []}
    for a in alerts:
        t = a.get("alert_tier", "WATCH")
        groups.setdefault(t, []).append(a)

    for tier in ["CRITICAL", "ALERT", "WATCH"]:
        group = groups.get(tier, [])
        if not group:
            continue

        cfg = _TIER_CFG.get(tier, _TIER_CFG["WATCH"])

        # Tier section header
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin:16px 0 8px 0;
                padding-bottom:6px;
                border-bottom:2px solid {cfg['color']}40;
            ">
              <span style="font-size:1.1em;font-weight:800;color:{cfg['color']};">
                {cfg['icon']} {cfg['label']}
              </span>
              <span style="font-size:0.8em;color:#64748B;">({len(group)} alert{'s' if len(group) > 1 else ''})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for alert in group:
            _render_alert_card(alert, cfg)


def _render_alert_card(alert: dict, cfg: dict):
    """Render a single alert card."""
    title = str(alert.get("source_title") or "Untitled")
    etype = str(alert.get("event_type") or "unknown").replace("_", " ").title()
    severity = int(alert.get("severity_score") or 0)
    confidence = float(alert.get("system_confidence") or 0)
    anchor = str(alert.get("anchor_name_norm") or "—")
    country = str(alert.get("country_iso") or "—")
    flag = _country_flag(alert.get("country_iso"))
    ingested = alert.get("ingested_at")
    url = str(alert.get("source_url") or "")
    domain = str(alert.get("source_domain") or "—")
    eid = str(alert.get("id", "?"))[:8]

    rel = _relative_time(ingested)
    sev_bar = _severity_bar_html(severity)

    st.markdown(
        f"""
        <div style="
            background: {cfg['bg']};
            border: 1px solid {cfg['border']};
            border-left: 4px solid {cfg['color']};
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            backdrop-filter: blur(4px);
        ">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;min-width:0;">
              <div style="font-weight:700;color:#F8FAFC;font-size:0.95em;line-height:1.4;word-break:break-word;margin-bottom:6px;">
                {html.escape(title[:160])}{"..." if len(title) > 160 else ""}
              </div>
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:0.78em;color:#94A3B8;">
                <span>{flag} {html.escape(country)}</span>
                <span>📍 {html.escape(anchor)}</span>
                <span>🏷️ {etype}</span>
                <span>🔮 {confidence:.0%}</span>
                <span>🌐 {html.escape(domain)}</span>
              </div>
            </div>
            <div style="text-align:right;flex-shrink:0;min-width:90px;">
              <div style="font-size:0.75em;color:#64748B;margin-bottom:4px;">{rel}</div>
              {sev_bar}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expand for details
    with st.expander("View details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Severity Score:** {severity}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.write(f"**Anchor:** {anchor}")
            st.write(f"**Country:** {country}")
        with col2:
            st.write(f"**Event Type:** {etype}")
            st.write(f"**Domain:** {domain}")
            st.write(f"**ID:** #{eid}")
            if url:
                st.markdown(f"[🔗 Open Source Article]({url})")

        # Content preview
        text = str(alert.get("canonical_text") or "")
        if text:
            st.caption("Content")
            st.markdown(
                f"""<div style="background:#0B1120;border:1px solid #1E293B;border-radius:6px;padding:10px 12px;font-size:0.82em;color:#94A3B8;max-height:120px;overflow-y:auto;line-height:1.5;">
                {html.escape(text[:500])}{"..." if len(text) > 500 else ""}
                </div>""",
                unsafe_allow_html=True,
            )
