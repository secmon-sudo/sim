"""
SIM — Alert Feed Component
Blueprint V20.1 §5.1

Real-time alert stream showing CRITICAL/ALERT/WATCH events.
"""

import html
import json
from pathlib import Path

import streamlit as st

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())


def render_alert_feed(alerts: list[dict]):
    """Render a real-time alert feed grouped by tier."""
    if not alerts:
        st.success("✅ No active alerts in the last 24 hours")
        return

    # Count by tier
    tier_counts = {}
    for a in alerts:
        t = a.get("alert_tier", "WATCH")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # Summary metrics
    cols = st.columns(3)
    for i, tier in enumerate(["CRITICAL", "ALERT", "WATCH"]):
        count = tier_counts.get(tier, 0)
        info = _UI_CONFIG["tiers"][tier]
        cols[i].metric(
            f"{info['icon']} {info['label']}",
            count,
        )

    st.divider()

    # Alert cards
    for alert in alerts:
        tier = alert.get("alert_tier", "WATCH")
        tier_info = _UI_CONFIG["tiers"].get(tier, _UI_CONFIG["tiers"]["WATCH"])

        # Escape all user-provided / LLM-generated text for HTML safety
        safe_title = html.escape(str(alert.get("source_title") or "Untitled"))[:100]
        safe_type = html.escape(str(alert.get("event_type") or "unknown"))
        safe_anchor = html.escape(str(alert.get("anchor_name_norm") or "—"))
        safe_country = html.escape(str(alert.get("country_iso") or "—"))
        severity = alert.get("severity_score", 0)
        confidence = alert.get("system_confidence", 0)

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: {tier_info['bg']};
                    border-left: 4px solid {tier_info['color']};
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin-bottom: 8px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.85em; color: {tier_info['color']}; font-weight: 700;">
                            {tier_info['icon']} {tier}
                        </span>
                        <span style="font-size: 0.75em; color: #94A3B8;">
                            Severity: {severity} &nbsp;|&nbsp;
                            Conf: {confidence:.2f}
                        </span>
                    </div>
                    <div style="margin-top: 6px; font-weight: 600; color: #F1F5F9;">
                        {safe_title}
                    </div>
                    <div style="margin-top: 4px; font-size: 0.8em; color: #94A3B8;">
                        {safe_type} &nbsp;•&nbsp;
                        {safe_anchor} &nbsp;•&nbsp;
                        {safe_country}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
