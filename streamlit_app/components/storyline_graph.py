"""
SIM — Storyline Timeline Component
Blueprint V20.1 §5.1

Chronological timeline visualization of linked incidents.
Replaces the old hairball graph with an elegant, responsive vertical timeline.
"""

import streamlit as st
import html
import textwrap
from datetime import datetime

# Tier colors
TIER_COLORS = {
    "CRITICAL": "#EF4444",
    "ALERT":    "#F97316",
    "WATCH":    "#EAB308",
    None:       "#64748B",
}

def render_storyline_graph(events: list[dict]):
    """Render an interactive, chronological storyline timeline."""
    if not events:
        st.info("🔗 No storyline data available")
        return

    # Group by storyline_id
    storylines = {}
    for e in events:
        sid = str(e.get("storyline_id", ""))
        if sid and sid != "None":
            if sid not in storylines:
                storylines[sid] = []
            storylines[sid].append(e)

    # Filter: only show storylines with 2+ events
    multi_event_sls = {sid: grp for sid, grp in storylines.items() if len(grp) >= 2}
    if not multi_event_sls:
        st.info("🔗 No multi-event storylines to display")
        return

    # Sort each group chronologically (oldest first)
    for sid in multi_event_sls:
        multi_event_sls[sid].sort(key=lambda x: x.get("occurred_at_est") or datetime.min)

    # Sort storylines by their newest event (most recently active storyline first)
    sorted_storylines = sorted(
        multi_event_sls.items(),
        key=lambda x: x[1][-1].get("occurred_at_est") or datetime.min,
        reverse=True
    )

    st.markdown(f"**Found {len(sorted_storylines)} active storylines.**")

    # CSS for the timeline
    st.markdown("""
    <style>
    .storyline-card {
        background: #151E32;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
    }
    .storyline-header {
        font-size: 1.15em;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .storyline-meta {
        font-size: 0.8em;
        color: #94A3B8;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px dashed #334155;
    }
    .timeline {
        position: relative;
        padding-left: 30px;
    }
    .timeline::before {
        content: '';
        position: absolute;
        top: 0;
        bottom: 0;
        left: 9px;
        width: 2px;
        background: #334155;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 20px;
    }
    .timeline-item:last-child {
        margin-bottom: 0;
    }
    .timeline-dot {
        position: absolute;
        left: -26px;
        top: 4px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        border: 2px solid #151E32;
    }
    .timeline-content {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
    }
    .timeline-date {
        font-size: 0.75em;
        color: #64748B;
        margin-bottom: 4px;
        font-family: monospace;
    }
    .timeline-title {
        font-size: 0.9em;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .timeline-tags {
        font-size: 0.7em;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .tag {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 2px 6px;
        border-radius: 4px;
        color: #CBD5E1;
    }
    @keyframes pulse-dot {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .pulse-dot {
        animation: pulse-dot 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    for sid, group in sorted_storylines:
        hint = group[0].get("storyline_hint", "")
        if not hint:
            et = group[0].get("event_type", "incident").replace("_", " ").title()
            country = group[0].get("country_iso", "?")
            hint = f"{et} — {country}"
        
        hint = html.escape(hint)
        
        countries = list(set([e.get("country_iso") for e in group if e.get("country_iso")]))
        flags = []
        for c in countries:
            if isinstance(c, str) and len(c) == 2:
                flags.append(chr(0x1F1E6 + ord(c[0].upper()) - 65) + chr(0x1F1E6 + ord(c[1].upper()) - 65))
        
        flag_str = "".join(flags)
        
        # Build HTML
        html_content = textwrap.dedent(f"""
        <div class="storyline-card">
            <div class="storyline-header">
                {flag_str} {hint}
            </div>
            <div class="storyline-meta">
                ID: {sid[:8]} &nbsp;|&nbsp; {len(group)} events
            </div>
            <div class="timeline">
        """)

        for e in group:
            tier = e.get("alert_tier")
            color = TIER_COLORS.get(tier, TIER_COLORS[None])
            severity = e.get("severity_score", 0)
            
            pulse_class = "pulse-dot" if tier == "CRITICAL" else ""
            
            date_val = e.get("occurred_at_est")
            date_str = date_val.strftime("%Y-%m-%d %H:%M") if hasattr(date_val, 'strftime') else str(date_val)[:16]
            
            title = e.get("source_title") or "Untitled Event"
            title = html.escape(title[:120] + ("..." if len(title) > 120 else ""))
            
            event_type = (e.get("event_type") or "unknown").replace("_", " ").title()
            anchor = e.get("anchor_name_norm") or e.get("country_iso") or "Unknown"
            
            html_content += textwrap.dedent(f"""
                <div class="timeline-item">
                    <div class="timeline-dot {pulse_class}" style="background: {color};"></div>
                    <div class="timeline-content">
                        <div class="timeline-date">{date_str}</div>
                        <div class="timeline-title">{title}</div>
                        <div class="timeline-tags">
                            <span class="tag" style="color:{color}; border-color:{color}40;">Tier: {tier or 'None'}</span>
                            <span class="tag">Sev: {severity}</span>
                            <span class="tag">📍 {html.escape(anchor)}</span>
                            <span class="tag">📌 {html.escape(event_type)}</span>
                        </div>
                    </div>
                </div>
            """)
            
        html_content += textwrap.dedent("""
            </div>
        </div>
        """)
        
        st.markdown(html_content, unsafe_allow_html=True)
