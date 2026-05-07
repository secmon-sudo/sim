"""
SIM — Storyline Graph Component
Blueprint V20.1 §5.1

NetworkX + streamlit-agraph visualization of linked incident storylines.
Improved layout with tier-based node coloring and storyline hub styling.
"""

import streamlit as st

try:
    from streamlit_agraph import Config, Edge, Node, agraph
    _AGRAPH_AVAILABLE = True
except ImportError:
    _AGRAPH_AVAILABLE = False

# Event type → color mapping (expanded)
TYPE_COLORS = {
    "bomb_threat":        "#EF4444",
    "active_shooter":     "#B91C1C",
    "hijacking":          "#7F1D1D",
    "security_incident":  "#F97316",
    "emergency_landing":  "#EAB308",
    "runway_incursion":   "#D97706",
    "fire_on_board":      "#EF4444",
    "engine_failure":     "#FB923C",
    "drone_incursion":    "#8B5CF6",
    "suspicious_package": "#E11D48",
    "evacuation":         "#F43F5E",
    "unruly_passenger":   "#06B6D4",
    "bird_strike":        "#10B981",
    "laser_attack":       "#A855F7",
    "depressurization":   "#EC4899",
    "hotel_attack":       "#DC2626",
    "hotel_bombing":      "#991B1B",
    "resort_attack":      "#EA580C",
    "drone_attack":       "#7C3AED",
    "mass_shooting":      "#DC2626",
    "mass_casualty":      "#B91C1C",
    "suicide_bombing":    "#7F1D1D",
    "missile_strike":     "#F59E0B",
    "airstrike":          "#D97706",
    "war_escalation":     "#EF4444",
    "geopolitical":       "#3B82F6",
}

# Tier → node border color
TIER_BORDER = {
    "CRITICAL": "#EF4444",
    "ALERT":    "#F97316",
    "WATCH":    "#EAB308",
    None:       "#475569",
}


def render_storyline_graph(events: list[dict]):
    """Render an interactive storyline graph using streamlit-agraph."""
    if not _AGRAPH_AVAILABLE:
        st.info("🔗 Storyline graph requires `streamlit-agraph`. Install with: `pip install streamlit-agraph`")
        if events:
            st.caption(f"{len(events)} storyline events available in database.")
        return

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

    if not storylines:
        st.info("🔗 No linked storylines found")
        return

    # Filter: only show storylines with 2+ events
    multi_event_sls = {sid: grp for sid, grp in storylines.items() if len(grp) >= 2}
    if not multi_event_sls:
        st.info("🔗 No multi-event storylines to display")
        return

    # Build graph
    nodes = []
    edges = []
    seen_nodes = set()

    for sid, group in multi_event_sls.items():
        hub_id = f"story_{sid[:8]}"
        hint = group[0].get("storyline_hint", "")
        if not hint:
            # Build hint from first event's type + country
            et = group[0].get("event_type", "incident").replace("_", " ").title()
            country = group[0].get("country_iso", "?")
            hint = f"{et} — {country}"

        if hub_id not in seen_nodes:
            nodes.append(Node(
                id=hub_id,
                label=hint[:35] if hint else f"Storyline {sid[:8]}",
                size=35,
                color="rgba(99,102,241,0.25)",
                borderColor="#6366F1",
                shape="diamond",
                font={"color": "#94A3B8", "size": 11, "face": "Inter"},
            ))
            seen_nodes.add(hub_id)

        for e in group:
            eid = str(e["id"])[:10]
            if eid not in seen_nodes:
                event_type = e.get("event_type", "other")
                color = TYPE_COLORS.get(event_type, "#64748B")
                tier = e.get("alert_tier")
                border = TIER_BORDER.get(tier, "#475569")
                severity = e.get("severity_score", 20)
                anchor = e.get("anchor_name_norm") or "?"

                label = f"{anchor}\n{event_type.replace('_', ' ')[:16]}"
                nodes.append(Node(
                    id=eid,
                    label=label,
                    size=max(18, severity // 4),
                    color=f"{color}30",
                    borderColor=border,
                    shape="dot",
                    font={"color": "#F8FAFC", "size": 10, "face": "Inter"},
                ))
                seen_nodes.add(eid)

            edges.append(Edge(
                source=hub_id,
                target=eid,
                color="#334155",
                width=1.5,
                dashes=False,
            ))

    st.caption(
        f"📊 {len(multi_event_sls)} storylines, {len(nodes) - len(multi_event_sls)} events, "
        f"{len(edges)} connections"
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("#### 🔗 Storyline Controls")
        physics = st.toggle("Physics simulation", value=True, key="sg_physics")
        hierarchical = st.toggle("Hierarchical layout", value=False, key="sg_hier")

    config = Config(
        width="100%",
        height=550,
        directed=False,
        physics=physics,
        hierarchical=hierarchical,
        nodeHighlightBehavior=True,
        highlightColor="#6366F1",
        collapsible=True,
        # Better physics params
        solver="forceAtlas2Based" if physics else "barnesHut",
        stabilization=True,
    )

    agraph(nodes=nodes, edges=edges, config=config)

    # Legend
    st.markdown(
        """
        <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-top:8px;font-size:0.75em;color:#64748B;">
          <span>◆ <b style="color:#6366F1;">Storyline Hub</b></span>
          <span>● <b style="color:#EF4444;">Critical</b> border</span>
          <span>● <b style="color:#F97316;">Alert</b> border</span>
          <span>● <b style="color:#EAB308;">Watch</b> border</span>
          <span>● <b style="color:#475569;">No Alert</b> border</span>
          <span>Size ∝ Severity</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
