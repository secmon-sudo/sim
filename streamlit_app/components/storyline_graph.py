"""
SIM — Storyline Graph Component
Blueprint V20.1 §5.1

NetworkX + streamlit-agraph visualization of linked incident storylines.
"""

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


# Event type → color mapping
TYPE_COLORS = {
    "bomb_threat":        "#DC2626",
    "active_shooter":     "#991B1B",
    "hijacking":          "#7F1D1D",
    "security_incident":  "#EA580C",
    "emergency_landing":  "#F59E0B",
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
}


def render_storyline_graph(events: list[dict]):
    """Render an interactive storyline graph using streamlit-agraph."""
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

    # Build graph
    nodes = []
    edges = []
    seen_nodes = set()

    for sid, group in storylines.items():
        if len(group) < 2:
            continue  # Single-event storylines are not interesting

        # Storyline hub node
        hub_id = f"story_{sid[:8]}"
        hint = group[0].get("storyline_hint", "Storyline")
        if hub_id not in seen_nodes:
            nodes.append(Node(
                id=hub_id,
                label=hint[:40] if hint else f"Storyline {sid[:8]}",
                size=30,
                color="#6366F1",
                shape="diamond",
            ))
            seen_nodes.add(hub_id)

        for e in group:
            eid = str(e["id"])[:8]
            if eid not in seen_nodes:
                event_type = e.get("event_type", "other_aviation_related")
                color = TYPE_COLORS.get(event_type, "#64748B")
                severity = e.get("severity_score", 20)

                nodes.append(Node(
                    id=eid,
                    label=f"{e.get('anchor_name_norm') or '?'}\n{event_type[:15]}",
                    size=max(15, severity // 5),
                    color=color,
                    shape="dot",
                ))
                seen_nodes.add(eid)

            edges.append(Edge(
                source=hub_id,
                target=eid,
                color="#475569",
                width=1,
            ))

    if not nodes:
        st.info("🔗 No multi-event storylines to display")
        return

    st.caption(f"📊 {len(storylines)} storylines, {len(nodes) - len(storylines)} events")

    config = Config(
        width=800,
        height=500,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#6366F1",
        collapsible=True,
    )

    agraph(nodes=nodes, edges=edges, config=config)
