"""
SIM — Event Table Component
Blueprint V20.1 §5.1

Filterable, sortable event dataframe with color-coded severity.
"""

import pandas as pd
import streamlit as st


def render_event_table(events: list[dict]):
    """Render a filterable, sortable event table."""
    if not events:
        st.info("📋 No events to display")
        return

    df = pd.DataFrame(events)

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        event_types = ["All"] + sorted(df["event_type"].dropna().unique().tolist())
        selected_type = st.selectbox("Event Type", event_types, key="filter_type")

    with col2:
        tiers = ["All", "CRITICAL", "ALERT", "WATCH", "None"]
        selected_tier = st.selectbox("Alert Tier", tiers, key="filter_tier")

    with col3:
        countries = ["All"] + sorted(df["country_iso"].dropna().unique().tolist())
        selected_country = st.selectbox("Country", countries, key="filter_country")

    with col4:
        min_severity = st.slider("Min Severity", 0, 100, 0, key="filter_severity")

    # Apply filters
    filtered = df.copy()
    if selected_type != "All":
        filtered = filtered[filtered["event_type"] == selected_type]
    if selected_tier != "All":
        if selected_tier == "None":
            filtered = filtered[filtered["alert_tier"].isna()]
        else:
            filtered = filtered[filtered["alert_tier"] == selected_tier]
    if selected_country != "All":
        filtered = filtered[filtered["country_iso"] == selected_country]
    if min_severity > 0:
        filtered = filtered[filtered["severity_score"] >= min_severity]

    st.caption(f"Showing {len(filtered)} of {len(df)} events")

    # Display columns
    display_cols = [
        "source_title", "event_type", "alert_tier",
        "severity_score", "system_confidence",
        "anchor_name_norm", "country_iso",
        "llm_provider", "llm_model",
        "status", "ingested_at",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    # Color coding for alert tier
    def color_tier(val):
        colors = {
            "CRITICAL": "background-color: rgba(220,38,38,0.2); color: #FCA5A5;",
            "ALERT": "background-color: rgba(234,88,12,0.2); color: #FDBA74;",
            "WATCH": "background-color: rgba(202,138,4,0.2); color: #FDE047;",
        }
        return colors.get(val, "")

    styled = filtered[available_cols].style
    if "alert_tier" in available_cols:
        styled = styled.map(color_tier, subset=["alert_tier"])

    st.dataframe(
        styled,
        use_container_width=True,
        height=500,
        column_config={
            "source_title": st.column_config.TextColumn("Title", width="large"),
            "event_type": st.column_config.TextColumn("Type", width="medium"),
            "severity_score": st.column_config.ProgressColumn("Severity", min_value=0, max_value=100),
            "system_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
    )
