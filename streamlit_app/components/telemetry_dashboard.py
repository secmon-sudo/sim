"""
SIM — Telemetry Dashboard Component
Blueprint V20.1 §5.1

Pipeline health metrics, LLM quota tracking, and per-account status.
"""

import json

import streamlit as st


def render_telemetry(stats: dict, llm_router=None):
    """Render pipeline telemetry dashboard with dynamic quota from LLMRouter."""
    col1, col2, col3, col4 = st.columns(4)

    # Dynamic quota from LLMRouter
    total_quota = llm_router.total_daily_quota if llm_router else 1000
    total_used = llm_router.total_daily_used if llm_router else stats.get("llm_calls_24h", 0)

    col1.metric("🤖 LLM Calls (24h)", stats.get("llm_calls_24h", 0))
    col2.metric("📊 Tokens Used (24h)", f"{stats.get('tokens_used_24h', 0):,}")
    col3.metric("🔒 Stale Locks (1h)", stats.get("stale_locks_1h", 0))
    col4.metric("📈 Quota Remaining", f"{total_quota - total_used} / {total_quota}")

    st.divider()

    # Event pipeline status
    event_counts = stats.get("event_counts", {})
    if event_counts:
        st.subheader("Pipeline Status")
        pipeline_cols = st.columns(len(event_counts))
        status_colors = {
            "raw": "🔵", "deduped": "🟣", "locked": "🔒",
            "classified": "🟡", "scored": "🟠", "reconciled": "🟢", "archived": "⚪",
        }
        for i, (status, count) in enumerate(sorted(event_counts.items())):
            icon = status_colors.get(status, "⬜")
            pipeline_cols[i].metric(f"{icon} {status}", count)

    # Last pipeline run info
    last_run = stats.get("last_run")
    if last_run:
        st.divider()
        st.subheader("Last Pipeline Run")
        run_cols = st.columns(3)
        run_cols[0].metric(
            "Status",
            "✅ Success" if last_run.get("success") else "❌ Failed",
        )
        run_cols[1].metric("Duration", f"{last_run.get('duration_seconds', 0):.1f}s")
        run_cols[2].metric("Run ID", last_run.get("run_id", "—"))

        # Pass-level details
        with st.expander("Pass Details"):
            for pass_name in ["pass_a", "pass_b", "pass_c", "pass_d", "pass_e", "pass_f"]:
                data = last_run.get(pass_name)
                if data:
                    st.json(data)

    # Per-account LLM status
    if llm_router:
        st.divider()
        st.subheader("Provider Account Status")
        status_data = []
        for acct in llm_router.accounts:
            remaining = acct.rpd - acct.bucket._daily_used
            pct = acct.bucket.utilization_pct
            status_data.append({
                "Provider": f"{acct.provider}/{acct.account_id}",
                "Model": acct.model,
                "Status": acct.status.value,
                "Used / Limit": f"{acct.bucket._daily_used} / {acct.rpd}",
                "Remaining": remaining,
                "Utilization %": pct,
                "RPM": acct.rpm,
                "Errors": acct.daily_errors,
            })
        st.dataframe(status_data, use_container_width=True)
