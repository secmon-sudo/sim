"""
SIM — Telemetry Dashboard Component
Blueprint V20.1 §5.1

Pipeline health metrics with KPI cards, quota gauge, pipeline flow viz,
and per-pass breakdown.
"""

import json
from pathlib import Path

import streamlit as st

_UI_CONFIG = json.loads((Path(__file__).parent.parent / "config" / "ui_settings.json").read_text())
_THEME = _UI_CONFIG["theme"]
_STATUS_CFG = _UI_CONFIG["status"]


def _quota_gauge(used: int, total: int) -> str:
    """HTML quota gauge bar."""
    pct = min(100, int(used / total * 100)) if total > 0 else 0
    if pct >= 90:
        color = _THEME["error"]
    elif pct >= 70:
        color = _THEME["warning"]
    else:
        color = _THEME["success"]
    return f"""
    <div style="margin-top:4px;">
      <div style="display:flex;justify-content:space-between;font-size:0.75em;color:#94A3B8;margin-bottom:3px;">
        <span>{used:,} used</span>
        <span>{pct}%</span>
      </div>
      <div style="width:100%;height:8px;background:#0B1120;border-radius:4px;overflow:hidden;border:1px solid #1E293B;">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width 0.3s;"></div>
      </div>
      <div style="font-size:0.7em;color:#64748B;text-align:right;margin-top:2px;">{total:,} daily limit</div>
    </div>
    """


def _pipeline_flow_card(status_counts: dict) -> str:
    """Visual pipeline flow: raw → deduped → locked → classified → scored → reconciled → archived."""
    stages = ["raw", "deduped", "locked", "classified", "scored", "reconciled", "archived"]
    icons = {
        "raw": "📥", "deduped": "🔍", "locked": "🔒",
        "classified": "🧠", "scored": "📊", "reconciled": "✅", "archived": "🗄️",
    }
    nodes = []
    for i, stage in enumerate(stages):
        count = status_counts.get(stage, 0)
        cfg = _STATUS_CFG.get(stage, _STATUS_CFG["raw"])
        active = count > 0
        opacity = "1" if active else "0.35"
        nodes.append(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px;opacity:{opacity};flex:1;min-width:60px;">
          <div style="width:36px;height:36px;border-radius:50%;background:{cfg['bg']};border:2px solid {cfg['color']}30;display:flex;align-items:center;justify-content:center;font-size:1.1em;">
            {icons[stage]}
          </div>
          <span style="font-size:0.65em;color:{cfg['color']};font-weight:600;text-align:center;line-height:1.2;">{cfg['label']}</span>
          <span style="font-size:0.75em;color:#F8FAFC;font-weight:700;">{count:,}</span>
        </div>
        """)
        if i < len(stages) - 1:
            nodes.append(f"""
            <div style="display:flex;align-items:center;opacity:0.5;padding-bottom:18px;">
              <span style="color:#475569;font-size:0.9em;">→</span>
            </div>
            """)

    return f"""
    <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;justify-content:center;padding:12px;background:#0B1120;border:1px solid #1E293B;border-radius:10px;">
      {''.join(nodes)}
    </div>
    """


def render_telemetry(stats: dict, llm_router=None):
    """Render pipeline telemetry dashboard."""

    # ── KPI Row ──
    st.markdown("#### 📊 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    total_quota = stats.get("daily_quota", 1000)
    total_used = stats.get("daily_used", stats.get("llm_calls_24h", 0))
    events_24h = stats.get("events_24h", 0)
    alert_counts = stats.get("alert_counts", {})

    with k1:
        st.metric("🤖 LLM Calls (24h)", f"{stats.get('llm_calls_24h', 0):,}")
        st.markdown(_quota_gauge(total_used, total_quota), unsafe_allow_html=True)
    with k2:
        st.metric("📥 Events Ingested (24h)", f"{events_24h:,}")
        st.metric("🔒 Stale Locks (1h)", stats.get("stale_locks_1h", 0))
    with k3:
        st.metric("🔴 Critical Alerts", alert_counts.get("CRITICAL", 0))
        st.metric("🟠 Alert Tier", alert_counts.get("ALERT", 0))
    with k4:
        st.metric("📊 Tokens Used (24h)", f"{stats.get('tokens_used_24h', 0):,}")
        st.metric("🟡 Watch Tier", alert_counts.get("WATCH", 0))

    st.divider()

    # ── Pipeline Flow ──
    st.markdown("#### 🔄 Pipeline Flow")
    event_counts = stats.get("event_counts", {})
    st.markdown(_pipeline_flow_card(event_counts), unsafe_allow_html=True)

    # ── Last Run ──
    last_run = stats.get("last_run")
    last_run_at = stats.get("last_run_at", "—")
    if last_run:
        st.divider()
        st.markdown("#### ⏱️ Last Pipeline Run")
        r1, r2, r3 = st.columns(3)
        success = last_run.get("success", False)
        with r1:
            st.metric("Status", "✅ Success" if success else "❌ Failed")
        with r2:
            st.metric("Duration", f"{last_run.get('duration_seconds', 0):.1f}s")
        with r3:
            st.metric("Run ID", str(last_run.get("run_id", "—"))[-12:])

        # Pass breakdown in expander
        with st.expander("Pass-level details"):
            passes = ["pass_a", "pass_b", "pass_c", "pass_d", "pass_e", "pass_f"]
            pcols = st.columns(len(passes))
            for i, pname in enumerate(passes):
                data = last_run.get(pname)
                if data and isinstance(data, dict):
                    inserted = data.get("events_inserted", data.get("events_updated", "—"))
                    pcols[i].metric(
                        f"📦 {pname.replace('pass_', 'Pass ').title()}",
                        inserted if isinstance(inserted, int) else "—",
                    )
                else:
                    pcols[i].metric(f"📦 {pname.replace('pass_', 'Pass ').title()}", "—")

            st.caption(f"Run timestamp: {last_run_at}")
            with st.popover("Raw JSON"):
                st.json(last_run)

    # ── Provider Status ──
    if llm_router:
        st.divider()
        st.markdown("#### 🤖 Provider Status")
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
                "Util %": f"{pct:.1f}%",
                "RPM": acct.rpm,
                "Errors": acct.daily_errors,
            })
        st.dataframe(
            status_data,
            use_container_width=True,
            column_config={
                "Util %": st.column_config.TextColumn("Util %", width="small"),
                "Errors": st.column_config.NumberColumn("Errors", width="small"),
            },
        )
