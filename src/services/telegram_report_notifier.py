"""
SIM — Telegram Report Notifier
Blueprint V20.1 §5 / Phase 3

Formats and dispatches weekly intelligence reports and flash updates to Telegram:
- Short HTML summary message.
- Full HTML report uploaded as an attachment (sendDocument).
- Includes the R2 public backup link.
"""

import html
import io
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from src.core.storyline import strip_date_hint
from src.services.telegram_notifier import _post_telegram

logger = logging.getLogger(__name__)


def generate_html_report_payload(
    week_start: str,
    week_end: str,
    top_countries: List[Dict[str, Any]],
    watchlist: List[str],
    emergings: List[str],
    global_assessment: Dict[str, Any]
) -> str:
    """Generates the full styled HTML report representing the weekly bulletin."""
    # Premium glassmorphism dark theme styling matching user dashboard
    styles = """
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
        margin: 0;
        padding: 40px 20px;
        line-height: 1.6;
    }
    .container {
        max-width: 900px;
        margin: 0 auto;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
    }
    h1 {
        font-size: 2.2rem;
        margin-top: 0;
        color: #38bdf8;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h2 {
        color: #f472b6;
        margin-top: 30px;
        font-size: 1.5rem;
    }
    .meta-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .badge-red { background: #ef4444; color: #fff; }
    .badge-orange { background: #f97316; color: #fff; }
    .badge-yellow { background: #eab308; color: #000; }
    .badge-blue { background: #3b82f6; color: #fff; }
    
    .country-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .country-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 12px;
    }
    .country-name {
        font-size: 1.25rem;
        font-weight: bold;
        color: #e2e8f0;
    }
    .metrics {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .spillover-card {
        background: rgba(244, 114, 182, 0.05);
        border: 1px solid rgba(244, 114, 182, 0.2);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .spillover-title {
        font-weight: bold;
        color: #f472b6;
    }
    footer {
        text-align: center;
        margin-top: 40px;
        font-size: 0.8rem;
        color: #64748b;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
    }
    """

    countries_html = ""
    for c in top_countries:
        ti = c.get("ti", 0.0)
        traj = c.get("trajectory", "Stabil")
        
        # Color coding trajectories
        if traj == "Tırmanıyor":
            traj_html = f'<span class="badge badge-red">{traj}</span>'
        elif traj == "Azalıyor":
            traj_html = f'<span class="badge badge-blue">{traj}</span>'
        else:
            traj_html = f'<span class="badge badge-yellow">{traj}</span>'

        # Fetch assessment details
        ass = c.get("assessment") or {}
        summary = ass.get("summary") or "Değerlendirme yapılamadı."
        drivers = "".join(f"<li>{html.escape(d)}</li>" for d in (ass.get("key_drivers") or []))
        forecast = ass.get("forecast") or {}
        risk_dir = forecast.get("risk_direction") or "Belirsiz"
        confidence = forecast.get("confidence") or "Belirsiz"
        most_likely = forecast.get("most_likely_scenario") or ""
        escalation = forecast.get("escalation_scenario") or ""
        
        countries_html += f"""
        <div class="country-card">
            <div class="country-card-header">
                <span class="country-name">📍 {html.escape(c.get('country_iso', ''))} — {html.escape(c.get('country_name', ''))}</span>
                {traj_html}
            </div>
            <div class="metrics">
                <b>Tension Index:</b> {ti:.1f} | <b>Haftalık Fark:</b> {c.get('delta', 0.0):+.1f} | <b>Z-Skor:</b> {c.get('z_score', 0.0):.2f}
            </div>
            <p><b>Özet Değerlendirme:</b> {html.escape(summary)}</p>
            <p><b>Temel Risk Sürücüleri:</b></p>
            <ul>{drivers}</ul>
            <p><b>Gidişat ve Senaryo Analizi:</b></p>
            <ul>
                <li><b>Gelecek Yönü:</b> {html.escape(risk_dir)} (Güven Seviyesi: {html.escape(confidence)})</li>
                <li><b>Olası Senaryo:</b> {html.escape(most_likely)}</li>
                <li><b>Eskalasyon Senaryosu:</b> {html.escape(escalation)}</li>
            </ul>
        </div>
        """

    spillovers_html = ""
    for sp in global_assessment.get("spillovers") or []:
        spillovers_html += f"""
        <div class="spillover-card">
            <div class="spillover-title">🔗 {html.escape(sp.get('spillover_title', ''))}</div>
            <p style="margin: 5px 0;">{html.escape(sp.get('description', ''))}</p>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                <b>İlgili Ülkeler:</b> {", ".join(sp.get('countries_involved', []))} | <b>Risk Etkisi:</b> {html.escape(sp.get('risk_impact', ''))}
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Weekly Strategic Intelligence Report</title>
    <style>{styles}</style>
</head>
<body>
    <div class="container">
        <h1>📊 Haftalık Stratejik Risk Bülteni</h1>
        
        <div class="meta-box">
            <b>Rapor Dönemi:</b> {week_start} / {week_end}<br>
            <b>Yayın Tarihi:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br>
            <b>Global Risk Yönü:</b> <span class="badge badge-orange">{html.escape(global_assessment.get('global_risk_direction', 'Stable'))}</span>
        </div>
        
        <h2>🌍 Küresel Özet & Sürücüler</h2>
        <p>{html.escape(global_assessment.get('executive_summary', ''))}</p>
        
        <h2>📍 Kritik Ülkeler Analizi</h2>
        {countries_html}
        
        <h2>⚠️ İzleme Listeleri</h2>
        <div class="country-card">
            <p><b>İzleme Listesi (Watchlist):</b> {", ".join(watchlist) if watchlist else "Yok"}</p>
            <p><b>Gelişmekte Olan Risk Noktaları (Emerging Concerns):</b> {", ".join(emergings) if emergings else "Yok"}</p>
        </div>
        
        <h2>🔗 Bölgesel Yayılma Etkileri (Spillover)</h2>
        {spillovers_html}
        
        <footer>
            Security Incident Monitor (SIM) — Aviation & Geopolitical Intelligence Platform<br>
            Tüm hakları saklıdır.
        </footer>
    </div>
</body>
</html>
"""
    return html_content


def send_weekly_report_telegram(
    week_start: str,
    week_end: str,
    top_countries: List[Dict[str, Any]],
    watchlist: List[str],
    emergings: List[str],
    global_assessment: Dict[str, Any],
    r2_url: Optional[str] = None,
    scorecard: Optional[str] = None,
) -> Optional[str]:
    """
    Dispatches the weekly report:
    1. Short summary message using sendMessage.
    2. Styled HTML report sent as a file attachment (sendDocument).

    scorecard: optional one-liner grading LAST week's forecast against measured
    outcomes (from the forecast resolver), shown under the global direction.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram weekly report skipped: missing configuration.")
        return None

    # 1. Format and send summary message
    summary_text = (
        f"📊 <b>HAFTALIK STRATEJİK RİSK BÜLTENİ</b>\n"
        f"📅 <b>Dönem:</b> <code>{week_start}</code> / <code>{week_end}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
    )

    if top_countries:
        summary_text += "🔥 <b>Kritik Gelişmeler (Top Countries):</b>\n"
        for c in top_countries[:5]:
            traj_emoji = "🔴" if c.get("trajectory") == "Tırmanıyor" else "🟡"
            summary_text += f"- {traj_emoji} {c.get('country_iso')}: TI={c.get('ti'):.1f} ({c.get('trajectory')})\n"
    
    summary_text += "\n⚠️ <b>İzleme Listeleri:</b>\n"
    summary_text += f"- 🔍 <b>Watchlist:</b> {', '.join(watchlist) if watchlist else 'Yok'}\n"
    summary_text += f"- 💡 <b>Emerging Concerns:</b> {', '.join(emergings) if emergings else 'Yok'}\n"
    
    summary_text += f"\n🌍 <b>Global Yön:</b> <code>{global_assessment.get('global_risk_direction', 'Stable')}</code>\n"

    if scorecard:
        summary_text += f"\n🎯 <b>Geçen Haftanın Karnesi:</b> {scorecard}\n"


    if r2_url:
        summary_text += f"\n🔗 <a href='{r2_url}'>Tarayıcıda Detaylı Raporu Oku (CF R2)</a>\n"
    
    summary_text += "━━━━━━━━━━━━━━━━━━━━━\n"
    summary_text += "ℹ️ <i>Detaylı HTML analiz raporu ekte gönderilmiştir.</i>"

    # Send summary message
    api_url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    telegram_message_id = None
    try:
        resp = _post_telegram(api_url_msg, {
            "chat_id": chat_id,
            "text": summary_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        })
        res_data = resp.json()
        if res_data.get("ok"):
            telegram_message_id = str(res_data["result"]["message_id"])
    except Exception as e:
        logger.error("Failed to send Telegram weekly summary: %s", str(e))

    # 2. Generate HTML report payload & send as document
    try:
        html_report = generate_html_report_payload(
            week_start, week_end, top_countries, watchlist, emergings, global_assessment
        )
        
        file_buffer = io.BytesIO(html_report.encode("utf-8"))
        filename = f"weekly_report_{week_start.replace('-', '')}.html"
        
        api_url_doc = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        
        # We need a separate post call to send multipart file
        # Using Tenacity retry wrapper in _post_telegram but for files we can do it directly:
        resp_doc = httpx.post(
            api_url_doc,
            data={"chat_id": chat_id, "caption": f"Weekly Risk Bulletin ({week_start} / {week_end})"},
            files={"document": (filename, file_buffer, "text/html")},
            timeout=20.0
        )
        resp_doc.raise_for_status()
        logger.info("Weekly HTML report document dispatched successfully.")
    except Exception as e:
        logger.error("Failed to send Telegram weekly HTML document: %s", str(e))

    return telegram_message_id


def send_flash_update_telegram(
    trigger_type: str,
    country_iso: str,
    reason: str,
    events: List[Dict[str, Any]],
    r2_url: Optional[str] = None
) -> Optional[str]:
    """Formats and dispatches a critical Flash Update to Telegram."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram flash update skipped: missing configuration.")
        return None

    # Format flash message
    flash_text = (
        f"🚨 <b>FLAŞ RİSK UYARISI / FLASH UPDATE</b> 🚨\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 <b>Ülke:</b> <code>{country_iso}</code>\n"
        f"⚠️ <b>Tetikleyici:</b> <code>{trigger_type}</code>\n"
        f"📝 <b>Açıklama:</b> {reason}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📰 <b>Tetikleyen Olaylar:</b>\n"
    )

    for ev in events[:3]:
        title = html.escape(
            ev.get("source_title")
            or strip_date_hint(ev.get("storyline_hint") or "")
            or "Unknown Event"
        )
        flash_text += f"- <code>[{ev.get('event_type', 'INCIDENT')}]</code> {title}\n"

    if r2_url:
        flash_text += f"\n🔗 <a href='{r2_url}'>R2 Rapor Bağlantısı</a>"

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = _post_telegram(api_url, {
            "chat_id": chat_id,
            "text": flash_text,
            "parse_mode": "HTML"
        })
        res_data = resp.json()
        if res_data.get("ok"):
            return str(res_data["result"]["message_id"])
    except Exception as e:
        logger.error("Failed to send Flash Update message: %s", str(e))

    return None


def send_sitrep_telegram(
    country_iso: str,
    country_name: str,
    window_start: str,
    window_end: str,
    clusters: List[Dict[str, Any]],
    html_doc: str,
    r2_url: Optional[str] = None,
) -> Optional[str]:
    """
    Dispatches a daily country SITREP:
    1. Short summary card (sendMessage) with label counts.
    2. The styled HTML report as a document attachment (sendDocument) —
       opens directly in the phone's browser.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram SITREP skipped: missing configuration.")
        return None

    label_counts: Dict[str, int] = {}
    for c in clusters:
        lbl = c.get("verification", "?")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    summary_text = (
        f"🗺 <b>GÜNLÜK DURUM RAPORU (SITREP)</b>\n"
        f"🌍 <b>Ülke:</b> {html.escape(country_name)} (<code>{country_iso}</code>)\n"
        f"📅 <b>Dönem:</b> <code>{window_start}</code> — <code>{window_end}</code> UTC\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Olay kümesi:</b> {len(clusters)}\n"
    )
    for lbl, n in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        summary_text += f"- {html.escape(lbl)}: {n}\n"

    if r2_url:
        summary_text += f"\n🔗 <a href='{r2_url}'>Tarayıcıda Oku (CF R2)</a>\n"
    summary_text += "ℹ️ <i>Tam rapor ekte gönderilmiştir.</i>"

    api_url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    telegram_message_id = None
    try:
        resp = _post_telegram(api_url_msg, {
            "chat_id": chat_id,
            "text": summary_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        })
        res_data = resp.json()
        if res_data.get("ok"):
            telegram_message_id = str(res_data["result"]["message_id"])
    except Exception as e:
        logger.error("Failed to send Telegram SITREP summary: %s", str(e))

    try:
        file_buffer = io.BytesIO(html_doc.encode("utf-8"))
        date_tag = window_end[:10].replace("-", "")
        filename = f"sitrep_{country_iso}_{date_tag}.html"

        api_url_doc = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        resp_doc = httpx.post(
            api_url_doc,
            data={"chat_id": chat_id,
                  "caption": f"SITREP {country_name} ({window_start} — {window_end} UTC)"},
            files={"document": (filename, file_buffer, "text/html")},
            timeout=20.0
        )
        resp_doc.raise_for_status()
        logger.info("SITREP document dispatched for %s.", country_iso)
    except Exception as e:
        logger.error("Failed to send Telegram SITREP document: %s", str(e))

    return telegram_message_id


def send_digest_telegram(
    digest: Dict[str, Any],
    window_start: str,
    window_end: str,
    html_doc: str,
) -> Optional[str]:
    """
    Dispatches the daily cross-country digest, sent last so it lands on top of
    the per-country reports in the chat.

    The message body IS the briefing (overview + country risk lines + aviation
    impact) — readable on a phone without opening anything. The HTML document
    follows for the full one-pager.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram digest skipped: missing configuration.")
        return None

    risk_icon = {"Kritik": "🔴", "Yüksek": "🟠", "Yükseltilmiş": "🔵", "Normal": "🟢"}

    text = (
        f"🧭 <b>GÜNLÜK HAP ÖZET</b>\n"
        f"📅 <code>{window_start}</code> — <code>{window_end}</code> UTC\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{html.escape(digest.get('overview', ''))}\n"
    )

    countries = digest.get("countries") or []
    if countries:
        text += "\n<b>🌍 Ülke Durumu</b>\n"
        for c in countries:
            icon = risk_icon.get(c.get("risk", ""), "⚪")
            text += (f"{icon} <b>{html.escape(c.get('name', ''))}</b> "
                     f"({html.escape(c.get('risk', ''))}): "
                     f"{html.escape(c.get('text', ''))}\n")

    aviation = digest.get("aviation") or []
    if aviation:
        text += "\n<b>✈️ Havacılık Etkisi</b>\n"
        for item in aviation:
            text += f"• {html.escape(item)}\n"

    # Telegram hard-caps sendMessage at 4096 chars; the rest is in the document.
    if len(text) > 3900:
        text = text[:3880] + "…\n"
    text += "\nℹ️ <i>Tam özet ekte gönderilmiştir.</i>"

    telegram_message_id = None
    try:
        resp = _post_telegram(f"https://api.telegram.org/bot{bot_token}/sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
        res_data = resp.json()
        if res_data.get("ok"):
            telegram_message_id = str(res_data["result"]["message_id"])
    except Exception as e:
        logger.error("Failed to send Telegram digest summary: %s", str(e))

    try:
        file_buffer = io.BytesIO(html_doc.encode("utf-8"))
        date_tag = window_end[:10].replace("-", "")
        resp_doc = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendDocument",
            data={"chat_id": chat_id,
                  "caption": f"Günlük Hap Özet ({window_start} — {window_end} UTC)"},
            files={"document": (f"ozet_{date_tag}.html", file_buffer, "text/html")},
            timeout=20.0,
        )
        resp_doc.raise_for_status()
        logger.info("Digest document dispatched.")
    except Exception as e:
        logger.error("Failed to send Telegram digest document: %s", str(e))

    return telegram_message_id
