"""
SIM — Daily SITREP Digest HTML renderer

Deliberately quieter than the per-country report: no verification badges, no
source chips, no raw event log. Those live in the country SITREPs, which ship
alongside. This page is the one-screen read — situation, per-country risk,
aviation impact, what to watch.

Self-contained inline CSS only (delivered as a Telegram document and served from
R2, so no external assets and no JavaScript).
"""

import html as _html
from typing import Any, Dict, List

from src.services.sitrep_digest import (
    RISK_CRITICAL,
    RISK_ELEVATED,
    RISK_HIGH,
    RISK_NORMAL,
)

# (text colour, background, border) per risk level
_RISK_STYLES = {
    RISK_CRITICAL: ("#fca5a5", "#3f1414", "#7f1d1d"),
    RISK_HIGH: ("#fbbf24", "#3d2607", "#78450f"),
    RISK_ELEVATED: ("#7db3ff", "#0c2d5e", "#1e3a6e"),
    RISK_NORMAL: ("#4ade80", "#0b3d23", "#14532d"),
}


def _esc(text: Any) -> str:
    return _html.escape(str(text or ""))


def _risk_badge(risk: str) -> str:
    fg, bg, border = _RISK_STYLES.get(risk, _RISK_STYLES[RISK_NORMAL])
    return (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:20px;'
        f'background:{bg};color:{fg};border:1px solid {border};font-size:10px;'
        f'font-weight:700;letter-spacing:.6px;text-transform:uppercase;'
        f'white-space:nowrap">{_esc(risk)}</span>'
    )


def _country_row(country: Dict[str, Any]) -> str:
    return (
        f'<div style="border-bottom:1px solid #1e293b;padding:11px 2px">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'<span style="font-weight:700;color:#e2e8f0;font-size:14px">'
        f'{_esc(country.get("name"))}</span>'
        f'<span style="color:#5b6b8a;font-size:11px">({_esc(country.get("iso"))})</span>'
        f'{_risk_badge(country.get("risk", RISK_NORMAL))}</div>'
        f'<div style="margin-top:5px;font-size:13px;line-height:1.6;color:#cbd5e1">'
        f'{_esc(country.get("text"))}</div>'
        f"</div>"
    )


def _section(title: str, icon: str, items: List[str], accent: str = "#7db3ff",
             highlight: bool = False) -> str:
    """Bulleted section; renders nothing when the digest had no items for it."""
    if not items:
        return ""
    bullets = "".join(
        f'<li style="margin:7px 0;line-height:1.6;color:#cbd5e1">{_esc(i)}</li>'
        for i in items
    )
    frame = (
        "background:#131f38;border:1px solid #1e3a6e;border-radius:12px;padding:14px 16px"
        if highlight else
        "padding:0 2px"
    )
    return (
        f'<div style="margin-top:22px;{frame}">'
        f'<div style="font-size:12px;font-weight:700;letter-spacing:1.4px;'
        f'color:{accent};text-transform:uppercase">{icon} {_esc(title)}</div>'
        f'<ul style="margin:8px 0 0;padding-left:20px;font-size:13px">{bullets}</ul>'
        f"</div>"
    )


def render_digest_html(digest: Dict[str, Any], window_start: str,
                       window_end: str) -> str:
    """Self-contained mobile-first HTML for the daily cross-country digest."""
    countries = digest.get("countries") or []
    country_rows = "".join(_country_row(c) for c in countries)
    overview = _esc(digest.get("overview"))

    country_block = (
        f'<div style="margin-top:22px">'
        f'<div style="font-size:12px;font-weight:700;letter-spacing:1.4px;'
        f'color:#7db3ff;text-transform:uppercase">🌍 ÜLKE DURUMU</div>'
        f'<div style="margin-top:6px">{country_rows}</div></div>'
        if country_rows else ""
    )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Günlük Hap Özet — {_esc(window_end[:10])}</title>
</head>
<body style="margin:0;background:#0a0f1c;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:15px">
<div style="max-width:720px;margin:0 auto;padding:0 16px 40px">

<div style="background:linear-gradient(135deg,#0c1a3a 0%,#12275c 100%);border:1px solid #1e3a6e;border-radius:0 0 16px 16px;padding:22px 18px 18px;margin:0 -4px">
  <div style="font-size:10px;letter-spacing:3px;color:#7db3ff;text-transform:uppercase">SIM · Security Incident Monitor</div>
  <h1 style="margin:8px 0 2px;font-size:22px;letter-spacing:.5px">GÜNLÜK HAP ÖZET</h1>
  <div style="font-size:13px;color:#93c5fd">{len(countries)} ülke · tek sayfa yönetici brifingi</div>
  <div style="margin-top:10px;font-size:12px;color:#8b9cb8">📅 {_esc(window_start)} — {_esc(window_end)} UTC · 24 saatlik pencere</div>
</div>

<div style="margin-top:16px;background:#101b33;border-left:3px solid #3b82f6;border-radius:0 12px 12px 0;padding:14px 16px">
  <div style="font-size:12px;font-weight:700;letter-spacing:1.4px;color:#7db3ff;text-transform:uppercase">📌 GÜNÜN TABLOSU</div>
  <p style="margin:8px 0 0;line-height:1.7;color:#e2e8f0;font-size:14px">{overview}</p>
</div>

{country_block}

{_section("HAVACILIK ETKİSİ", "✈️", digest.get("aviation") or [], accent="#93c5fd", highlight=True)}

{_section("ÖNE ÇIKANLAR", "⚡", digest.get("highlights") or [])}

{_section("İZLEME · 24-72 SAAT", "🔭", digest.get("watch") or [], accent="#8b9cb8")}

<div style="margin-top:36px;padding-top:14px;border-top:1px solid #1e293b;font-size:11px;color:#5b6b8a;line-height:1.6">
Bu özet, aynı döneme ait ülke bazlı SITREP raporlarından SIM tarafından otomatik
sentezlenmiştir. Kaynak atıfları, doğrulama etiketleri ve tam olay künyesi ilgili
ülke raporlarında yer alır. {_esc(window_end)} UTC itibarıyla mevcut açık kaynaklara dayanır.
</div>

</div>
</body>
</html>"""
