"""
SIM — SITREP HTML Renderer
Turns the LLM's structured Turkish SITREP text into a self-contained,
mobile-first HTML report (inline CSS only — delivered as a Telegram document
and served from R2, so no external assets are allowed).

Parsing contract with the prompt in sitrep_generator:
  - Section headers: short ALL-UPPERCASE lines ("YÖNETİCİ ÖZETİ", "SAHA OLAYLARI", …) —
    the model picks its own section titles, only the casing/length is contractual
  - Location subheaders: short non-bullet lines without ending punctuation
  - Event bullets: "• [tarih] detay — Doğruluk Durumu: <label> — Kaynak: name (url), …"
"""

import html as _html
import re
from typing import Any, Dict, List

from src.core.sitrep_verify import LABEL_MULTI, LABEL_OFFICIAL, LABEL_SINGLE

_BADGE_STYLES = {
    LABEL_OFFICIAL: ("#0b3d23", "#4ade80", "#14532d"),
    LABEL_MULTI: ("#0c2d5e", "#7db3ff", "#1e3a6e"),
    LABEL_SINGLE: ("#4a2c07", "#fbbf24", "#78450f"),
}
_CARD_BORDER = {
    LABEL_OFFICIAL: "#22c55e",
    LABEL_MULTI: "#3b82f6",
    LABEL_SINGLE: "#f59e0b",
}

_SOURCE_RE = re.compile(r"([\w][\w .\-]{1,60}?)\s*\((https?://[^)\s]+)\)")
_URL_RE = re.compile(r"(https?://[^\s<]+)")
_DATE_PREFIX_RE = re.compile(r"^\[?([^\]—]{4,60}?)\]\s*")


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


def _strip_md(line: str) -> str:
    """Drop markdown bold/italic markers the model sometimes adds."""
    return re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", line).strip()


def _badge(label: str) -> str:
    bg, fg, border = _BADGE_STYLES.get(label, ("#333", "#ddd", "#555"))
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:99px;'
        f'font-size:11px;font-weight:600;letter-spacing:.3px;background:{bg};'
        f'color:{fg};border:1px solid {border};white-space:nowrap">{_esc(label)}</span>'
    )


def _source_chips(pairs: List[tuple]) -> str:
    chips = []
    for name, url in pairs[:5]:
        chips.append(
            f'<a href="{_esc(url)}" style="display:inline-block;margin:2px 6px 2px 0;'
            f'padding:2px 10px;border-radius:99px;font-size:11px;color:#93c5fd;'
            f'background:#111c33;border:1px solid #1f3255;text-decoration:none">'
            f'{_esc(name.strip())} ↗</a>'
        )
    return "".join(chips)


def _linkify(escaped_text: str) -> str:
    return _URL_RE.sub(
        lambda m: f'<a href="{m.group(1)}" style="color:#93c5fd">{m.group(1)}</a>',
        escaped_text,
    )


def _render_bullet(line: str) -> str:
    """One event bullet → a card with date chip, body, badge, source chips."""
    body = line.lstrip("•-* ").strip()

    label = None
    for known in (LABEL_OFFICIAL, LABEL_MULTI, LABEL_SINGLE):
        if known in body:
            label = known
            break

    sources = []
    if "Kaynak:" in body:
        body, _, source_part = body.partition("Kaynak:")
        sources = _SOURCE_RE.findall(source_part)
    if "Doğruluk Durumu:" in body:
        body = body.split("Doğruluk Durumu:")[0]
    body = body.strip(" —-–")

    date_chip = ""
    m = _DATE_PREFIX_RE.match(body)
    if m and any(ch.isdigit() for ch in m.group(1)):
        date_chip = (
            f'<span style="color:#8b9cb8;font-size:12px;font-weight:600">'
            f'{_esc(m.group(1).strip())}</span> '
        )
        body = body[m.end():].strip(" —-–")

    border = _CARD_BORDER.get(label, "#334155")
    badge_html = f'<div style="margin-top:8px">{_badge(label)}</div>' if label else ""
    chips_html = (
        f'<div style="margin-top:6px">{_source_chips(sources)}</div>' if sources else ""
    )
    return (
        f'<div style="background:#0f1729;border:1px solid #1e293b;'
        f'border-left:3px solid {border};border-radius:10px;padding:12px 14px;'
        f'margin:10px 0">'
        f'<div style="line-height:1.55">{date_chip}{_linkify(_esc(body))}</div>'
        f"{badge_html}{chips_html}</div>"
    )


def _section_header(title: str) -> str:
    return (
        f'<h2 style="margin:28px 0 4px;font-size:15px;letter-spacing:1.5px;'
        f'text-transform:uppercase;color:#e2e8f0;border-bottom:2px solid #1e3a6e;'
        f'padding-bottom:8px">{_esc(title)}</h2>'
    )


def _stat_card(value: str, caption: str, color: str) -> str:
    return (
        f'<div style="flex:1;min-width:70px;background:#0f1729;border:1px solid #1e293b;'
        f'border-radius:10px;padding:10px 8px;text-align:center">'
        f'<div style="font-size:22px;font-weight:700;color:{color}">{_esc(value)}</div>'
        f'<div style="font-size:10px;letter-spacing:.5px;color:#8b9cb8;'
        f'text-transform:uppercase;margin-top:2px">{_esc(caption)}</div></div>'
    )


def render_sitrep_html(country_name: str, country_iso: str,
                       window_start: str, window_end: str,
                       report_text: str, clusters: List[Dict[str, Any]]) -> str:
    """Self-contained mobile-first HTML for the SITREP."""
    counts = {LABEL_OFFICIAL: 0, LABEL_MULTI: 0, LABEL_SINGLE: 0}
    for c in clusters:
        if c.get("verification") in counts:
            counts[c["verification"]] += 1

    body_parts: List[str] = []
    saw_section = False
    for raw in report_text.splitlines():
        line = _strip_md(raw).lstrip("#").strip()
        if not line:
            continue
        letters = [ch for ch in line if ch.isalpha()]
        is_upper_header = (
            len(line) <= 80 and letters and all(ch == ch.upper() for ch in letters)
        )
        if is_upper_header:
            body_parts.append(_section_header(line))
            saw_section = True
        elif line.startswith(("•", "- ", "* ")):
            body_parts.append(_render_bullet(line))
        elif saw_section and len(line) <= 60 and not line.endswith((".", ":", "!", "?")):
            body_parts.append(
                f'<h3 style="margin:18px 0 2px;font-size:14px;color:#7db3ff">'
                f'📍 {_esc(line)}</h3>'
            )
        else:
            body_parts.append(
                f'<p style="margin:10px 0;line-height:1.65;color:#cbd5e1">'
                f"{_linkify(_esc(line))}</p>"
            )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SITREP — {_esc(country_name)} — {_esc(window_end[:10])}</title>
</head>
<body style="margin:0;background:#0a0f1c;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:15px">
<div style="max-width:720px;margin:0 auto;padding:0 16px 40px">

<div style="background:linear-gradient(135deg,#0c1a3a 0%,#12275c 100%);border:1px solid #1e3a6e;border-radius:0 0 16px 16px;padding:22px 18px 18px;margin:0 -4px">
  <div style="font-size:10px;letter-spacing:3px;color:#7db3ff;text-transform:uppercase">SIM · Security Incident Monitor</div>
  <h1 style="margin:8px 0 2px;font-size:22px;letter-spacing:.5px">GÜNLÜK DURUM RAPORU</h1>
  <div style="font-size:17px;font-weight:600;color:#93c5fd">{_esc(country_name)} <span style="color:#5b6b8a;font-size:13px">({_esc(country_iso)})</span></div>
  <div style="margin-top:10px;font-size:12px;color:#8b9cb8">📅 {_esc(window_start)} — {_esc(window_end)} UTC · 24 saatlik pencere</div>
</div>

<div style="display:flex;gap:8px;margin:14px 0 4px;flex-wrap:wrap">
  {_stat_card(str(len(clusters)), "Olay Kümesi", "#e2e8f0")}
  {_stat_card(str(counts[LABEL_OFFICIAL]), "Resmî", "#4ade80")}
  {_stat_card(str(counts[LABEL_MULTI]), "Çoklu Kaynak", "#7db3ff")}
  {_stat_card(str(counts[LABEL_SINGLE]), "Doğrulanmamış", "#fbbf24")}
</div>

{"".join(body_parts)}

<div style="margin-top:36px;padding-top:14px;border-top:1px solid #1e293b;font-size:11px;color:#5b6b8a;line-height:1.6">
Bu rapor SIM tarafından otomatik üretilmiştir. Doğrulama etiketleri kaynak alan adlarından
kural tabanlı türetilir; "Doğrulanmamış" öğeler tek kaynaklıdır ve teyit gerektirir.
Rapor {_esc(window_end)} UTC itibarıyla mevcut açık kaynaklara dayanır.
</div>

</div>
</body>
</html>"""
