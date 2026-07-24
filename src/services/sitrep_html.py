"""
SIM — SITREP HTML Renderer (executive briefing template)
Turns the LLM's structured Turkish SITREP text into a self-contained,
mobile-first HTML report (inline CSS only — delivered as a Telegram document
and served from R2, so no external assets and no JavaScript are allowed;
progressive disclosure uses native <details>).

Reading order is summary-first: KPI tiles → top incidents at a glance →
executive summary callout → full narrative → collapsible raw daily log.

Parsing contract with the prompt in sitrep_generator:
  - Section headers: short ALL-UPPERCASE lines ("YÖNETİCİ ÖZETİ", "SAHA OLAYLARI", …) —
    the model picks its own section titles, only the casing/length is contractual
  - Location subheaders: short non-bullet lines without ending punctuation
  - Event bullets: "• [tarih] detay — Doğruluk Durumu: <label> — Kaynak: name (url), …"
"""

import html as _html
import re
from typing import Any, Dict, List, Optional

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

# Turkish display labels for event_type_catalog codes shown to the reader
# (highlights + raw log). Unknown codes fall back to a de-snaked form.
_EVENT_TYPE_TR = {
    "missile_strike": "Füze Saldırısı",
    "military_action": "Askeri Harekât",
    "war_escalation": "Savaş Tırmanması",
    "geopolitical_conflict": "Jeopolitik Çatışma",
    "ceasefire_violation": "Ateşkes İhlali",
    "drone_attack_critical_infra": "Kritik Altyapıya İHA Saldırısı",
    "drone_airport_attack": "Havalimanına İHA Saldırısı",
    "drone_port_attack": "Limana İHA Saldırısı",
    "drone_energy_attack": "Enerji Tesisine İHA Saldırısı",
    "drone_military_base_attack": "Askeri Üsse İHA Saldırısı",
    "drone_incursion": "İHA İhlali",
    "terrorism": "Terör Saldırısı",
    "jihadist_attack": "Cihatçı Saldırı",
    "insurgency_attack": "İsyancı Saldırısı",
    "suicide_bombing": "İntihar Saldırısı",
    "mass_casualty_event": "Toplu Can Kaybı",
    "civilian_casualties": "Sivil Can Kaybı",
    "mass_shooting": "Silahlı Saldırı",
    "active_shooter": "Aktif Saldırgan",
    "civil_unrest": "Sivil Kargaşa",
    "protest": "Protesto",
    "mass_demonstration": "Kitlesel Gösteri",
    "riot": "Ayaklanma",
    "general_strike": "Genel Grev",
    "coup_attempt": "Darbe Girişimi",
    "political_event": "Siyasi Gelişme",
    "travel_advisory": "Seyahat Uyarısı",
    "travel_ban": "Seyahat Yasağı",
    "embassy_closure": "Elçilik Kapatma",
    "evacuation": "Tahliye",
    "humanitarian_crisis": "İnsani Kriz",
    "security_incident": "Güvenlik Olayı",
    "hijacking": "Uçak Kaçırma",
    "bomb_threat": "Bomba Tehdidi",
    "web_discovery": "Web Taraması Bulgusu",
    "other_aviation_related": "Diğer Havacılık Olayı",
    "unclassified": "Sınıflandırılmamış",
}

_SOURCE_RE = re.compile(r"([\w][\w .\-]{1,60}?)\s*\((https?://[^)\s]+)\)")
_URL_RE = re.compile(r"(https?://[^\s<]+)")
_DATE_PREFIX_RE = re.compile(r"^\[?([^\]—]{4,60}?)\]\s*")

_EXEC_HEADER = "YÖNETİCİ ÖZETİ"


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


def _strip_md(line: str) -> str:
    """Drop markdown bold/italic markers the model sometimes adds."""
    return re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", line).strip()


def _event_type_label(code: str) -> str:
    if not code:
        return ""
    return _EVENT_TYPE_TR.get(code, code.replace("_", " ").title())


def _badge(label: str) -> str:
    bg, fg, border = _BADGE_STYLES.get(label, ("#333", "#ddd", "#555"))
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:99px;'
        f'font-size:11px;font-weight:600;letter-spacing:.3px;background:{bg};'
        f'color:{fg};border:1px solid {border};white-space:nowrap">{_esc(label)}</span>'
    )


def _severity_meter(severity: int) -> str:
    """Thin single-hue magnitude bar with the numeric value as text (never
    color-alone). severity is 0-100."""
    sev = max(0, min(int(severity or 0), 100))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;vertical-align:middle">'
        f'<span style="display:inline-block;width:72px;height:6px;border-radius:99px;'
        f'background:#1e293b;overflow:hidden">'
        f'<span style="display:block;width:{sev}%;height:100%;border-radius:99px;'
        f'background:#fbbf24"></span></span>'
        f'<span style="font-size:11px;font-weight:700;color:#cbd5e1">{sev}</span>'
        f'<span style="font-size:10px;color:#5b6b8a">/100</span></span>'
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


def _section_header(title: str, icon: str = "") -> str:
    prefix = f"{icon} " if icon else ""
    return (
        f'<h2 style="margin:30px 0 4px;font-size:15px;letter-spacing:1.5px;'
        f'text-transform:uppercase;color:#e2e8f0;border-bottom:2px solid #1e3a6e;'
        f'padding-bottom:8px">{prefix}{_esc(title)}</h2>'
    )


def _stat_card(value: str, caption: str, color: str) -> str:
    return (
        f'<div style="flex:1;min-width:64px;background:#0f1729;border:1px solid #1e293b;'
        f'border-radius:10px;padding:10px 8px;text-align:center">'
        f'<div style="font-size:22px;font-weight:700;color:{color}">{_esc(value)}</div>'
        f'<div style="font-size:10px;letter-spacing:.5px;color:#8b9cb8;'
        f'text-transform:uppercase;margin-top:2px">{_esc(caption)}</div></div>'
    )


def _exec_summary_card(paragraphs: List[str]) -> str:
    """The executive summary as a visually dominant callout — the one block a
    hurried reader is meant to absorb."""
    body = "".join(
        f'<p style="margin:10px 0 0;line-height:1.7;font-size:15.5px;color:#e8eef7">'
        f"{_linkify(_esc(p))}</p>"
        for p in paragraphs
    )
    return (
        f'<div style="background:linear-gradient(135deg,#101b36 0%,#0e1a30 100%);'
        f'border:1px solid #27406e;border-left:4px solid #7db3ff;border-radius:12px;'
        f'padding:16px 18px;margin:12px 0 4px">'
        f'<div style="font-size:11px;letter-spacing:2px;color:#7db3ff;'
        f'text-transform:uppercase;font-weight:700">📌 {_esc(_EXEC_HEADER)}</div>'
        f"{body}</div>"
    )


def _highlights(clusters: List[Dict[str, Any]], top_n: int = 3) -> str:
    """Deterministic 'at a glance' strip: the day's highest-severity clusters,
    so a manager gets the critical picture before any prose."""
    ranked = sorted(
        (c for c in clusters if (c.get("severity") or 0) > 0),
        key=lambda c: -(c.get("severity") or 0),
    )[:top_n]
    if not ranked:
        return ""
    rows = []
    for i, c in enumerate(ranked, 1):
        label = c.get("verification")
        badge = _badge(label) if label else ""
        rows.append(
            f'<div style="display:flex;gap:12px;align-items:flex-start;'
            f'background:#0f1729;border:1px solid #1e293b;border-radius:10px;'
            f'padding:12px 14px;margin:8px 0">'
            f'<div style="min-width:26px;height:26px;border-radius:8px;background:#12275c;'
            f'color:#93c5fd;font-weight:700;font-size:13px;display:flex;'
            f'align-items:center;justify-content:center">{i}</div>'
            f'<div style="flex:1">'
            f'<div style="font-weight:600;font-size:14px;color:#e2e8f0">'
            f'{_esc(c.get("location") or "—")}'
            f' <span style="font-weight:400;color:#8b9cb8;font-size:12px">· '
            f'{_esc(_event_type_label(c.get("event_type") or ""))}</span></div>'
            f'<div style="font-size:11px;color:#5b6b8a;margin:2px 0 6px">'
            f'{_esc(str(c.get("date") or ""))}</div>'
            f'<div>{_severity_meter(c.get("severity") or 0)}'
            f'<span style="margin-left:10px">{badge}</span></div>'
            f"</div></div>"
        )
    return (
        _section_header("GÜNÜN ÖNE ÇIKANLARI", "⚡")
        + '<div style="font-size:11px;color:#5b6b8a;margin:6px 0 2px">'
        "Şiddet puanına göre günün en kritik olay kümeleri.</div>"
        + "".join(rows)
    )


def _appendix_row(cluster: Dict[str, Any]) -> str:
    """One raw cluster record for the full daily log."""
    sources = [
        (s.get("name") or "kaynak", s["url"])
        for s in cluster.get("sources", []) if s.get("url")
    ]
    chips = f'<div style="margin-top:4px">{_source_chips(sources)}</div>' if sources else ""
    label = cluster.get("verification")
    badge = f'<span style="margin-left:6px">{_badge(label)}</span>' if label else ""
    snippet = (cluster.get("snippet") or "")[:220]
    meta = " · ".join(
        str(v) for v in (cluster.get("date"),
                         _event_type_label(cluster.get("event_type") or "")) if v
    )
    sev = cluster.get("severity") or 0
    meter = f'<div style="margin-top:4px">{_severity_meter(sev)}</div>' if sev else ""
    return (
        f'<div style="border-bottom:1px solid #1e293b;padding:10px 2px">'
        f'<div style="font-size:13px;line-height:1.5">'
        f'<span style="font-weight:600;color:#e2e8f0">{_esc(cluster.get("location") or "—")}</span>'
        f'{badge}</div>'
        f'<div style="font-size:11px;color:#5b6b8a;margin-top:2px">{_esc(meta)}</div>'
        f'<div style="margin-top:4px;font-size:12px;color:#8b9cb8;line-height:1.5">{_esc(snippet)}</div>'
        f"{meter}{chips}</div>"
    )


def _appendix(clusters: List[Dict[str, Any]]) -> str:
    """
    Deterministic full-day log inside a collapsed <details>: every cluster the
    pipeline produced, rule-based with verification label, severity and source
    links — the report stays a complete daily country record even when the LLM
    narrative condenses, and executives only expand it on demand.
    """
    if not clusters:
        return ""
    rows = "".join(_appendix_row(c) for c in clusters)
    return (
        f'<details style="margin-top:30px;background:#0d1526;border:1px solid #1e293b;'
        f'border-radius:12px;padding:0 14px">'
        f'<summary style="cursor:pointer;list-style:none;padding:14px 0;'
        f'font-size:13px;font-weight:700;letter-spacing:1.2px;color:#e2e8f0;'
        f'text-transform:uppercase">📋 GÜNLÜK OLAY KÜNYESİ — TÜM KAYITLAR '
        f'<span style="color:#7db3ff">({len(clusters)})</span>'
        f'<span style="float:right;color:#5b6b8a;font-weight:400;font-size:11px;'
        f'text-transform:none">aç / kapat</span></summary>'
        f'<div style="font-size:11px;color:#5b6b8a;margin:0 0 6px;line-height:1.5">'
        "Pencere içindeki tüm olay kümelerinin ham kaydı (orijinal dilinde) — "
        "anlatı bölümünde ayrıntılandırılmayan olaylar dahil.</div>"
        f"{rows}"
        f'<div style="height:10px"></div></details>'
    )


def _aviation_section(clusters: List[Dict[str, Any]], top_n: int = 8) -> str:
    """Deterministic regional-aviation block: flight-disruption events relevant
    to this country but attributed to the region or a neighbour (null / other
    country_iso), which the per-country flow never surfaces. Aviation is the
    priority domain, so this renders open near the top — not buried in a
    collapsed appendix and not left to the LLM narrative to remember."""
    if not clusters:
        return ""
    ranked = sorted(clusters, key=lambda c: -(c.get("severity") or 0))[:top_n]
    rows = "".join(_appendix_row(c) for c in ranked)
    return (
        _section_header("BÖLGESEL HAVACILIK KESİNTİLERİ", "✈")
        + '<div style="font-size:11px;color:#5b6b8a;margin:6px 0 2px">'
        "Bu ülkeyi ilgilendiren, ancak bölgeye veya komşu ülkelere atfedilen "
        "uçuş kesintileri (uçuş durdurma, havalimanı/hava sahası kapanışı).</div>"
        + rows
    )


def render_sitrep_html(country_name: str, country_iso: str,
                       window_start: str, window_end: str,
                       report_text: str, clusters: List[Dict[str, Any]],
                       aviation_clusters: Optional[List[Dict[str, Any]]] = None) -> str:
    """Self-contained mobile-first HTML for the SITREP."""
    counts = {LABEL_OFFICIAL: 0, LABEL_MULTI: 0, LABEL_SINGLE: 0}
    for c in clusters:
        if c.get("verification") in counts:
            counts[c["verification"]] += 1
    max_sev = max((c.get("severity") or 0 for c in clusters), default=0)

    body_parts: List[str] = []
    exec_paragraphs: List[str] = []
    in_exec = False
    saw_section = False

    def _flush_exec() -> None:
        nonlocal in_exec
        if exec_paragraphs:
            body_parts.append(_exec_summary_card(exec_paragraphs))
            exec_paragraphs.clear()
        in_exec = False

    for raw in report_text.splitlines():
        line = _strip_md(raw).lstrip("#").strip()
        if not line or not any(ch.isalnum() for ch in line):
            continue  # empty or decorative separator ("---", "***") — never content
        letters = [ch for ch in line if ch.isalpha()]
        is_upper_header = (
            len(line) <= 80 and letters and all(ch == ch.upper() for ch in letters)
        )
        if is_upper_header:
            saw_section = True
            if _EXEC_HEADER in line:
                # header itself is drawn by the callout card
                in_exec = True
                continue
            _flush_exec()
            body_parts.append(_section_header(line))
        elif line.startswith(("•", "- ", "* ")):
            _flush_exec()
            body_parts.append(_render_bullet(line))
        elif in_exec:
            exec_paragraphs.append(line)
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
    _flush_exec()

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
  {_stat_card(str(max_sev), "Maks. Şiddet", "#fbbf24") if max_sev else ""}
</div>

{_highlights(clusters)}

{_aviation_section(aviation_clusters or [])}

{"".join(body_parts)}

{_appendix(clusters)}

<div style="margin-top:36px;padding-top:14px;border-top:1px solid #1e293b;font-size:11px;color:#5b6b8a;line-height:1.6">
Bu rapor SIM tarafından otomatik üretilmiştir. Doğrulama etiketleri kaynak alan adlarından
kural tabanlı türetilir; "Doğrulanmamış" öğeler tek kaynaklıdır ve teyit gerektirir.
Rapor {_esc(window_end)} UTC itibarıyla mevcut açık kaynaklara dayanır.
</div>

</div>
</body>
</html>"""
