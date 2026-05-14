"""
SIM — Anchor Lookup Component
Blueprint V20.1 §5.1

Search anchor_master database and manually add aliases.
Modern card-based layout with stats and search UX.
"""

import html
import json

import streamlit as st

from streamlit_app.services.cache import _safe_execute


def _country_flag(iso: str | None) -> str:
    if not isinstance(iso, str) or len(iso) != 2:
        return "🌐"
    return chr(0x1F1E6 + ord(iso[0].upper()) - 65) + chr(0x1F1E6 + ord(iso[1].upper()) - 65)


def render_anchor_lookup(db_conn):
    """Render anchor lookup with search and alias management."""
    st.subheader("🔍 Anchor Lookup")

    # Search input with icon hint
    search = st.text_input(
        "Search by IATA, ICAO, or name",
        placeholder="e.g. CAI, HECA, Cairo International",
        key="anchor_search",
    )

    if search and len(search) >= 2:
        results = _safe_execute(
            db_conn,
            """SELECT iata_code, icao_code, canonical_name, country_iso,
                      anchor_type, czib_flag, aliases, latitude, longitude
               FROM anchor_master
               WHERE iata_code ILIKE %s
                  OR icao_code ILIKE %s
                  OR canonical_name ILIKE %s
                  OR aliases::text ILIKE %s
               ORDER BY canonical_name
               LIMIT 20""",
            (f"%{search}%", f"%{search}%", f"%{search}%", f"%{search}%"),
        ).fetchall()

        if results:
            st.caption(f"Found {len(results)} result{'s' if len(results) > 1 else ''}")
            for row in results:
                iata, icao, name, country, atype, czib, aliases, lat, lon = row
                flag = _country_flag(country)
                czib_badge = "🔴 CZIB" if czib else ""

                with st.expander(
                    f"✈️ {iata or '—'} / {icao or '—'} — {name} ({flag} {country}){czib_badge}",
                    expanded=False,
                ):
                    # Info row
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Type", atype or "—")
                    c2.metric(
                        "Location",
                        f"{lat:.4f}, {lon:.4f}" if lat and lon else "—",
                    )
                    c3.metric("CZIB", "Yes 🔴" if czib else "No")

                    # Aliases
                    alias_list = aliases if isinstance(aliases, list) else json.loads(aliases or "[]")
                    if alias_list:
                        tags = " ".join(
                            f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(99,102,241,0.15);color:#6366F1;font-size:0.75em;margin-right:4px;margin-bottom:4px;'>{html.escape(a)}</span>"
                            for a in alias_list
                        )
                        st.markdown(f"**Aliases:** {tags}", unsafe_allow_html=True)
                    else:
                        st.write("**Aliases:** None")

                    # Add alias form
                    st.divider()
                    new_alias = st.text_input(
                        "Add new alias",
                        key=f"alias_{iata}_{icao}",
                        placeholder="e.g. Cairo Airport",
                    )
                    if st.button("➕ Add Alias", key=f"btn_{iata}_{icao}"):
                        if new_alias and new_alias.strip():
                            try:
                                db_conn.execute(
                                    """UPDATE anchor_master
                                       SET aliases = COALESCE(aliases, '[]'::jsonb) || %s::jsonb
                                       WHERE iata_code = %s""",
                                    (json.dumps([new_alias.strip()]), iata),
                                )
                                db_conn.commit()
                                st.success(f"Added alias '{new_alias.strip()}' to {iata}")
                                st.rerun()
                            except Exception as e:
                                db_conn.rollback()
                                st.error(f"Error: {e}")
        else:
            st.warning(f"No results for '{search}'")

    # ── Stats Dashboard ──
    st.divider()
    st.subheader("📊 Anchor Database")

    total = _safe_execute(db_conn, "SELECT COUNT(*) FROM anchor_master").fetchone()
    czib_count = _safe_execute(db_conn, "SELECT COUNT(*) FROM anchor_master WHERE czib_flag = TRUE").fetchone()
    countries = _safe_execute(db_conn, "SELECT COUNT(DISTINCT country_iso) FROM anchor_master").fetchone()
    airports = _safe_execute(db_conn, "SELECT COUNT(*) FROM anchor_master WHERE anchor_type = 'airport'").fetchone()
    hotels = _safe_execute(db_conn, "SELECT COUNT(*) FROM anchor_master WHERE anchor_type = 'hotel_chain'").fetchone()

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Anchors", total[0] if total else 0)
    s2.metric("Airports", airports[0] if airports else 0)
    s3.metric("Hotels", hotels[0] if hotels else 0)
    s4.metric("CZIB Zones", czib_count[0] if czib_count else 0)
    s5.metric("Countries", countries[0] if countries else 0)

    # Recent CZIB anchors table
    czib_rows = _safe_execute(
        db_conn,
        """SELECT iata_code, icao_code, canonical_name, country_iso
           FROM anchor_master
           WHERE czib_flag = TRUE
           ORDER BY canonical_name
           LIMIT 20"""
    ).fetchall()

    if czib_rows:
        st.markdown("#### 🔴 CZIB Zones")
        czib_data = [
            {
                "IATA": r[0],
                "ICAO": r[1],
                "Name": r[2],
                "Country": f"{_country_flag(r[3])} {r[3]}",
            }
            for r in czib_rows
        ]
        st.dataframe(czib_data, width="stretch", hide_index=True)
