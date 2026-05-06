"""
SIM — Anchor Lookup Component
Blueprint V20.1 §5.1

Search anchor_master database and manually add aliases.
"""

import json

import streamlit as st


def render_anchor_lookup(db_conn):
    """Render anchor lookup with search and alias management."""
    st.subheader("🔍 Anchor Lookup")

    search = st.text_input(
        "Search by IATA, ICAO, or name",
        placeholder="e.g. CAI, HECA, Cairo International",
        key="anchor_search",
    )

    if search and len(search) >= 2:
        # Search across all fields
        results = db_conn.execute(
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
            st.caption(f"Found {len(results)} results")
            for row in results:
                iata, icao, name, country, atype, czib, aliases, lat, lon = row
                czib_badge = " 🔴 CZIB" if czib else ""

                with st.expander(f"✈️ {iata or '—'} / {icao or '—'} — {name} ({country}){czib_badge}"):
                    cols = st.columns(3)
                    cols[0].write(f"**Type:** {atype}")
                    cols[1].write(f"**Location:** {lat:.4f}, {lon:.4f}" if lat and lon else "**Location:** —")
                    cols[2].write(f"**CZIB:** {'Yes 🔴' if czib else 'No'}")

                    # Show aliases
                    alias_list = aliases if isinstance(aliases, list) else json.loads(aliases or "[]")
                    if alias_list:
                        st.write(f"**Aliases:** {', '.join(alias_list)}")
                    else:
                        st.write("**Aliases:** None")

                    # Add alias form
                    new_alias = st.text_input(
                        "Add new alias",
                        key=f"alias_{iata}_{icao}",
                        placeholder="e.g. Cairo Airport",
                    )
                    if st.button("Add Alias", key=f"btn_{iata}_{icao}"):
                        if new_alias and new_alias.strip():
                            try:
                                db_conn.execute(
                                    """UPDATE anchor_master
                                       SET aliases = aliases || %s::jsonb
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

    # Stats
    st.divider()
    st.subheader("📊 Anchor Database Stats")
    stats_cols = st.columns(3)

    total = db_conn.execute("SELECT COUNT(*) FROM anchor_master").fetchone()
    czib_count = db_conn.execute("SELECT COUNT(*) FROM anchor_master WHERE czib_flag = TRUE").fetchone()
    countries = db_conn.execute("SELECT COUNT(DISTINCT country_iso) FROM anchor_master").fetchone()

    stats_cols[0].metric("Total Anchors", total[0] if total else 0)
    stats_cols[1].metric("CZIB Zones", czib_count[0] if czib_count else 0)
    stats_cols[2].metric("Countries", countries[0] if countries else 0)
