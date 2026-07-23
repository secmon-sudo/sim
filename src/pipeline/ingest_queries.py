"""
SIM — Pass A ingest: search-query construction

Builds the per-run Google News RSS query set (static tiers + dynamic
active-storyline queries).
Split out of pass_a_ingest.py on 2026-07-16.
"""

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Max dynamic (active-storyline) search queries per run. Dynamic queries run first and
# eat into MAX_QUERIES_PER_RUN (50); capping them leaves room for static-tier discovery
# so tracking existing storylines never fully starves finding NEW ones.
MAX_DYNAMIC_QUERIES = 15

# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_search_queries(db_conn=None) -> list[dict]:
    """Build focused search queries.  ONLY aviation / hotel / tourism / mass-casualty security.

    Strategy:
    1.  Keep the query list SHORT (≈40 queries).  300+ low-quality queries drown the signal.
    2.  Every query MUST contain aviation / hotel / tourism context, OR be an
        unambiguous mass-casualty security phrase.
    3.  Google News RSS handles simple `phrase airport` syntax reliably.
        Complex boolean with many negative keywords breaks or returns 0 results.
    4.  Remaining noise is caught by `is_noise()` post-filter.
    """
    active_queries = []
    if db_conn is not None:
        try:
            # Dynamic queries track *developing* storylines. Discipline (each rule keeps
            # noise from permanently occupying a search slot and the LLM quota downstream):
            #   - HAVING COUNT(*) >= 2: a storyline backed by a single article never earns
            #     a recurring search. Singletons are usually one-off/low-quality and were
            #     the main way junk storylines self-perpetuated (search → same article →
            #     re-score → search again).
            #   - The window is measured from the LAST event time, so a storyline that
            #     stops producing new events ages out of its window automatically (early
            #     retirement — no cron needed).
            with db_conn.transaction():
                rows = db_conn.execute(
                    """SELECT storyline_hint,
                              MAX(occurred_at_est) AS last_update,
                              MAX(severity_score)  AS max_severity,
                              COUNT(*)             AS event_count
                       FROM events
                       WHERE status IN ('scored', 'reconciled')
                         AND storyline_hint IS NOT NULL
                         AND occurred_at_est > NOW() - INTERVAL '14 days'
                       GROUP BY storyline_id, storyline_hint
                       HAVING COUNT(*) >= 2"""
                ).fetchall()

            now = datetime.now(timezone.utc)
            candidates = []  # (clean_query, max_severity, last_update)
            for row in rows:
                hint = row[0]
                last_update = row[1]
                max_severity = row[2] or 0

                # Ensure last_update is timezone-aware for math comparison
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=timezone.utc)

                age_hours = (now - last_update).total_seconds() / 3600

                # Determine tracking window based on severity
                if max_severity >= 80:
                    window = 168  # 7 days
                elif max_severity >= 60:
                    window = 72   # 3 days
                else:
                    window = 36   # 36 hours

                if age_hours <= window:
                    # Clean the hint (strip the date hint from the end, e.g. " Jun9")
                    clean_query = re.sub(r'\s+[A-Z][a-z]{2}\d{1,2}$', '', hint)
                    if clean_query:
                        candidates.append((clean_query, max_severity, last_update))

            # Cap dynamic queries so they can't crowd out static-tier discovery in the
            # per-run MAX_QUERIES_PER_RUN budget. Priority: higher severity first, then
            # most recently active.
            candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
            active_queries = [c[0] for c in candidates[:MAX_DYNAMIC_QUERIES]]
        except Exception:
            logger.exception("Error building dynamic search queries from active storylines")

    queries = []
    seen = set()

    def _add(q: str, dynamic: bool = False):
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append({"query": q, "broad": False, "dynamic": dynamic})

    # Add active dynamic queries FIRST so they are run with highest priority
    for q in active_queries:
        _add(q, dynamic=True)

    # ── Tier 1: Unambiguous aviation / airport security phrases ──
    tier1 = [
        '"airport attack"',
        '"airport shooting"',
        '"airport bombing"',
        '"airport stabbing"',
        '"airport security breach"',
        '"airport threat"',
        '"airport explosion"',
        '"airport terror"',
        '"airport gunfire"',
        '"airline crew attack"',
        '"airline staff assault"',
        '"flight attendant attack"',
        '"pilot assault"',
        '"pilot attacked"',
        '"ground crew attack"',
        '"ground staff stabbed"',
        '"cabin crew assaulted"',
        '"airport worker killed"',
        '"check-in agent attacked"',
        '"air traffic controller threat"',
    ]
    for q in tier1:
        _add(q)

    # ── Tier 2: Hotel / resort / tourism security phrases ──
    tier2 = [
        '"hotel attack"',
        '"hotel bombing"',
        '"hotel shooting"',
        '"hotel siege"',
        '"hotel explosion"',
        '"hotel terror"',
        '"resort attack"',
        '"resort bombing"',
        '"beach attack"',
        '"cruise ship attack"',
        '"tourist hotel attack"',
        '"hostage hotel"',
    ]
    for q in tier2:
        _add(q)

    # ── Tier 3: Generic words FORCED into aviation/hotel context ──
    # Simple `phrase airport` format — Google News handles this reliably.
    # Noise filters catch any remaining false positives post-fetch.
    tier3 = [
        '"bomb threat" airport',
        '"active shooter" airport',
        '"security breach" airport',
        '"evacuation" airport',
        '"explosion" airport',
        '"suspicious package" airport',
        '"hostage" airport',
        '"hijack" airport',
        '"mass shooting" airport',
        '"mass casualty" airport',
        '"suicide bombing" airport',
        '"drone attack" airport',
        '"unruly passenger" airport',
        '"passenger attack crew" airport',
        '"laser attack" airport',
        '"runway incursion"',
        '"emergency landing"',
        '"engine failure" flight',
        '"fire on board" flight',
        '"bird strike" airport',
        '"depressurization" flight',
        '"drone incursion" airport',
    ]
    for q in tier3:
        _add(q)

    # ── Tier 4: Geopolitical / African terrorism (narrowed to infrastructure-relevant) ──
    geo = [
        '"missile strike" airport OR airspace',
        '"airstrike" civilian airport OR infrastructure',
        '"war escalation" airspace OR NOTAM',
        '"Iran Israel" military strike',
        '"Ukraine Russia" drone attack infrastructure',
        '"Boko Haram" attack',
        '"Al-Shabaab" attack',
        '"jihadist attack"',
        '"ISIS Africa" attack',
        '"Sahel crisis" attack',
        '"civilian casualties" airstrike',
        '"Houthi attack" ship OR Red Sea',
        '"drone swarm" attack',
        '"India Pakistan" military',
    ]
    for q in geo:
        _add(q)

    # ── Tier 5: Maritime / rail / mass-transit security ──
    transport = [
        '"train station attack"',
        '"train station bombing"',
        '"metro attack"',
        '"subway attack"',
        '"bus terminal attack"',
        '"port attack"',
        '"piracy attack"',
        '"tanker hijack"',
        '"ship hijacked"',
        '"maritime security incident"',
    ]
    for q in transport:
        _add(q)

    # ── Tier 6: Tourist-specific threats ──
    tourist = [
        '"tourists killed"',
        '"tourist attack"',
        '"tourist kidnapped"',
        '"travelers warned"',
        '"travel advisory" raised',
        '"travel warning" issued',
        '"do not travel" warning',
        '"embassy attack"',
        '"consulate attack"',
        '"tourist area bombing"',
    ]
    for q in tourist:
        _add(q)

    # ── Tier 7: Protest & Civil Unrest ──
    protest = [
        '"mass protest"',
        '"violent protest"',
        '"anti-government protest"',
        '"protest crackdown"',
        '"riot police" protest',
        '"tear gas" protest',
        '"general strike"',
        '"nationwide strike"',
        '"demonstration violence"',
        '"protesters killed"',
        '"protesters shot"',
        '"protest shooting"',
        '"coup attempt"',
        '"political unrest"',
        '"state of emergency" protest',
        '"curfew imposed"',
        '"martial law"',
        '"protest" clashes',
        '"uprising"',
        '"civil unrest"',
    ]
    for q in protest:
        _add(q)

    # ── Tier 8: Travel Advisory / Travel Warning ──
    advisory = [
        '"travel advisory" country',
        '"travel warning" country',
        '"do not travel" advisory',
        '"travel ban"',
        '"embassy closed"',
        '"consulate closed"',
        '"evacuate citizens"',
        '"security alert" embassy',
        '"Level 4" travel advisory',
        '"Level 3" travel advisory',
        '"reconsider travel"',
        '"travel restriction"',
    ]
    for q in advisory:
        _add(q)

    return queries
