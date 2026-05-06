"""
SIM — Anchor Normalization
Blueprint V20.1 §2.2

Normalizes raw airport/location text to IATA/ICAO codes using
exact match, alias lookup, and trigram fuzzy matching.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def normalize_anchor(raw_text: str, db_conn) -> tuple[str | None, float]:
    """
    Normalize raw location text to IATA/ICAO code.

    Returns:
        (normalized_id, confidence)
        normalized_id: IATA code (preferred), ICAO, or None
        confidence: 1.0 (exact match), 0.8 (alias), 0.6 (fuzzy), 0.0 (not found)
    """
    # Input guard: reject non-string, empty, or excessively long input
    if not isinstance(raw_text, str) or len(raw_text) > 200:
        return None, 0.0
    raw_text = raw_text.strip()
    if not raw_text:
        return None, 0.0

    try:
        # 1. Direct IATA / ICAO exact match (case insensitive)
        if re.match(r"^[A-Za-z]{3,4}$", raw_text):
            upper_text = raw_text.upper()
            row = db_conn.execute(
                "SELECT iata_code FROM anchor_master WHERE iata_code=%s OR icao_code=%s",
                (upper_text, upper_text),
            ).fetchone()
            if row:
                return row[0], 1.0

        # 2. Case-insensitive alias JSONB search
        row = db_conn.execute(
            "SELECT iata_code FROM anchor_master WHERE aliases @> %s::jsonb",
            (json.dumps([raw_text]),),
        ).fetchone()
        if row:
            return row[0], 0.8

        # 3. Trigram fuzzy match (pg_trgm)
        row = db_conn.execute(
            """SELECT iata_code, similarity(canonical_name, %s) AS sim
               FROM anchor_master
               WHERE similarity(canonical_name, %s) > 0.5
               ORDER BY sim DESC LIMIT 1""",
            (raw_text, raw_text),
        ).fetchone()
        if row:
            return row[0], round(row[1] * 0.6, 2)

    except Exception:
        logger.exception("Anchor normalization error for: %s", raw_text[:50])

    return None, 0.0


def get_anchor_confidence_level(confidence: float) -> str:
    """Convert numeric confidence to tier string."""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"
