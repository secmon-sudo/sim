"""
Tests for the budgeted storyline narrator (Faz 3.2 LLM prose layer).

Covers the pure/token-free logic: signature stability, prompt construction, and
the cache-skip + cap behavior of run_storyline_narratives with a mocked DB/router
(no real LLM or DB).
"""

from datetime import datetime, timedelta

from src.services import storyline_narrator as narrator

_T0 = datetime(2026, 6, 8, 10, 0)


def _ev(eid, title, when, sev=70):
    return {
        "id": eid, "source_title": title, "storyline_hint": title,
        "occurred_at_est": when, "severity_score": sev,
        "source_domain": "reuters.com", "country_iso": "AF",
        "anchor_name_norm": "KBL", "event_type": "security_incident",
    }


class TestSignature:
    def test_stable_for_same_events(self):
        evs = [_ev("1", "a", _T0), _ev("2", "b", _T0 + timedelta(hours=2))]
        assert narrator.compute_signature(evs) == narrator.compute_signature(list(reversed(evs)))

    def test_changes_when_event_added(self):
        evs = [_ev("1", "a", _T0)]
        sig1 = narrator.compute_signature(evs)
        evs.append(_ev("2", "b", _T0 + timedelta(hours=2)))
        assert narrator.compute_signature(evs) != sig1


class TestPrompt:
    def test_prompt_includes_chronology_and_facts(self):
        evs = [
            _ev("1", "First blast at Kabul airport", _T0, sev=60),
            _ev("2", "Second explosion reported", _T0 + timedelta(hours=3), sev=85),
        ]
        prompt = narrator.build_narrative_prompt(evs)
        assert "Chronological reports" in prompt
        assert "First blast at Kabul airport" in prompt
        assert "peak severity 85" in prompt


# ── Mock DB / router for the run loop ──

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _MockDB:
    """Returns active-storyline rows, then events, then cached-signature lookups."""
    def __init__(self, cached_signature=None):
        self.cached_signature = cached_signature
        self.commits = 0
        self.upserts = 0

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if "GROUP BY storyline_id" in s:
            return _Cursor([("sid-1", 2, 90)])
        if "FROM events WHERE storyline_id" in s:
            return _Cursor([
                (1, "First blast", "hint", _T0, 60, "reuters.com", "AF", "KBL", "security_incident"),
                (2, "Second blast", "hint", _T0 + timedelta(hours=3), 90, "bbc.co.uk", "AF", "KBL", "security_incident"),
            ])
        if "SELECT signature FROM storyline_narratives" in s:
            return _Cursor([(self.cached_signature,)] if self.cached_signature else [])
        if "INSERT INTO storyline_narratives" in s:
            self.upserts += 1
            return _Cursor([])
        return _Cursor([])

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class _MockRouter:
    accounts = [object()]


def _patch_call_llm(monkeypatch, content="A blast struck Kabul airport, then a second explosion followed."):
    monkeypatch.setattr(narrator, "call_llm", lambda *a, **k: {
        "content": content, "provider": "groq", "model": "openai/gpt-oss-20b",
    })


class TestRunLoop:
    def test_generates_when_uncached(self, monkeypatch):
        _patch_call_llm(monkeypatch)
        db = _MockDB(cached_signature=None)
        stats = narrator.run_storyline_narratives(db, _MockRouter())
        assert stats["generated"] == 1
        assert stats["skipped_cached"] == 0
        assert db.upserts == 1
        assert db.commits == 1

    def test_skips_when_cached_signature_matches(self, monkeypatch):
        # First compute the signature the loop will produce for the mocked events.
        events = narrator.fetch_storyline_events(_MockDB(), "sid-1")
        sig = narrator.compute_signature(events)

        called = {"n": 0}
        monkeypatch.setattr(narrator, "call_llm", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {"content": "x"})

        db = _MockDB(cached_signature=sig)
        stats = narrator.run_storyline_narratives(db, _MockRouter())
        assert stats["skipped_cached"] == 1
        assert stats["generated"] == 0
        assert called["n"] == 0  # no LLM call spent on an unchanged storyline

    def test_disabled_short_circuits(self, monkeypatch):
        monkeypatch.setattr(narrator, "NARRATIVE_ENABLED", False)
        stats = narrator.run_storyline_narratives(_MockDB(), _MockRouter())
        assert stats == {"candidates": 0, "generated": 0, "skipped_cached": 0, "failed": 0}
