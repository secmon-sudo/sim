"""SIM — named counters for the degradation paths that would otherwise be silent.

Every LLM incident this pipeline has had was a DEGRADATION, not a crash. The
reports kept arriving and kept looking right:

  * 2026-09-04 minimax-m2.7 narrated all five SITREPs and shortened every one of
    its 108 citation URLs to a bare domain. Every citation was blanked. Nothing
    failed.
  * 2026-09-04 the bulletin printed "us_coalition" in Turkish prose, eight times.
  * 2026-07-23 gemini-2.5-flash-lite had been retired early and every grounded
    call was answering 404 — for two weeks the aviation block was simply empty,
    and the blame went to a quota that was never the cause.
  * 2026-09-02 Google's gtx translate endpoint began answering 429 from every IP
    and the fallback swallowed it. The note that came out of that one says, in as
    many words: always put a counter on a silent fallback.

The deadman can already see that the pipeline STOPPED. It cannot see that the
pipeline is running perfectly and producing rubbish. That is what these are for:
a fallback path that increments nothing leaves no evidence it was taken, and a
metric nobody records is a metric nobody can alarm on.

Process-global on purpose. One run is one process, the orchestrator snapshots the
whole registry into its telemetry row at the end, and `reset()` exists so a test
(or a second run in the same process) starts from zero.
"""

import threading
from typing import Dict

_LOCK = threading.Lock()
_COUNTS: Dict[str, int] = {}


def bump(name: str, n: int = 1) -> None:
    """Record that a degradation path was taken `n` times.

    Never raises: a counter that can break the thing it measures is worse than no
    counter, and every call site here sits on an error path already.
    """
    if n <= 0:
        return
    try:
        with _LOCK:
            _COUNTS[name] = _COUNTS.get(name, 0) + n
    except Exception:  # pragma: no cover - defensive
        pass


def snapshot() -> Dict[str, int]:
    """Everything counted so far. Zero-valued names are absent, not zero: the
    registry records what HAPPENED, and a reader that needs a full key list should
    say so itself rather than trust an implicit one."""
    with _LOCK:
        return dict(_COUNTS)


def reset() -> None:
    with _LOCK:
        _COUNTS.clear()


# ── Names ──────────────────────────────────────────────────────────────────
#
# Constants rather than string literals at the call sites, because these end up in
# telemetry and in the deadman's thresholds, and a typo in one of those three
# places produces a counter that silently never fires — the exact failure the
# module exists to prevent.

# A slot answered HTTP 200 with text the CALLER rejected (see call_llm's `accept`).
# This is the citation-collapse detector: nonzero means a model wrote something
# well-formed that would not have done the job.
LLM_CONTRACT_REJECTED = "llm_contract_rejected"

# A slot answered HTTP 200 carrying an error body or an empty completion.
LLM_UNUSABLE_200 = "llm_unusable_200"

# One batch of the bulletin's direction extraction could not be parsed or called.
# Fails open by design — those events keep the unattributed default and fall to the
# regional section — which is precisely why it needs a number: the report looks
# normal, it has simply stopped saying which way anything was going.
BULLETIN_DIRECTION_BATCH_FAILED = "bulletin_direction_batch_failed"
