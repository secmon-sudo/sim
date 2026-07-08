"""
SIM — TokenBucket Rate Limiter
Blueprint V20.1 §4.5.3

Thread-safe sliding-window token bucket with reliable day-boundary reset.
Each LLM account gets its own TokenBucket instance.
"""

import datetime
import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """
    Thread-safe sliding-window token bucket rate limiter.

    rate_per_minute : maximum requests allowed per minute
    daily_limit     : hard daily cap (None = unlimited)
    burst           : max tokens held at once (None = rate_per_minute). Caps the
                      initial/idle burst so we don't fire a full minute of requests
                      back-to-back and trip provider RPM/TPM limits.
    tpm_limit       : tokens-per-minute ceiling (None = untracked). Modeled as a second
                      sliding window because for Groq's free tier TPM (8K) is far tighter
                      than RPM (30) — a burst of requests trips TPM long before RPM, which
                      is what drove the "all accounts 429" cascade. acquire() charges the
                      caller's estimated tokens against this window.
    """
    rate_per_minute: float
    daily_limit: int | None = None
    burst: float | None = None
    tpm_limit: int | None = None

    _tokens: float = field(init=False)
    _tpm_tokens: float = field(init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)
    _daily_used: int = field(default=0, init=False)
    _current_day: datetime.date = field(default_factory=datetime.date.today, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.rate_per_minute
        self._tokens = self.burst
        self._tpm_tokens = float(self.tpm_limit) if self.tpm_limit is not None else 0.0

    def acquire(self, est_tokens: int = 0, timeout: float = 300.0) -> bool:
        """Block until a request slot is available or timeout is reached.

        est_tokens: estimated tokens this call will consume (prompt + max_tokens),
        charged against the TPM window. Ignored when tpm_limit is None.

        Raises:
            RuntimeError: If daily quota is exhausted.
            TimeoutError: If a slot cannot be acquired within timeout.
        """
        deadline = time.monotonic() + timeout

        # Always attempt at least once (handles timeout=0)
        while True:
            with self._lock:
                self._refill()
                effective_limit = self.daily_limit if self.daily_limit is not None else float("inf")
                if self._daily_used >= effective_limit:
                    raise RuntimeError(
                        "Daily LLM quota exhausted. Pipeline will resume tomorrow."
                    )
                # A request needs both an RPM token and enough TPM budget. Clamp the TPM
                # requirement so an oversized prompt just waits for a near-full window
                # rather than deadlocking on a demand no window can ever satisfy.
                tpm_need = 0.0 if self.tpm_limit is None else min(est_tokens, self.tpm_limit)
                if self._tokens >= 1 and self._tpm_tokens >= tpm_need:
                    self._tokens -= 1
                    self._daily_used += 1
                    if self.tpm_limit is not None:
                        self._tpm_tokens = max(0.0, self._tpm_tokens - est_tokens)
                    return True

            if time.monotonic() >= deadline:
                break
            time.sleep(min(1.0, max(0.01, deadline - time.monotonic())))

        raise TimeoutError(
            f"Rate limiter: could not acquire token within {timeout}s"
        )

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.burst,
            self._tokens + elapsed * (self.rate_per_minute / 60.0),
        )
        if self.tpm_limit is not None:
            self._tpm_tokens = min(
                float(self.tpm_limit),
                self._tpm_tokens + elapsed * (self.tpm_limit / 60.0),
            )
        self._last_refill = now
        # Reliable day-boundary reset using date comparison
        today = datetime.date.today()
        if self._current_day != today:
            self._daily_used = 0
            self._current_day = today

    @property
    def daily_used(self) -> int:
        """Number of requests used today."""
        return self._daily_used

    @property
    def remaining_daily(self) -> int | None:
        """Remaining daily requests. None if unlimited."""
        if self.daily_limit is None:
            return None
        return max(0, self.daily_limit - self._daily_used)

    @property
    def utilization_pct(self) -> float:
        """Daily utilization percentage."""
        if self.daily_limit is None or self.daily_limit == 0:
            return 0.0
        return round(self._daily_used / self.daily_limit * 100, 1)
