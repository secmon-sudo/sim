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
    """
    rate_per_minute: float
    daily_limit: int | None = None
    burst: float | None = None

    _tokens: float = field(init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)
    _daily_used: int = field(default=0, init=False)
    _current_day: datetime.date = field(default_factory=datetime.date.today, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.rate_per_minute
        self._tokens = self.burst

    def acquire(self, timeout: float = 300.0) -> bool:
        """Block until a token is available or timeout is reached.

        Raises:
            RuntimeError: If daily quota is exhausted.
            TimeoutError: If token cannot be acquired within timeout.
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
                if self._tokens >= 1:
                    self._tokens -= 1
                    self._daily_used += 1
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
