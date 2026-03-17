"""Circuit breaker pattern for external service calls.

Prevents cascading failures when external APIs (Nemotron, Yahoo, Brave, BrightData) are down.
"""

import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Simple circuit breaker with configurable thresholds.

    Usage:
        cb = CircuitBreaker("nemotron", failure_threshold=3, recovery_timeout=30)
        with cb:
            result = await call_api()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (ConnectionError, TimeoutError),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker [{self.name}]: OPEN -> HALF_OPEN")
        return self._state

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(f"Circuit breaker [{self.name}]: HALF_OPEN -> CLOSED")
        self._failure_count = 0

    def record_failure(self, exc: Exception):
        self._failure_count += 1
        self._last_failure_time = time.time()
        logger.warning(
            f"Circuit breaker [{self.name}]: failure {self._failure_count}/{self.failure_threshold} — {exc}"
        )
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(f"Circuit breaker [{self.name}]: CLOSED -> OPEN")

    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN

    def __enter__(self):
        if not self.can_execute():
            raise CircuitOpenError(
                f"Circuit breaker [{self.name}] is OPEN — service unavailable"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        elif exc_type and issubclass(exc_type, self.expected_exceptions):
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ── Pre-configured breakers for each service ──────────────────────────────

nemotron_breaker = CircuitBreaker(
    "nemotron", failure_threshold=3, recovery_timeout=30,
    expected_exceptions=(ConnectionError, TimeoutError, RuntimeError),
)

yfinance_breaker = CircuitBreaker(
    "yfinance", failure_threshold=5, recovery_timeout=60,
    expected_exceptions=(Exception,),
)

brave_breaker = CircuitBreaker(
    "brave_search", failure_threshold=10, recovery_timeout=120,
    expected_exceptions=(ConnectionError, TimeoutError),
)

brightdata_breaker = CircuitBreaker(
    "brightdata", failure_threshold=3, recovery_timeout=120,
    expected_exceptions=(ConnectionError, TimeoutError),
)
