"""Test circuit breaker state machine."""

import time

from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def test_starts_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1.0)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True


def test_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=1.0)
    cb.record_failure(ConnectionError("fail 1"))
    assert cb.state == CircuitState.CLOSED
    cb.record_failure(ConnectionError("fail 2"))
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False


def test_half_open_after_timeout():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure(ConnectionError("fail"))
    assert cb.state == CircuitState.OPEN
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN


def test_success_closes_half_open():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure(ConnectionError("fail"))
    time.sleep(0.15)
    _ = cb.state  # triggers transition to HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_context_manager_success():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=1.0)
    with cb:
        pass  # success
    assert cb.state == CircuitState.CLOSED


def test_context_manager_raises_when_open():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=100.0)
    cb.record_failure(ConnectionError("fail"))
    try:
        with cb:
            pass
        assert False, "Should have raised CircuitOpenError"
    except CircuitOpenError:
        pass
