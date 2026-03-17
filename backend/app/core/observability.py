"""OpenTelemetry instrumentation for the analysis pipeline.

Span structure:
  analysis_session (root)
  ├── predict_alpha
  ├── round_1
  │   └── agent:{name} (per agent)
  │       └── tool_call:{tool_name}
  ├── extract_claims
  ├── round_2
  │   └── agent:{name}_debate
  └── round_3 (synthesis)
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

_tracer: Optional[trace.Tracer] = None


def init_tracing(service_name: str = "bam-backend", endpoint: Optional[str] = None):
    """Initialize OpenTelemetry tracing.

    If endpoint is provided, exports to OTLP collector.
    Otherwise uses a no-op exporter (spans still work for local timing).
    """
    global _tracer
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            logger.info(f"OTel tracing: exporting to {endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not installed — tracing is local-only")
    else:
        logger.info("OTel tracing: local-only (no export endpoint)")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("bam")


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        init_tracing()
    return _tracer


@contextmanager
def trace_span(name: str, attributes: Optional[dict[str, Any]] = None):
    """Context manager for creating a traced span with attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool)) else v)
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise


def trace_tool_call(agent_name: str, tool_name: str, args: dict, result_summary: str, latency_ms: float):
    """Record a tool call as a span event on the current span."""
    span = trace.get_current_span()
    span.add_event(
        f"tool_call:{tool_name}",
        attributes={
            "agent": agent_name,
            "tool": tool_name,
            "args": str(args),
            "result_summary": result_summary[:500],
            "latency_ms": latency_ms,
        },
    )


def trace_llm_call(agent_name: str, prompt_tokens: int, completion_tokens: int, latency_ms: float):
    """Record an LLM call as a span event."""
    span = trace.get_current_span()
    span.add_event(
        "llm_call",
        attributes={
            "agent": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        },
    )
