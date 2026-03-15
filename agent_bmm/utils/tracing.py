# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
OpenTelemetry Distributed Tracing — Trace agent operations end-to-end.

Instruments LLM calls, tool executions, and routing decisions
with OpenTelemetry spans for distributed tracing.

Requires: pip install opentelemetry-api opentelemetry-sdk
Optional: pip install opentelemetry-exporter-otlp (for Jaeger/Tempo)

Configured via agent-bmm.yaml:
    tracing:
      enabled: true
      endpoint: http://localhost:4317  # OTLP gRPC endpoint
      service_name: agent-bmm
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_tracer = None


def init_tracing(
    service_name: str = "agent-bmm",
    endpoint: str = "http://localhost:4317",
) -> bool:
    """Initialize OpenTelemetry tracing. Returns True if successful."""
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("agent_bmm")
        logger.info("OpenTelemetry tracing initialized: %s → %s", service_name, endpoint)
        return True
    except ImportError:
        logger.debug("OpenTelemetry not installed, tracing disabled")
        return False
    except Exception as e:
        logger.warning("Failed to init tracing: %s", e)
        return False


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator:
    """Create a traced span. No-op if tracing is not initialized."""
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool)) else v)
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


def trace_llm_call(model: str, messages_count: int):
    """Create a span for an LLM API call."""
    return trace_span("llm.chat", {"llm.model": model, "llm.messages": messages_count})


def trace_tool_execution(tool_name: str, query: str):
    """Create a span for a tool execution."""
    return trace_span("tool.execute", {"tool.name": tool_name, "tool.query": query[:200]})


def trace_routing(strategy: str, num_queries: int, num_tools: int):
    """Create a span for BMM routing."""
    return trace_span(
        "bmm.route",
        {"routing.strategy": strategy, "routing.queries": num_queries, "routing.tools": num_tools},
    )


def trace_agent_step(step: int, action: str):
    """Create a span for an agent step."""
    return trace_span("agent.step", {"agent.step": step, "agent.action": action})
