# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Telemetry — Metrics collection for monitoring.

Tracks BMM routing decisions, tool latencies, error rates,
and throughput. Exposes metrics via a simple HTTP endpoint
compatible with Prometheus scraping.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and exposes agent metrics.

    Metrics:
        - agent_bmm_queries_total: Total queries processed
        - agent_bmm_query_duration_ms: Query processing time
        - agent_bmm_routing_decisions: Routing decisions per expert
        - agent_bmm_tool_calls_total: Tool calls per tool
        - agent_bmm_tool_duration_ms: Tool execution time
        - agent_bmm_tool_errors_total: Tool errors per tool
        - agent_bmm_llm_calls_total: LLM API calls
        - agent_bmm_llm_duration_ms: LLM call duration
        - agent_bmm_active_connections: Current WebSocket connections
    """

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._labeled_counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._start_time = time.time()

    # Counters
    def inc_queries(self):
        self._counters["agent_bmm_queries_total"] += 1

    def inc_tool_calls(self, tool_name: str):
        self._labeled_counters["agent_bmm_tool_calls_total"][tool_name] += 1

    def inc_tool_errors(self, tool_name: str):
        self._labeled_counters["agent_bmm_tool_errors_total"][tool_name] += 1

    def inc_llm_calls(self):
        self._counters["agent_bmm_llm_calls_total"] += 1

    def inc_routing(self, expert_name: str):
        self._labeled_counters["agent_bmm_routing_decisions"][expert_name] += 1

    # Histograms
    def observe_query_duration(self, ms: float):
        self._histograms["agent_bmm_query_duration_ms"].append(ms)

    def observe_tool_duration(self, tool_name: str, ms: float):
        self._histograms[f"agent_bmm_tool_duration_ms_{tool_name}"].append(ms)

    def observe_llm_duration(self, ms: float):
        self._histograms["agent_bmm_llm_duration_ms"].append(ms)

    # Gauges
    def set_active_connections(self, count: int):
        self._gauges["agent_bmm_active_connections"] = count

    # Export
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        uptime = time.time() - self._start_time

        # Uptime
        lines.append("# HELP agent_bmm_uptime_seconds Agent uptime")
        lines.append("# TYPE agent_bmm_uptime_seconds gauge")
        lines.append(f"agent_bmm_uptime_seconds {uptime:.1f}")

        # Counters
        for name, value in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Labeled counters
        for name, labels in self._labeled_counters.items():
            lines.append(f"# TYPE {name} counter")
            for label, value in labels.items():
                lines.append(f'{name}{{name="{label}"}} {value}')

        # Histograms (summary stats)
        for name, values in self._histograms.items():
            if values:
                sum(values) / len(values)
                p50 = sorted(values)[len(values) // 2]
                p99 = sorted(values)[int(len(values) * 0.99)]
                lines.append(f"# TYPE {name} summary")
                lines.append(f'{name}{{quantile="0.5"}} {p50:.1f}')
                lines.append(f'{name}{{quantile="0.99"}} {p99:.1f}')
                lines.append(f"{name}_sum {sum(values):.1f}")
                lines.append(f"{name}_count {len(values)}")

        # Gauges
        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"

    def to_json(self) -> dict:
        """Export metrics as JSON."""
        result = {
            "uptime_seconds": time.time() - self._start_time,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

        for name, labels in self._labeled_counters.items():
            result[name] = dict(labels)

        for name, values in self._histograms.items():
            if values:
                result[name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "p50_ms": sorted(values)[len(values) // 2],
                    "p99_ms": sorted(values)[int(len(values) * 0.99)],
                    "max_ms": max(values),
                }

        return result

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()
        self._labeled_counters.clear()
        self._start_time = time.time()


# Global metrics instance
metrics = MetricsCollector()
