# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Prometheus Metrics — HTTP /metrics endpoint for monitoring.

Exposes agent performance metrics for Prometheus scraping.
Start with: agent-bmm serve --metrics-port 9090
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """Collect and expose Prometheus-format metrics."""

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0):
        self._counters[name] += value

    def observe(self, name: str, value: float):
        self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float):
        self._gauges[name] = value

    def format_prometheus(self) -> str:
        """Format all metrics in Prometheus text exposition format."""
        lines = []
        lines.append("# HELP agent_bmm_uptime_seconds Time since start")
        lines.append("# TYPE agent_bmm_uptime_seconds gauge")
        lines.append(f"agent_bmm_uptime_seconds {time.time() - self._start_time:.1f}")

        for name, value in self._counters.items():
            safe = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE agent_bmm_{safe}_total counter")
            lines.append(f"agent_bmm_{safe}_total {value}")

        for name, value in self._gauges.items():
            safe = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE agent_bmm_{safe} gauge")
            lines.append(f"agent_bmm_{safe} {value}")

        for name, values in self._histograms.items():
            if not values:
                continue
            safe = name.replace(".", "_").replace("-", "_")
            sum(values) / len(values)
            lines.append(f"# TYPE agent_bmm_{safe} summary")
            lines.append(f'agent_bmm_{safe}{{quantile="0.5"}} {sorted(values)[len(values) // 2]}')
            lines.append(f'agent_bmm_{safe}{{quantile="0.99"}} {sorted(values)[-1]}')
            lines.append(f"agent_bmm_{safe}_sum {sum(values)}")
            lines.append(f"agent_bmm_{safe}_count {len(values)}")

        return "\n".join(lines) + "\n"


# Global metrics instance
metrics = MetricsCollector()


async def metrics_server(host: str = "0.0.0.0", port: int = 9090):
    """Start a simple HTTP server for /metrics endpoint."""
    from aiohttp import web

    async def handle_metrics(request: Any) -> web.Response:
        return web.Response(text=metrics.format_prometheus(), content_type="text/plain")

    app = web.Application()
    app.router.add_get("/metrics", handle_metrics)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
