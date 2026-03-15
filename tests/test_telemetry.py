# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import sys
sys.path.insert(0, ".")

from agent_bmm.telemetry import MetricsCollector


def test_counters():
    m = MetricsCollector()
    m.inc_queries()
    m.inc_queries()
    m.inc_tool_calls("search")
    m.inc_tool_calls("search")
    m.inc_tool_calls("math")
    m.inc_tool_errors("search")

    data = m.to_json()
    assert data["counters"]["agent_bmm_queries_total"] == 2
    assert data["agent_bmm_tool_calls_total"]["search"] == 2
    assert data["agent_bmm_tool_calls_total"]["math"] == 1
    assert data["agent_bmm_tool_errors_total"]["search"] == 1
    print("OK: counters")


def test_histograms():
    m = MetricsCollector()
    m.observe_query_duration(100)
    m.observe_query_duration(200)
    m.observe_query_duration(150)

    data = m.to_json()
    stats = data["agent_bmm_query_duration_ms"]
    assert stats["count"] == 3
    assert stats["avg_ms"] == 150.0
    print("OK: histograms")


def test_prometheus_export():
    m = MetricsCollector()
    m.inc_queries()
    m.inc_routing("search")
    m.observe_query_duration(42.0)
    m.set_active_connections(5)

    prom = m.to_prometheus()
    assert "agent_bmm_queries_total 1" in prom
    assert 'agent_bmm_routing_decisions{name="search"} 1' in prom
    assert "agent_bmm_active_connections 5" in prom
    assert "agent_bmm_uptime_seconds" in prom
    print("OK: prometheus export")


def test_reset():
    m = MetricsCollector()
    m.inc_queries()
    m.reset()
    data = m.to_json()
    assert data["counters"] == {}
    print("OK: reset")


if __name__ == "__main__":
    test_counters()
    test_histograms()
    test_prometheus_export()
    test_reset()
    print("\nAll telemetry tests passed!")
