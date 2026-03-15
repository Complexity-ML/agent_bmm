#!/usr/bin/env python3
# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
GPU Benchmark Suite — CUDA profiling for BMM routing.

Profiles memory usage, kernel launch overhead, throughput,
and latency across different configurations.

Usage:
    python benchmarks/bench_gpu_profiling.py
    python benchmarks/bench_gpu_profiling.py --profile  # with torch profiler
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

import torch
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
console = Console()


def profile_memory(num_tools: int, hidden_size: int, expert_size: int, device: str) -> dict:
    """Profile memory usage of BMMRouter."""
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    from agent_bmm.core.router import BMMRouter

    before = torch.cuda.memory_allocated() if device == "cuda" else 0
    router = BMMRouter(hidden_size, num_tools, expert_size, routing="learned").to(device)
    after = torch.cuda.memory_allocated() if device == "cuda" else 0

    param_bytes = sum(p.numel() * p.element_size() for p in router.parameters())

    return {
        "params": sum(p.numel() for p in router.parameters()),
        "param_mb": param_bytes / 1024 / 1024,
        "gpu_alloc_mb": (after - before) / 1024 / 1024 if device == "cuda" else 0,
    }


def profile_throughput(
    router,
    batch_sizes: list[int],
    device: str,
    warmup: int = 10,
    iters: int = 100,
) -> list[dict]:
    """Profile throughput (queries/sec) across batch sizes."""
    results = []
    for n in batch_sizes:
        x = torch.randn(n, router.hidden_size, device=device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                router(x)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                router(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters

        results.append(
            {
                "batch_size": n,
                "latency_ms": elapsed * 1000,
                "throughput_qps": n / elapsed,
            }
        )
    return results


def profile_kernel_launches(router, batch_size: int, device: str) -> dict:
    """Profile number of CUDA kernel launches."""
    if device != "cuda":
        return {"kernel_launches": "N/A (CPU)"}

    x = torch.randn(batch_size, router.hidden_size, device=device)

    # Use torch profiler to count kernels
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                router(x)
            torch.cuda.synchronize()

        events = prof.key_averages()
        cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
        return {
            "kernel_launches": len(cuda_events),
            "total_cuda_time_ms": sum(e.cuda_time_total for e in cuda_events) / 1000,
            "top_kernels": [(e.key, f"{e.cuda_time_total / 1000:.3f}ms") for e in cuda_events[:5]],
        }
    except Exception as e:
        return {"error": str(e)}


def profile_routing_strategies(hidden_size: int, num_tools: int, expert_size: int, device: str) -> list[dict]:
    """Compare latency across routing strategies."""
    from agent_bmm.core.router import BMMRouter

    results = []
    x = torch.randn(64, hidden_size, device=device)

    for strategy in ("learned", "round_robin", "embedding"):
        router = BMMRouter(hidden_size, num_tools, expert_size, routing=strategy).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                router(x)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                router(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 100

        results.append(
            {
                "strategy": strategy,
                "latency_ms": elapsed * 1000,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark Suite")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    console.print("[bold cyan]GPU Benchmark Suite[/]")
    console.print(f"Device: [bold]{device}[/]")
    if device == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name()}")
        console.print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    console.print()

    # 1. Memory profiling
    console.print("[bold]1. Memory Usage[/]")
    configs = [(4, 256, 128), (8, 256, 128), (4, 768, 256), (8, 768, 256), (16, 768, 256)]
    table = Table(border_style="cyan")
    table.add_column("Tools")
    table.add_column("H×E")
    table.add_column("Params")
    table.add_column("Param MB")
    table.add_column("GPU MB")

    for k, h, e in configs:
        mem = profile_memory(k, h, e, device)
        table.add_row(str(k), f"{h}×{e}", f"{mem['params']:,}", f"{mem['param_mb']:.2f}", f"{mem['gpu_alloc_mb']:.2f}")
    console.print(table)

    # 2. Throughput
    console.print("\n[bold]2. Throughput[/]")
    from agent_bmm.core.router import BMMRouter

    router = BMMRouter(256, 4, 128, routing="learned").to(device)
    batches = [1, 4, 16, 64, 256, 1024]
    tp = profile_throughput(router, batches, device)

    table = Table(border_style="green")
    table.add_column("Batch")
    table.add_column("Latency (ms)")
    table.add_column("Throughput (q/s)")

    for r in tp:
        table.add_row(str(r["batch_size"]), f"{r['latency_ms']:.3f}", f"{r['throughput_qps']:,.0f}")
    console.print(table)

    # 3. Routing strategy comparison
    console.print("\n[bold]3. Routing Strategies (N=64, H=256, K=4)[/]")
    strats = profile_routing_strategies(256, 4, 128, device)
    table = Table(border_style="yellow")
    table.add_column("Strategy")
    table.add_column("Latency (ms)")

    for r in strats:
        table.add_row(r["strategy"], f"{r['latency_ms']:.3f}")
    console.print(table)

    # 4. Kernel profiling (CUDA only)
    if device == "cuda" and args.profile:
        console.print("\n[bold]4. CUDA Kernel Analysis[/]")
        kernels = profile_kernel_launches(router, 64, device)
        for k, v in kernels.items():
            console.print(f"  {k}: {v}")

    console.print("\n[dim]Done.[/]")


if __name__ == "__main__":
    main()
