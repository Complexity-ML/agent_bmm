#!/usr/bin/env python3
# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Benchmark: BMM parallel dispatch vs sequential dispatch.

Measures the speedup of routing N queries to K tools via BMM
compared to the traditional sequential loop approach.
"""

import time
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.table import Table

console = Console()


def bench_sequential(x: torch.Tensor, weights_up: torch.Tensor, weights_down: torch.Tensor, expert_ids: torch.Tensor, num_experts: int):
    """Sequential dispatch — one expert at a time (LangChain-style)."""
    N, H = x.shape
    I = weights_up.shape[2]
    output = torch.zeros_like(x)
    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        x_e = x[mask]
        up = F.silu(x_e @ weights_up[e])
        out_e = up @ weights_down[e]
        output[mask] = out_e
    return output


def bench_bmm(x: torch.Tensor, weights_up: torch.Tensor, weights_down: torch.Tensor, expert_ids: torch.Tensor, num_experts: int):
    """BMM parallel dispatch — all experts at once."""
    sel_up = weights_up[expert_ids]
    sel_down = weights_down[expert_ids]
    up = torch.bmm(x.unsqueeze(1), sel_up).squeeze(1)
    activated = F.silu(up)
    return torch.bmm(activated.unsqueeze(1), sel_down).squeeze(1)


def run_benchmark(N: int, H: int, I: int, K: int, device: str, warmup: int = 5, iters: int = 50):
    """Run a single benchmark configuration."""
    x = torch.randn(N, H, device=device)
    weights_up = torch.randn(K, H, I, device=device)
    weights_down = torch.randn(K, I, H, device=device)
    expert_ids = torch.randint(0, K, (N,), device=device)

    # Warmup
    for _ in range(warmup):
        bench_sequential(x, weights_up, weights_down, expert_ids, K)
        bench_bmm(x, weights_up, weights_down, expert_ids, K)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark sequential
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        bench_sequential(x, weights_up, weights_down, expert_ids, K)
    if device == "cuda":
        torch.cuda.synchronize()
    seq_time = (time.perf_counter() - t0) / iters * 1000

    # Benchmark BMM
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        bench_bmm(x, weights_up, weights_down, expert_ids, K)
    if device == "cuda":
        torch.cuda.synchronize()
    bmm_time = (time.perf_counter() - t0) / iters * 1000

    # Verify correctness
    out_seq = bench_sequential(x, weights_up, weights_down, expert_ids, K)
    out_bmm = bench_bmm(x, weights_up, weights_down, expert_ids, K)
    max_diff = (out_seq - out_bmm).abs().max().item()

    return seq_time, bmm_time, max_diff


def main():
    console.print("[bold cyan]Agent BMM Benchmark: BMM vs Sequential Dispatch[/]\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: [bold]{device}[/]")
    if device == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name()}\n")

    configs = [
        # (N queries, H hidden, I intermediate, K tools)
        (1, 256, 128, 4),
        (4, 256, 128, 4),
        (16, 256, 128, 4),
        (64, 256, 128, 4),
        (256, 256, 128, 4),
        (1024, 256, 128, 4),
        (16, 768, 256, 4),
        (64, 768, 256, 4),
        (256, 768, 256, 4),
        (16, 768, 256, 8),
        (64, 768, 256, 8),
        (256, 768, 256, 16),
    ]

    table = Table(title="BMM vs Sequential Dispatch", border_style="cyan")
    table.add_column("N queries", style="bold", width=10)
    table.add_column("H×I", width=8)
    table.add_column("K tools", width=8)
    table.add_column("Sequential", style="yellow", width=12)
    table.add_column("BMM", style="green", width=12)
    table.add_column("Speedup", style="bold cyan", width=10)
    table.add_column("Max diff", style="dim", width=10)

    for N, H, I, K in configs:
        seq_ms, bmm_ms, diff = run_benchmark(N, H, I, K, device)
        speedup = seq_ms / bmm_ms if bmm_ms > 0 else float("inf")

        speedup_style = "bold green" if speedup > 1.5 else "yellow" if speedup > 1.0 else "red"

        table.add_row(
            str(N),
            f"{H}×{I}",
            str(K),
            f"{seq_ms:.3f}ms",
            f"{bmm_ms:.3f}ms",
            f"[{speedup_style}]{speedup:.1f}x[/]",
            f"{diff:.2e}",
        )

    console.print(table)

    console.print("\n[bold]Summary:[/]")
    console.print("  BMM dispatch is a single batched operation — O(1) kernel launches")
    console.print("  Sequential dispatch has O(K) kernel launches per forward pass")
    console.print("  Speedup increases with batch size N and number of tools K")


if __name__ == "__main__":
    main()
