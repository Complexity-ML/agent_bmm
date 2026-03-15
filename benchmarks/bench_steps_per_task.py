# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Benchmark: Steps Per Task — Measure how many steps different models
need for common coding tasks.

Usage:
    python benchmarks/bench_steps_per_task.py --model gpt-4o-mini
    python benchmarks/bench_steps_per_task.py --model gpt-4o --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_bmm.coder.engine import CoderAgent

# ── Benchmark tasks ──

TASKS = [
    {
        "name": "hello_world",
        "prompt": "Create a hello.py that prints 'Hello World' and run it.",
        "validate": lambda d: (d / "hello.py").exists(),
    },
    {
        "name": "fizzbuzz",
        "prompt": "Create fizzbuzz.py that prints FizzBuzz for 1-20 and run it.",
        "validate": lambda d: (d / "fizzbuzz.py").exists(),
    },
    {
        "name": "read_and_edit",
        "prompt": "Read app.py and change the greeting from 'hello' to 'bonjour'.",
        "setup": lambda d: (d / "app.py").write_text("def greet():\n    return 'hello'\n"),
        "validate": lambda d: "bonjour" in (d / "app.py").read_text(),
    },
    {
        "name": "create_and_test",
        "prompt": "Create a calc.py with add(a,b) function, then create test_calc.py and run it.",
        "validate": lambda d: (d / "calc.py").exists() and (d / "test_calc.py").exists(),
    },
]


async def run_benchmark(model: str, task: dict, max_steps: int = 15) -> dict:
    """Run a single benchmark task, return results."""
    with tempfile.TemporaryDirectory() as tmp:
        # Setup
        if "setup" in task:
            task["setup"](Path(tmp))

        agent = CoderAgent(
            model=model,
            project_dir=tmp,
            max_steps=max_steps,
            permission="yolo",
            stream=False,
        )

        t0 = time.time()
        result = await agent.arun(task["prompt"])
        elapsed = time.time() - t0

        passed = task["validate"](Path(tmp))
        steps = agent.llm.call_count if hasattr(agent.llm, "call_count") else "?"

        return {
            "task": task["name"],
            "model": model,
            "steps": steps,
            "time_s": round(elapsed, 2),
            "passed": passed,
            "result": result[:100],
            "tokens": agent._estimate_tokens(),
        }


async def run_all(models: list[str], max_steps: int = 15):
    """Run all benchmarks for all models."""
    results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        for task in TASKS:
            print(f"\n  Task: {task['name']}...")
            try:
                r = await run_benchmark(model, task, max_steps)
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  {status} | {r['steps']} steps | {r['time_s']}s | ~{r['tokens']} tokens")
                results.append(r)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "task": task["name"],
                    "model": model,
                    "error": str(e),
                })
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark: steps per task")
    parser.add_argument("-m", "--model", action="append", default=[], help="Models to benchmark")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per task")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    args = parser.parse_args()

    models = args.model or ["gpt-4o-mini"]
    results = asyncio.run(run_all(models, args.max_steps))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Task':<20} {'Model':<20} {'Steps':<8} {'Time':<8} {'Pass'}")
    print(f"{'-'*60}")
    for r in results:
        if "error" in r:
            print(f"{r['task']:<20} {r['model']:<20} {'ERR':<8}")
        else:
            print(f"{r['task']:<20} {r['model']:<20} {str(r['steps']):<8} {r['time_s']:<8} {r['passed']}")


if __name__ == "__main__":
    main()
