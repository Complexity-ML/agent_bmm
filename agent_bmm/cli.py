# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
CLI — Command-line interface for Agent BMM.

Usage:
    agent-bmm run "What is quantum computing?"
    agent-bmm serve --port 8765
    agent-bmm batch queries.txt -o results.json
    agent-bmm config init
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console

console = Console()


def load_config(path: str | None = None) -> dict:
    """Load YAML or JSON config."""
    if path is None:
        for default in ["agent-bmm.yaml", "agent-bmm.yml", "agent-bmm.json"]:
            if Path(default).exists():
                path = default
                break
    if path is None:
        return {}

    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(p.read_text()) or {}
        except ImportError:
            console.print("[yellow]PyYAML not installed, trying JSON...[/]")
    return json.loads(p.read_text())


def build_agent_from_config(config: dict):
    """Build an Agent from config dict."""
    from agent_bmm.agent import Agent
    from agent_bmm.tools.builtin import (
        APITool,
        CodeExecTool,
        DockerTool,
        FileIOTool,
        GitHubTool,
        MathTool,
        SlackTool,
        SQLTool,
        WebSearchTool,
    )

    BUILTIN_TOOLS = {
        "search": WebSearchTool,
        "sql": SQLTool,
        "api": APITool,
        "file": FileIOTool,
        "code": CodeExecTool,
        "math": MathTool,
        "github": GitHubTool,
        "slack": SlackTool,
        "docker": DockerTool,
    }

    llm = config.get("llm", {})
    agent = Agent(
        model=llm.get("model", ""),
        base_url=llm.get("base_url", "http://localhost:8081/v1"),
        api_key=llm.get("api_key", ""),
        provider=llm.get("provider", "openai"),
        hidden_size=config.get("router", {}).get("hidden_size", 256),
        expert_size=config.get("router", {}).get("expert_size", 128),
        routing=config.get("router", {}).get("routing", "round_robin"),
        max_steps=config.get("max_steps", 5),
    )

    # Register tools
    tools_config = config.get("tools", ["math", "code", "search"])
    if isinstance(tools_config, list):
        for tool_name in tools_config:
            if isinstance(tool_name, str) and tool_name in BUILTIN_TOOLS:
                tool = BUILTIN_TOOLS[tool_name]()
                agent.add_tool(
                    name=tool.name,
                    description=tool.description,
                    fn=tool.fn,
                    async_fn=tool.async_fn,
                )
            elif isinstance(tool_name, dict):
                name = tool_name.get("name", "")
                if name in BUILTIN_TOOLS:
                    kwargs = {k: v for k, v in tool_name.items() if k != "name"}
                    tool = BUILTIN_TOOLS[name](**kwargs)
                    agent.add_tool(
                        name=tool.name,
                        description=tool.description,
                        fn=tool.fn,
                        async_fn=tool.async_fn,
                    )

    return agent


def cmd_run(args):
    """Run a single query."""
    config = load_config(args.config)
    if args.model:
        config.setdefault("llm", {})["model"] = args.model
    if args.tools:
        config["tools"] = args.tools.split(",")

    agent = build_agent_from_config(config)
    query = " ".join(args.query)

    answer = agent.ask_sync(query)
    console.print(answer)


def cmd_serve(args):
    """Start WebSocket server."""
    config = load_config(args.config)
    if args.model:
        config.setdefault("llm", {})["model"] = args.model

    agent = build_agent_from_config(config)

    from agent_bmm.server import run_server

    console.print(f"[bold cyan]Starting Agent BMM server on port {args.port}[/]")
    asyncio.run(run_server(agent, host=args.host, port=args.port))


def cmd_batch(args):
    """Process a batch of queries."""
    config = load_config(args.config)
    agent = build_agent_from_config(config)

    input_path = Path(args.input)
    if input_path.suffix == ".json":
        queries = json.loads(input_path.read_text())
    else:
        queries = [
            line.strip() for line in input_path.read_text().splitlines() if line.strip()
        ]

    async def process_batch():
        results = []
        for i, query in enumerate(queries):
            console.print(f"[dim][{i + 1}/{len(queries)}][/] {query[:80]}")
            try:
                answer = await agent.ask(query)
                results.append({"query": query, "answer": answer, "status": "ok"})
            except Exception as e:
                results.append({"query": query, "answer": str(e), "status": "error"})
        return results

    results = asyncio.run(process_batch())

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"[green]Results saved to {args.output}[/]")
    else:
        for r in results:
            console.print(f"[bold]Q:[/] {r['query']}")
            console.print(f"[green]A:[/] {r['answer']}\n")


def cmd_config_init(args):
    """Generate a default config file."""
    default_config = {
        "llm": {
            "provider": "openai",
            "base_url": "http://localhost:8081/v1",
            "model": "your-model-here",
            "api_key": "",
        },
        "router": {
            "hidden_size": 256,
            "expert_size": 128,
            "routing": "round_robin",
        },
        "max_steps": 5,
        "tools": ["search", "math", "code"],
    }

    out = args.output or "agent-bmm.json"
    Path(out).write_text(json.dumps(default_config, indent=2))
    console.print(f"[green]Config written to {out}[/]")
    console.print("[dim]Edit the file to configure your agent.[/]")


def main():
    parser = argparse.ArgumentParser(
        prog="agent-bmm",
        description="Agent BMM — GPU-accelerated agent framework",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a single query")
    p_run.add_argument("query", nargs="+", help="The query to ask")
    p_run.add_argument("-c", "--config", help="Config file path")
    p_run.add_argument("-m", "--model", help="LLM model name")
    p_run.add_argument("-t", "--tools", help="Comma-separated tool names")
    p_run.set_defaults(func=cmd_run)

    # serve
    p_serve = sub.add_parser("serve", help="Start WebSocket server")
    p_serve.add_argument("-c", "--config", help="Config file path")
    p_serve.add_argument("-m", "--model", help="LLM model name")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=cmd_serve)

    # batch
    p_batch = sub.add_parser("batch", help="Process a batch of queries")
    p_batch.add_argument("input", help="Input file (txt or json)")
    p_batch.add_argument("-o", "--output", help="Output JSON file")
    p_batch.add_argument("-c", "--config", help="Config file path")
    p_batch.set_defaults(func=cmd_batch)

    # config
    p_config = sub.add_parser("config", help="Config management")
    p_config_sub = p_config.add_subparsers(dest="config_command")
    p_init = p_config_sub.add_parser("init", help="Generate default config")
    p_init.add_argument("-o", "--output", help="Output file path")
    p_init.set_defaults(func=cmd_config_init)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
