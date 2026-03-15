# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
CLI — Command-line interface for Agent BMM.

All defaults come from agent_bmm.config (yaml / .env / env vars).
CLI args override config. Zero hardcoded values here.

Usage:
    agent-bmm run "What is quantum computing?"
    agent-bmm code "Add login page"
    agent-bmm chat
    agent-bmm serve
    agent-bmm config init
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console

from agent_bmm.config.config import generate_default_config, load_config

console = Console()


def _cfg(args) -> dict:
    """Load config, then overlay CLI args on top."""
    cli_overrides = {}

    # Model override → llm.model
    model = getattr(args, "model", None)
    if model:
        cli_overrides.setdefault("llm", {})["model"] = model

    # Permission, max_steps, etc. → coder.*
    for key in ("max_steps", "permission", "token_budget"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cli_overrides.setdefault("coder", {})[key] = val

    # Server overrides
    for key in ("host", "port"):
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides.setdefault("server", {})[key] = val

    # Tools override
    tools = getattr(args, "tools", None)
    if tools:
        cli_overrides["tools"] = tools.split(",")

    return load_config(
        path=getattr(args, "config", None),
        cli_overrides=cli_overrides if cli_overrides else None,
    )


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

    llm = config["llm"]
    router = config["router"]
    agent = Agent(
        model=llm["model"],
        base_url=llm["base_url"],
        api_key=llm["api_key"],
        provider=llm["provider"],
        hidden_size=router["hidden_size"],
        expert_size=router["expert_size"],
        routing=router["routing"],
        max_steps=config["coder"]["max_steps"],
    )

    # Register tools
    for tool_name in config.get("tools", []):
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


# ── Commands ──


def cmd_run(args):
    """Run a single query."""
    config = _cfg(args)
    agent = build_agent_from_config(config)
    query = " ".join(args.query)
    answer = agent.ask_sync(query)
    console.print(answer)


def cmd_serve(args):
    """Start WebSocket server."""
    config = _cfg(args)
    agent = build_agent_from_config(config)

    from agent_bmm.server.server import run_server

    srv = config["server"]
    console.print(f"[bold cyan]Starting Agent BMM server on port {srv['port']}[/]")
    asyncio.run(run_server(agent, host=srv["host"], port=srv["port"]))


def cmd_batch(args):
    """Process a batch of queries."""
    config = _cfg(args)
    agent = build_agent_from_config(config)

    input_path = Path(args.input)
    if input_path.suffix == ".json":
        queries = json.loads(input_path.read_text())
    else:
        queries = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]

    async def process_batch():
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

        results = []
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(queries))
            for query in queries:
                try:
                    answer = await agent.ask(query)
                    results.append({"query": query, "answer": answer, "status": "ok"})
                except Exception as e:
                    results.append({"query": query, "answer": str(e), "status": "error"})
                progress.advance(task)
        return results

    results = asyncio.run(process_batch())

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"[green]Results saved to {args.output}[/]")
    else:
        for r in results:
            console.print(f"[bold]Q:[/] {r['query']}")
            console.print(f"[green]A:[/] {r['answer']}\n")


def cmd_code(args):
    """Run the coding agent."""
    config = _cfg(args)
    llm = config["llm"]
    coder_cfg = config["coder"]

    from agent_bmm.coder import CoderAgent

    task = " ".join(args.task)
    coder = CoderAgent(
        model=llm["model"],
        base_url=llm["base_url"],
        api_key=llm["api_key"],
        project_dir=getattr(args, "dir", "."),
        max_steps=coder_cfg["max_steps"],
        permission=coder_cfg["permission"],
        stream=coder_cfg["stream"],
        auto_commit=coder_cfg["auto_commit"],
        token_budget=coder_cfg["token_budget"],
    )
    coder.run(task)


def cmd_remote(args):
    """Connect to remote Agent BMM server."""
    from agent_bmm.server.remote import run_remote

    run_remote(args.url)


def cmd_chat(args):
    """Interactive coding agent chat."""
    config = _cfg(args)
    llm = config["llm"]
    coder_cfg = config["coder"]

    from agent_bmm.coder.chat import ChatSession

    session = ChatSession(
        model=llm["model"],
        base_url=llm["base_url"],
        api_key=llm["api_key"],
        project_dir=getattr(args, "dir", "."),
        max_steps=coder_cfg["max_steps"],
    )
    session.run()


def cmd_workflow(args):
    """Run a YAML/JSON workflow."""
    from agent_bmm.utils.workflow import run_workflow

    asyncio.run(run_workflow(args.file, dry_run=args.dry_run, output=args.output))


def cmd_history(args):
    """List previous coding sessions."""
    from agent_bmm.utils.persistence import ConversationStore

    store = ConversationStore()
    conversations = store.get_conversations(limit=args.limit)
    store.close()

    if not conversations:
        console.print("[dim]No previous sessions found.[/]")
        return

    from datetime import datetime

    console.print(f"\n[bold cyan]Previous Sessions[/] ({len(conversations)} found)\n")
    for conv in conversations:
        ts = datetime.fromtimestamp(conv["created_at"]).strftime("%Y-%m-%d %H:%M")
        console.print(f"  [bold]#{conv['id']}[/] [dim]{ts}[/] — {conv['title']}")

    if args.show:
        messages = ConversationStore().get_messages(args.show)
        console.print(f"\n[bold]Session #{args.show}[/]\n")
        for msg in messages:
            role = msg["role"]
            color = "cyan" if role == "user" else "green"
            console.print(f"  [{color}]{role}:[/] {msg['content'][:200]}")


def cmd_config_init(args):
    """Generate a default agent-bmm.yaml config file."""
    out = args.output or "agent-bmm.yaml"
    Path(out).write_text(generate_default_config())
    console.print(f"[green]Config written to {out}[/]")
    console.print("[dim]Edit the file to configure your agent.[/]")


# ── CLI parser ──


def main():
    parser = argparse.ArgumentParser(
        prog="agent-bmm",
        description="Agent BMM — GPU-accelerated agent framework",
    )
    parser.add_argument("-c", "--config", help="Config file path (yaml/json)")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a single query")
    p_run.add_argument("query", nargs="+", help="The query to ask")
    p_run.add_argument("-m", "--model", help="LLM model (overrides config)")
    p_run.add_argument("-t", "--tools", help="Comma-separated tool names")
    p_run.set_defaults(func=cmd_run)

    # serve
    p_serve = sub.add_parser("serve", help="Start WebSocket server")
    p_serve.add_argument("-m", "--model", help="LLM model (overrides config)")
    p_serve.add_argument("--host", help="Server host")
    p_serve.add_argument("--port", type=int, help="Server port")
    p_serve.set_defaults(func=cmd_serve)

    # batch
    p_batch = sub.add_parser("batch", help="Process a batch of queries")
    p_batch.add_argument("input", help="Input file (txt or json)")
    p_batch.add_argument("-o", "--output", help="Output JSON file")
    p_batch.set_defaults(func=cmd_batch)

    # workflow
    p_wf = sub.add_parser("workflow", help="Run a YAML/JSON workflow")
    p_wf.add_argument("file", help="Workflow file (.yaml or .json)")
    p_wf.add_argument("-o", "--output", help="Save results to JSON file")
    p_wf.add_argument("--dry-run", action="store_true", help="Show tasks without executing")
    p_wf.set_defaults(func=cmd_workflow)

    # code
    p_code = sub.add_parser("code", help="Coding agent — edit your project with AI")
    p_code.add_argument("task", nargs="+", help="What to code")
    p_code.add_argument("-m", "--model", help="LLM model (overrides config)")
    p_code.add_argument("-d", "--dir", default=".", help="Project directory")
    p_code.add_argument("--max-steps", type=int, help="Max agent steps")
    p_code.add_argument("--permission", choices=["ask", "allow_reads", "yolo"], help="Permission level")
    p_code.add_argument("--token-budget", type=int, help="Max token budget (0=unlimited)")
    p_code.set_defaults(func=cmd_code)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive coding agent chat")
    p_chat.add_argument("-m", "--model", help="LLM model (overrides config)")
    p_chat.add_argument("-d", "--dir", default=".", help="Project directory")
    p_chat.add_argument("--max-steps", type=int, help="Max steps per request")
    p_chat.set_defaults(func=cmd_chat)

    # remote
    p_remote = sub.add_parser("remote", help="Connect to remote Agent BMM server")
    p_remote.add_argument("url", help="WebSocket URL (ws://host:port)")
    p_remote.set_defaults(func=cmd_remote)

    # history
    p_history = sub.add_parser("history", help="List previous coding sessions")
    p_history.add_argument("-n", "--limit", type=int, default=20, help="Max sessions to show")
    p_history.add_argument("-s", "--show", type=int, help="Show messages from session ID")
    p_history.set_defaults(func=cmd_history)

    # config
    p_config = sub.add_parser("config", help="Config management")
    p_config_sub = p_config.add_subparsers(dest="config_command")
    p_init = p_config_sub.add_parser("init", help="Generate default agent-bmm.yaml")
    p_init.add_argument("-o", "--output", help="Output file path")
    p_init.set_defaults(func=cmd_config_init)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
