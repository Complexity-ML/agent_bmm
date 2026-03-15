# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized config system (`agent_bmm/config.py`) — all settings via `agent-bmm.yaml`, `.env`, or env vars
- Config validation with pydantic (`config_schema.py`)
- Claude API native support (Anthropic Messages API)
- Ollama local model support (`ollama:codellama` syntax)
- Model fallback chain (`llm/fallback.py`)
- Model router — auto-select LLM per query complexity
- Embedding-based tool routing via sentence-transformers
- Rate limiter with exponential backoff (`llm/rate_limiter.py`)
- HTTP connection pooling for LLM backends
- LRU cache for tool results + query deduplication (`core/cache.py`)
- Lazy tool loading — deps imported only when tool is used
- Progress spinner during LLM calls
- Streaming diff display — char-by-char animation
- Diff preview before applying edits (y/n/a confirmation)
- Token budget per step — warn at 80%, stop at 100%
- Cost estimation before execution
- Multi-file edit — edit 2+ files in one action
- Regex-based file edit
- Parallel tool execution via asyncio
- Git PR workflow — branch, commit, push, open PR in one action
- Auto-fix on test failure — retry with error feedback
- Undo stack — multi-level rollback with history
- Color themes (dark/light/minimal)
- Custom system prompts per project (`.agent-bmm-prompt`)
- Codebase summarizer for efficient LLM context
- Session memory — remember preferences across sessions
- Export conversation to markdown or JSON
- File watcher — auto-reload on external changes
- Session history — `agent-bmm history` command
- Config profiles (`~/.agent-bmm/profiles/`)
- CLI autocomplete (bash/zsh/fish)
- Smart context — TF-IDF file ranking
- Conversation branching — parallel reasoning paths
- Agent self-reflection — evaluate own reasoning quality
- Multi-agent debate — agents argue, judge synthesizes
- Tool chaining / pipelines
- Long-term memory with FAISS vector store
- Image analysis tool (OpenAI Vision)
- Audio transcription tool (Whisper)
- Prometheus metrics endpoint
- OAuth2 authentication (JWT)
- Audit logging (JSONL)
- Encryption at rest (Fernet)
- Security policies: command whitelist, file blacklist, input sanitization
- Per-key rate limiting tiers (free/pro/enterprise)
- Network isolation for code execution
- Plugin marketplace — search/install from registry
- Integration tests with mock LLM (8 tests)
- Benchmark suite: steps per task
- vLLM integration guide
- Dockerfile for `docker run agent-bmm serve`
- `pip install agent-bmm` with optional deps (`[gpu]`, `[smart]`, `[browser]`)

### Changed
- Provider detection is now data-driven (tables instead of hardcoded URLs)
- CLI args override config file values (config < env vars < CLI)
- `pyproject.toml` restructured with optional dependency groups

## [0.1.0] - 2026-03-15

### Added
- Initial release
- BMM Router — GPU-accelerated parallel tool dispatch
- ReAct agent loop (Think → Route → Act → Observe)
- Coder agent — read, write, edit, run, commit
- Chat mode — interactive coding sessions
- WebSocket server for real-time agent API
- 10 built-in tools (search, code, math, file, sql, api, github, slack, docker, browser)
- Rich terminal UI with colored output
- Cost tracking per model
- Permission system (ask/allow_reads/yolo)
- Git checkpoint rollback
- Streaming token display
