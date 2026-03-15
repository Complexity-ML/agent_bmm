# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Shell completions — Tab completion for agent-bmm CLI.

Usage:
    # Bash
    eval "$(agent-bmm --completions bash)"

    # Zsh
    eval "$(agent-bmm --completions zsh)"

    # Fish
    agent-bmm --completions fish | source
"""

from __future__ import annotations

BASH_COMPLETION = r"""
_agent_bmm_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    local commands="run serve batch code chat remote workflow history config"
    local code_opts="--model --dir --max-steps --permission --token-budget"
    local permissions="ask allow_reads yolo"

    case "$prev" in
        agent-bmm)
            COMPREPLY=($(compgen -W "$commands" -- "$cur"))
            ;;
        code|chat)
            COMPREPLY=($(compgen -W "$code_opts" -- "$cur"))
            ;;
        --permission)
            COMPREPLY=($(compgen -W "$permissions" -- "$cur"))
            ;;
        --model|-m)
            local models="gpt-4o gpt-4o-mini claude-sonnet-4-20250514 claude-haiku-3-5 ollama:codellama"
            COMPREPLY=($(compgen -W "$models" -- "$cur"))
            ;;
        --dir|-d)
            COMPREPLY=($(compgen -d -- "$cur"))
            ;;
        *)
            COMPREPLY=($(compgen -f -- "$cur"))
            ;;
    esac
}
complete -F _agent_bmm_completions agent-bmm
"""

ZSH_COMPLETION = r"""
#compdef agent-bmm
_agent_bmm() {
    local -a commands=(
        'run:Run a single query'
        'serve:Start WebSocket server'
        'batch:Process a batch of queries'
        'code:Coding agent'
        'chat:Interactive chat'
        'remote:Connect to remote server'
        'workflow:Run a workflow'
        'history:List previous sessions'
        'config:Config management'
    )
    _arguments '1:command:->cmds' '*::arg:->args'
    case "$state" in
        cmds) _describe 'command' commands ;;
        args)
            case "$words[1]" in
                code|chat)
                    _arguments \
                        '-m[Model]:model:(gpt-4o gpt-4o-mini claude-sonnet-4-20250514 ollama:codellama)' \
                        '-d[Directory]:dir:_directories' \
                        '--max-steps[Max steps]:steps:' \
                        '--permission[Permission]:perm:(ask allow_reads yolo)' \
                        '--token-budget[Token budget]:budget:'
                    ;;
            esac
            ;;
    esac
}
_agent_bmm
"""

FISH_COMPLETION = r"""
complete -c agent-bmm -n '__fish_use_subcommand' -a run -d 'Run a single query'
complete -c agent-bmm -n '__fish_use_subcommand' -a serve -d 'Start WebSocket server'
complete -c agent-bmm -n '__fish_use_subcommand' -a batch -d 'Process a batch'
complete -c agent-bmm -n '__fish_use_subcommand' -a code -d 'Coding agent'
complete -c agent-bmm -n '__fish_use_subcommand' -a chat -d 'Interactive chat'
complete -c agent-bmm -n '__fish_use_subcommand' -a remote -d 'Connect to remote'
complete -c agent-bmm -n '__fish_use_subcommand' -a workflow -d 'Run a workflow'
complete -c agent-bmm -n '__fish_use_subcommand' -a history -d 'List sessions'
complete -c agent-bmm -n '__fish_use_subcommand' -a config -d 'Config management'
complete -c agent-bmm -n '__fish_seen_subcommand_from code chat' -l model -s m -d 'LLM model'
complete -c agent-bmm -n '__fish_seen_subcommand_from code chat' -l permission -a 'ask allow_reads yolo'
"""


def get_completion(shell: str) -> str:
    """Get completion script for the given shell."""
    shells = {"bash": BASH_COMPLETION, "zsh": ZSH_COMPLETION, "fish": FISH_COMPLETION}
    if shell not in shells:
        return f"Unknown shell: {shell}. Available: {', '.join(shells)}"
    return shells[shell]
