# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Project Templates — Scaffold projects from templates.

Usage:
    agent-bmm init flask
    agent-bmm init fastapi
    agent-bmm init cli

Templates are defined as dicts of {path: content}. Easy to extend.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES: dict[str, dict[str, str]] = {
    "flask": {
        "app.py": (
            "from flask import Flask\n\n"
            "app = Flask(__name__)\n\n\n"
            '@app.route("/")\n'
            "def index():\n"
            '    return "Hello, World!"\n\n\n'
            'if __name__ == "__main__":\n'
            "    app.run(debug=True)\n"
        ),
        "requirements.txt": "flask>=3.0\n",
        "README.md": "# Flask App\n\n`python app.py`\n",
    },
    "fastapi": {
        "main.py": (
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n\n"
            '@app.get("/")\n'
            "def read_root():\n"
            '    return {"message": "Hello, World!"}\n'
        ),
        "requirements.txt": "fastapi>=0.110\nuvicorn>=0.29\n",
        "README.md": "# FastAPI App\n\n`uvicorn main:app --reload`\n",
    },
    "cli": {
        "cli.py": (
            "import argparse\n\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("name", help="Your name")\n'
            "    args = parser.parse_args()\n"
            '    print(f"Hello, {args.name}!")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        ),
        "README.md": "# CLI Tool\n\n`python cli.py yourname`\n",
    },
    "react": {
        "package.json": (
            '{\n  "name": "my-app",\n  "version": "0.1.0",\n'
            '  "scripts": {"dev": "vite", "build": "vite build"},\n'
            '  "dependencies": {"react": "^18.0", "react-dom": "^18.0"},\n'
            '  "devDependencies": {"vite": "^5.0", "@vitejs/plugin-react": "^4.0"}\n'
            "}\n"
        ),
        "src/App.jsx": ("export default function App() {\n  return <h1>Hello, World!</h1>;\n}\n"),
        "README.md": "# React App\n\n`npm install && npm run dev`\n",
    },
}


def scaffold(template_name: str, target_dir: str | Path) -> list[str]:
    """
    Create project files from a template.

    Args:
        template_name: Template name (flask, fastapi, cli, react).
        target_dir: Directory to create files in.

    Returns:
        List of created file paths.
    """
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

    target = Path(target_dir)
    created = []

    for rel_path, content in TEMPLATES[template_name].items():
        full = target / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        created.append(str(rel_path))

    return created


def list_templates() -> list[str]:
    """List available template names."""
    return list(TEMPLATES.keys())
