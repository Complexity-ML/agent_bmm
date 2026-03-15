# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
SQL Tool — Execute read-only SQL queries against any database.
Supports SQLite, PostgreSQL, MySQL via standard Python DB-API.
"""

from __future__ import annotations

import sqlite3

from agent_bmm.tools.registry import Tool


def create_sql_tool(
    db_path: str | None = None,
    connection_string: str | None = None,
    max_rows: int = 50,
    read_only: bool = True,
) -> Tool:
    """
    Create a SQL query tool.

    Args:
        db_path: Path to SQLite database file.
        connection_string: PostgreSQL/MySQL connection string.
        max_rows: Maximum rows to return.
        read_only: If True, only SELECT queries allowed.
    """

    def _validate_query(query: str) -> str:
        """Validate and sanitize SQL query."""
        q = query.strip().upper()
        if read_only:
            forbidden = [
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "ALTER",
                "CREATE",
                "TRUNCATE",
            ]
            for word in forbidden:
                if word in q:
                    return f"Error: {word} queries not allowed in read-only mode"
        return ""

    def _execute_sqlite(query: str) -> str:
        error = _validate_query(query)
        if error:
            return error

        try:
            conn = sqlite3.connect(db_path, timeout=5)
            if read_only:
                conn.execute("PRAGMA query_only = ON")
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            conn.close()

            if not rows:
                return "Query returned no results."

            # Format as table
            header = " | ".join(columns)
            separator = "-" * len(header)
            lines = [header, separator]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            if len(rows) == max_rows:
                lines.append(f"... (limited to {max_rows} rows)")
            return "\n".join(lines)
        except Exception as e:
            return f"SQL Error: {e}"

    def _execute_pg(query: str) -> str:
        error = _validate_query(query)
        if error:
            return error

        try:
            import psycopg2

            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            conn.close()

            if not rows:
                return "Query returned no results."

            header = " | ".join(columns)
            separator = "-" * len(header)
            lines = [header, separator]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)
        except ImportError:
            return "Error: psycopg2 not installed. pip install psycopg2-binary"
        except Exception as e:
            return f"SQL Error: {e}"

    if db_path:
        fn = _execute_sqlite
    elif connection_string:
        fn = _execute_pg
    else:

        def fn(q):
            return "Error: no database configured"

    return Tool(
        name="sql",
        description="Execute read-only SQL queries against a database",
        fn=fn,
    )


SQLTool = create_sql_tool
