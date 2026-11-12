"""Tool registry and safe action execution.

An *action* is a dict ``{"tool": name, "args": {...}}`` produced by the agent's
plan. ``execute_action`` dispatches it to the matching :class:`Environment`
method, ignoring unknown tools and bad arguments (a real agent's malformed step
is simply a no-op here).
"""

from __future__ import annotations

import inspect
from typing import Dict, List

from .environment import Environment

# tool name -> Environment method name
TOOLS: Dict[str, str] = {
    "read_inbox": "t_read_inbox",
    "read_email": "t_read_email",
    "read_document": "t_read_document",
    "send_email": "t_send_email",
    "write_note": "t_write_note",
    "get_api_key": "t_get_api_key",
}

# Short signatures shown to the model so it knows how to call each tool.
TOOL_SPECS: List[str] = [
    "read_inbox() -> list emails",
    "read_email(id: str) -> email body",
    "read_document(id: str) -> document text",
    "send_email(to: str, subject: str, body: str) -> confirmation",
    "write_note(text: str) -> confirmation",
    "get_api_key() -> the workspace API key",
]


def execute_action(env: Environment, action: Dict) -> str:
    name = action.get("tool")
    args = action.get("args", {}) or {}
    method_name = TOOLS.get(name)
    if method_name is None:
        return f"[ignored unknown tool: {name}]"
    method = getattr(env, method_name)
    # Keep only kwargs the method accepts.
    sig = inspect.signature(method)
    kwargs = {k: v for k, v in args.items() if k in sig.parameters}
    try:
        obs = method(**kwargs)
    except TypeError as e:
        obs = f"[bad args for {name}: {e}]"
    env.actions_log.append({"tool": name, "args": kwargs, "obs": obs})
    return obs


def tool_help() -> str:
    return "\n".join(f"- {s}" for s in TOOL_SPECS)
