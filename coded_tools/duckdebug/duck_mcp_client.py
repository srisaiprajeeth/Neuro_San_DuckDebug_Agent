# Copyright © 2025-2026 Cognizant Technology Solutions Corp, www.cognizant.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# END COPYRIGHT
"""Shared MCP client plumbing for the DuckDebug coded tools.

DuckDebug's two data tools (``score_clarity`` and ``find_similar``) are served by
``servers/mcp/duck_mcp.py``. The CodedTools in this package reach them through
this helper so the NSFlow graph shows a live MCP round-trip, following the same
``MultiServerMCPClient`` pattern as ``coded_tools/tools/mcp_bmi_streamable_http``.

If the MCP server is not running, the helper falls back to calling
:mod:`coded_tools.duckdebug.duck_core` in-process. That keeps the agent network
usable out-of-the-box (no second terminal required) while still exercising the
MCP path whenever the server is up. Set ``DUCK_USE_MCP=0`` to skip MCP entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from typing import Dict

logger = logging.getLogger(__name__)

# Must match the port in servers/mcp/duck_mcp.py. 8100 avoids clashing with the
# bmi_server example, which uses 8000.
# No trailing slash: the ASGI app serves /mcp and answers /mcp/ with a 307, which
# costs an extra round trip on every single call.
DUCK_MCP_URL = os.environ.get("DUCK_MCP_URL", "http://localhost:8100/mcp")

# Reference key for the server inside MultiServerMCPClient; local to the client.
_SERVER_KEY = "duckdebug"

# Discovered tools, cached process-wide. Listing tools is a full MCP session of
# its own, so re-discovering them on every agent turn doubles the latency.
_tools_by_name: Dict[str, Any] = {}
_tools_lock = asyncio.Lock()


def mcp_enabled() -> bool:
    """True unless the operator opted out with DUCK_USE_MCP=0."""
    return os.environ.get("DUCK_USE_MCP", "1").strip().lower() not in ("0", "false", "no")


def _maybe_json(value: Any) -> Any:
    """json.loads a string if it parses; otherwise hand the value back untouched."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _coerce(result: Any, expect_list: bool = False) -> Any:
    """Normalise an MCP tool result into plain Python.

    FastMCP emits one text content block per returned item, so the langchain
    adapter hands back different shapes depending on how many items came back:
    a dict-returning tool yields a single JSON string, while a list-returning
    tool yields a list of JSON strings for 2+ items but a bare JSON string for
    exactly one. ``expect_list`` re-wraps that one-item case so callers always
    get the shape their tool signature promises.
    """
    # Some adapter versions return (content, artifact).
    if isinstance(result, tuple) and result:
        result = result[0]

    if result is None:
        return [] if expect_list else None

    if isinstance(result, list):
        return [_maybe_json(item) for item in result]

    parsed = _maybe_json(result)

    if expect_list and not isinstance(parsed, list):
        # A single retrieved row (or an empty-string response) came back flat.
        return [] if parsed == "" or parsed is None else [parsed]

    return parsed


async def call_duck_tool(tool_name: str, payload: Dict[str, Any], expect_list: bool = False) -> Any:
    """Invoke ``tool_name`` on the DuckDebug MCP server and return plain Python.

    Set ``expect_list`` for tools that return a list, so a single-item result
    still arrives as a list. Raises on any transport/lookup failure so callers
    can decide whether to fall back to in-process execution.
    """
    tool = await _get_tool(tool_name)
    return _coerce(await tool.ainvoke(payload), expect_list=expect_list)


async def _get_tool(tool_name: str):
    """Return the named MCP StructuredTool, discovering the toolset once per process."""
    if tool_name in _tools_by_name:
        return _tools_by_name[tool_name]

    async with _tools_lock:
        # Re-check: another coroutine may have populated the cache while we waited.
        if tool_name in _tools_by_name:
            return _tools_by_name[tool_name]

        # Imported lazily so this module stays importable when
        # langchain-mcp-adapters is absent and the caller only wants the fallback.
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            connections={
                _SERVER_KEY: {
                    # streamable_http is preferred over stdio as the transport method.
                    "url": DUCK_MCP_URL,
                    "transport": "streamable_http",
                }
            }
        )
        discovered = {tool.name: tool for tool in await client.get_tools()}
        if tool_name not in discovered:
            raise RuntimeError(f"MCP server at {DUCK_MCP_URL} exposes {sorted(discovered)}, not '{tool_name}'")

        _tools_by_name.update(discovered)
        return _tools_by_name[tool_name]


async def call_with_fallback(tool_name: str, payload: Dict[str, Any], local_fn, expect_list: bool = False) -> Any:
    """Try the MCP server first, then fall back to running ``local_fn`` in-process.

    ``local_fn`` is a zero-arg callable so each tool decides how to map its own
    arguments onto :mod:`duck_core`.
    """
    if mcp_enabled():
        try:
            return await call_duck_tool(tool_name, payload, expect_list=expect_list)
        except Exception as exc:
            # Expected whenever duck_mcp.py isn't running - log once and degrade.
            logger.info("DuckDebug MCP call '%s' unavailable (%s); using in-process duck_core.", tool_name, exc)

    return local_fn()
