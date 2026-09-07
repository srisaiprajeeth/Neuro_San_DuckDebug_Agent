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
"""MCP server exposing DuckDebug's five data-grounded tools.

Coaching workflow, over the labeled Stack Overflow corpus:
    score_clarity, find_similar

Incident-triage chain, over the IT support ticket corpus:
    find_tickets, failure_pattern, sla_metrics

Run it from the repo root so ``coded_tools`` is importable:

    python servers/mcp/duck_mcp.py

The DuckDebug coded tools connect to it over streamable HTTP via
``coded_tools/duckdebug/duck_mcp_client.py``. They fall back to calling
``duck_core`` / ``ticket_core`` in-process if this server is not running, so
starting it is optional — it exists to make the MCP hop real and visible in the
graph.

Point it at different corpora (optional):

    export DUCK_DATA_CSV=/path/to/60k_stackoverflow.csv
    export DUCK_TICKET_CSV=/path/to/tickets.csv
"""

import os
import sys

from mcp.server.fastmcp import FastMCP

# Allow `python servers/mcp/duck_mcp.py` from the repo root by putting the repo
# root on sys.path, so `coded_tools.duckdebug` resolves the same way it does
# inside the neuro-san server process.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from coded_tools.duckdebug import duck_core  # noqa: E402  pylint: disable=wrong-import-position
from coded_tools.duckdebug import ticket_core  # noqa: E402  pylint: disable=wrong-import-position

# Port 8100 keeps this clear of bmi_server.py, which uses 8000.
mcp = FastMCP("duckdebug", port=8100)

# Warm the TF-IDF + classifier cache at IMPORT time, not under __main__.
# Fitting over the 45k-row corpus takes ~20s, and doing it lazily on the first
# tool call means the first user turn blocks for over a minute (it is much slower
# inside a live request than standalone). Warming under __main__ is not enough:
# the ASGI layer serves tool calls from a context that re-imports this module, so
# the __main__ guard's cache does not reach the code path that answers requests.
duck_core.warm_up()
ticket_core.warm_up()


# langchain-mcp-adapters only supports mcp tools, not resources or prompts.
# Each function's name, docstring, and type hints become the tool's name,
# description, and args_schema respectively.
@mcp.tool()
def score_clarity(text: str) -> dict:
    """Score how clearly a stuck problem is articulated (0-100), grounded in HQ vs
    LQ Stack Overflow patterns. Returns the band, what is missing, and the
    ensemble vote breakdown."""
    return duck_core.score_clarity(text)


@mcp.tool()
def find_similar(text: str, tag: str = "", k: int = 3) -> list:
    """Retrieve up to k real Stack Overflow questions framed like this problem,
    each with tags, quality label (HQ/LQ_CLOSE/LQ_EDIT) and similarity."""
    return duck_core.find_similar(text, tag=tag, k=k)


# ---------------------------------------------------------------------------
# Incident-triage chain, over the 100k-row IT support ticket corpus.
# ---------------------------------------------------------------------------


@mcp.tool()
def find_tickets(
    text: str,
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    k: int = 5,
) -> dict:
    """Find historical IT support tickets that look like this problem, each with
    its resolution, resolution time in hours, reopen flag and CSAT score."""
    return ticket_core.find_tickets(text, product_area=product_area, issue_type=issue_type, priority=priority, k=k)


@mcp.tool()
def failure_pattern(
    text: str,
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    neighbours: int = 60,
) -> dict:
    """Extract the common failure pattern across tickets similar to this problem:
    dominant product area and issue type, what actually resolved them, the share
    that got resolved at all, and the reopen rate."""
    return ticket_core.failure_pattern(
        text,
        product_area=product_area,
        issue_type=issue_type,
        priority=priority,
        neighbours=neighbours,
    )


@mcp.tool()
def sla_metrics(
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    recent_months: int = 6,
) -> dict:
    """Measure a ticket slice against SLA targets (keyed on priority) plus reopen
    rate, CSAT and volume trend, each reported beside the corpus-wide baseline."""
    return ticket_core.sla_metrics(
        product_area=product_area,
        issue_type=issue_type,
        priority=priority,
        recent_months=recent_months,
    )


if __name__ == "__main__":
    print("DuckDebug MCP on http://localhost:8100/mcp", flush=True)
    print(f"  clarity/echo  <- {duck_core.DATA_CSV}", flush=True)
    print(f"  ticket chain  <- {ticket_core.TICKET_CSV}", flush=True)
    # streamable http is preferred over stdio as the transport method.
    mcp.run(transport="streamable-http")
