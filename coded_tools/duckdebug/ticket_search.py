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
"""Incident chain stage 2 — find similar historical tickets.

Referenced in ``registries/duckdebug.hocon`` as
``"class": "ticket_search.TicketSearch"``.
"""

from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.duckdebug import ticket_core
from coded_tools.duckdebug.duck_mcp_client import call_with_fallback

_DEFAULT_K = 5
_MAX_K = 25


class TicketSearch(CodedTool):
    """Retrieves historical IT support tickets that look like the reported problem,
    each with its resolution, resolution time, reopen flag, and CSAT."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Find similar tickets in the 100k-row corpus.

        :param args: 'text' plus optional 'product_area', 'issue_type', 'priority', 'k'.
        :param sly_data: Out-of-band data shared across the agent hierarchy.
            Keys expected for this implementation: None
        :return: Matching tickets with metadata, or an error message.
        """
        text: str = args.get("text") or args.get("problem") or ""
        if not text.strip():
            return {"error": "No problem text provided. Ask the user what incident they are investigating."}

        product_area: str = args.get("product_area") or ""
        issue_type: str = args.get("issue_type") or ""
        priority: str = args.get("priority") or ""

        # The LLM may hand back k as a string.
        try:
            k = int(args.get("k", _DEFAULT_K))
        except (TypeError, ValueError):
            k = _DEFAULT_K
        k = max(1, min(k, _MAX_K))

        payload = {
            "text": text,
            "product_area": product_area,
            "issue_type": issue_type,
            "priority": priority,
            "k": k,
        }
        return await call_with_fallback(
            "find_tickets",
            payload,
            lambda: ticket_core.find_tickets(
                text,
                product_area=product_area,
                issue_type=issue_type,
                priority=priority,
                k=k,
            ),
        )
