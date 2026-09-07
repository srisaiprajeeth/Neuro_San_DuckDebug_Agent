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
"""Incident chain stage 3 — extract the common failure pattern.

Referenced in ``registries/duckdebug.hocon`` as
``"class": "root_cause.RootCause"``.
"""

from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.duckdebug import ticket_core
from coded_tools.duckdebug.duck_mcp_client import call_with_fallback

# Aggregate over a wide neighbourhood: a "pattern" claimed from the 5 tickets the
# user was shown is noise, not a pattern.
_DEFAULT_NEIGHBOURS = 60
_MIN_NEIGHBOURS = 10
_MAX_NEIGHBOURS = 500


class RootCause(CodedTool):
    """Aggregates the neighbourhood around a reported problem into a common failure
    pattern: dominant product area and issue type, what actually resolved those
    tickets, how many got resolved at all, and the reopen rate."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Extract the failure pattern for a reported problem.

        :param args: 'text' plus optional 'product_area', 'issue_type', 'priority', 'neighbours'.
        :param sly_data: Out-of-band data shared across the agent hierarchy.
            Keys expected for this implementation: None
        :return: The aggregated failure pattern, or an error message.
        """
        text: str = args.get("text") or args.get("problem") or ""
        if not text.strip():
            return {"error": "No problem text provided. Cannot infer a failure pattern without a problem."}

        product_area: str = args.get("product_area") or ""
        issue_type: str = args.get("issue_type") or ""
        priority: str = args.get("priority") or ""

        try:
            neighbours = int(args.get("neighbours", _DEFAULT_NEIGHBOURS))
        except (TypeError, ValueError):
            neighbours = _DEFAULT_NEIGHBOURS
        neighbours = max(_MIN_NEIGHBOURS, min(neighbours, _MAX_NEIGHBOURS))

        payload = {
            "text": text,
            "product_area": product_area,
            "issue_type": issue_type,
            "priority": priority,
            "neighbours": neighbours,
        }
        return await call_with_fallback(
            "failure_pattern",
            payload,
            lambda: ticket_core.failure_pattern(
                text,
                product_area=product_area,
                issue_type=issue_type,
                priority=priority,
                neighbours=neighbours,
            ),
        )
