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
"""Incident chain stage 4 — verify a slice against SLA and incident trends.

Referenced in ``registries/duckdebug.hocon`` as
``"class": "metric_agent.MetricAgent"``.
"""

from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.duckdebug import ticket_core
from coded_tools.duckdebug.duck_mcp_client import call_with_fallback

_DEFAULT_RECENT_MONTHS = 6
_MAX_RECENT_MONTHS = 24


class MetricAgent(CodedTool):
    """Measures a ticket slice against SLA targets and volume trends, always beside
    the corpus-wide baseline for the same measure — a breach rate means nothing
    without the number it is being compared to."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Compute SLA and trend metrics for a slice of the ticket corpus.

        :param args: Optional 'product_area', 'issue_type', 'priority', 'recent_months'.
            With no filters, reports the entire corpus.
        :param sly_data: Out-of-band data shared across the agent hierarchy.
            Keys expected for this implementation: None
        :return: SLA breach, reopen rate, CSAT, and volume trend vs baseline.
        """
        product_area: str = args.get("product_area") or ""
        issue_type: str = args.get("issue_type") or ""
        priority: str = args.get("priority") or ""

        try:
            recent_months = int(args.get("recent_months", _DEFAULT_RECENT_MONTHS))
        except (TypeError, ValueError):
            recent_months = _DEFAULT_RECENT_MONTHS
        recent_months = max(1, min(recent_months, _MAX_RECENT_MONTHS))

        payload = {
            "product_area": product_area,
            "issue_type": issue_type,
            "priority": priority,
            "recent_months": recent_months,
        }
        return await call_with_fallback(
            "sla_metrics",
            payload,
            lambda: ticket_core.sla_metrics(
                product_area=product_area,
                issue_type=issue_type,
                priority=priority,
                recent_months=recent_months,
            ),
        )
