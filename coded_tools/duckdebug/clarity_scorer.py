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
"""MAP step 1 of the DuckDebug network: score how clearly a problem is stated.

Resolver convention: this module lives in ``coded_tools/duckdebug/`` (matching
``registries/duckdebug.hocon``), so the network references it as
``"class": "clarity_scorer.ClarityScorer"``.
"""

from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.duckdebug import duck_core
from coded_tools.duckdebug.duck_mcp_client import call_with_fallback


class ClarityScorer(CodedTool):
    """Scores how clearly a stuck problem is articulated (0-100), grounded in the
    60k Stack Overflow HQ/LQ dataset, and lists what is missing.

    Delegates to the DuckDebug MCP server when it is running, and to
    :mod:`duck_core` in-process otherwise.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score the caller's problem statement.

        :param args: Dictionary containing 'text' (the problem statement).
        :param sly_data: Out-of-band data shared across the agent hierarchy.
            Keys expected for this implementation: None
        :return: Clarity score, band, missing pieces, and the ensemble vote breakdown.
        """
        text: str = args.get("text") or args.get("problem") or ""

        if not text.strip():
            return {"error": "No problem text provided. Ask the user to describe what they are stuck on."}

        return await call_with_fallback(
            "score_clarity",
            {"text": text},
            lambda: duck_core.score_clarity(text),
        )
