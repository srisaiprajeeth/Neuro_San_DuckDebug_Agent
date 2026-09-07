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
"""MAP step 2 of the DuckDebug network: retrieve real questions framed like this one.

Resolver convention: this module lives in ``coded_tools/duckdebug/`` (matching
``registries/duckdebug.hocon``), so the network references it as
``"class": "echo_retriever.EchoRetriever"``.
"""

from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.duckdebug import duck_core
from coded_tools.duckdebug.duck_mcp_client import call_with_fallback

# Keep the retrieved set small enough that the Synthesizer can quote it verbatim.
_DEFAULT_K = 3
_MAX_K = 10


class EchoRetriever(CodedTool):
    """Retrieves real Stack Overflow questions framed like the user's problem, each
    with its quality label (HQ/LQ_CLOSE/LQ_EDIT) and similarity score.

    Delegates to the DuckDebug MCP server when it is running, and to
    :mod:`duck_core` in-process otherwise.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Find similar questions in the labeled corpus.

        :param args: Dictionary containing 'text', plus optional 'tag' and 'k'.
        :param sly_data: Out-of-band data shared across the agent hierarchy.
            Keys expected for this implementation: None
        :return: List of similar questions, or an error message.
        """
        text: str = args.get("text") or args.get("problem") or ""
        tag: str = args.get("tag") or ""

        if not text.strip():
            return {"error": "No problem text provided. Ask the user to describe what they are stuck on."}

        # The LLM may hand back a string here, so coerce before clamping.
        try:
            k = int(args.get("k", _DEFAULT_K))
        except (TypeError, ValueError):
            k = _DEFAULT_K
        k = max(1, min(k, _MAX_K))

        return await call_with_fallback(
            "find_similar",
            {"text": text, "tag": tag, "k": k},
            lambda: duck_core.find_similar(text, tag=tag, k=k),
            expect_list=True,
        )
