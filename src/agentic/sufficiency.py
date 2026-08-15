"""
Evidence sufficiency checker (v3.0).
LLM-driven evaluation of whether gathered evidence is adequate to answer.
"""

from openai import AsyncOpenAI

from src.config import LLMConfig


SUFFICIENCY_PROMPT = """You are a research quality evaluator. Given a user query and the evidence gathered
so far, determine whether the evidence is sufficient to provide a complete, accurate answer.

User query: {query}

Evidence gathered so far:
{evidence_summary}

Evaluate based on these criteria:
1. Coverage: Does the evidence address ALL aspects of the query?
2. Diversity: Are there at least 2 different sources?
3. Depth: Is there detailed information (not just snippets)?
4. Currency: Is the information recent enough for the query?
5. Conflicts: Are there any contradictions that need resolution?

Respond with ONLY a JSON object:
{{
    "sufficient": true/false,
    "confidence": 0.0-1.0,
    "gaps": ["gap 1", "gap 2"],  // what's missing if not sufficient
    "recommendation": "next steps if not sufficient"
}}"""


class SufficiencyChecker:
    """Checks if agent-gathered evidence is sufficient to answer the query."""

    def __init__(self, llm_config: LLMConfig = None):
        self.llm_config = llm_config
        self._client = None

    @property
    def client(self):
        if self._client is None and self.llm_config:
            self._client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )
        return self._client

    async def check(
        self, query: str, evidence_summary: str
    ) -> dict:
        """
        Check if evidence is sufficient.

        Returns:
            {
                "sufficient": bool,
                "confidence": float,
                "gaps": List[str],
                "recommendation": str,
            }
        """
        import json

        if not self.client:
            # Fallback heuristic: if evidence > 1000 chars, probably sufficient
            rough_sufficient = len(evidence_summary) > 800
            return {
                "sufficient": rough_sufficient,
                "confidence": 0.6 if rough_sufficient else 0.3,
                "gaps": [],
                "recommendation": "Insufficient evidence" if not rough_sufficient else "",
            }

        prompt = SUFFICIENCY_PROMPT.format(
            query=query,
            evidence_summary=evidence_summary[:4000],
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "{}"

            # Extract JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except Exception:
            # Fallback heuristic
            sufficient = len(evidence_summary) > 1000
            return {
                "sufficient": sufficient,
                "confidence": 0.5,
                "gaps": [],
                "recommendation": "",
            }


# Module-level singleton
_sufficiency_checker: SufficiencyChecker = None


def get_sufficiency_checker(llm_config: LLMConfig = None) -> SufficiencyChecker:
    global _sufficiency_checker
    if _sufficiency_checker is None:
        _sufficiency_checker = SufficiencyChecker(llm_config)
    return _sufficiency_checker
