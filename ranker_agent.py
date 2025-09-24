# backend/agents/ranker_agent.py
import logging
from typing import List, Dict, Any
import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger("RankerAgent")


class RankerAgent:
    """
    RankerAgent evaluates candidate test cases and selects the top_k most useful.
    Uses an LLM via LangChain.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        # --- Directly set your API key here ---
        self.api_key = "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"


        if not self.api_key:
            raise RuntimeError(
                "❌ OPENAI_API_KEY not found. Please set self.api_key in RankerAgent.__init__"
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key,
        )

    async def rank_and_select(
        self, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank the given test candidates and return the top_k.
        """
        logger.info(
            "RankerAgent ranking %d candidates; selecting top %d", len(candidates), top_k
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a QA strategist. Rank the given test cases by importance, "
                    "coverage, and ability to find bugs in puzzle games.",
                ),
                (
                    "user",
                    "Here are the candidate test cases:\n"
                    f"{candidates}\n\n"
                    f"Select the top {top_k} most promising ones and return only them as JSON list.",
                ),
            ]
        )

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({})
            text = response.content

            top_candidates = json.loads(text)
            if isinstance(top_candidates, dict):
                top_candidates = [top_candidates]
            return top_candidates
        except Exception as e:
            logger.error("❌ RankerAgent LLM failed: %s", e)
            raise RuntimeError(
                "RankerAgent failed to rank candidates using LLM"
            ) from e
