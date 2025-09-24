# backend/agents/planner_agent.py
import logging
from typing import List, Dict, Any, Optional
import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger("PlannerAgent")


class PlannerAgent:
    """
    PlannerAgent generates candidate test cases for the game.
    Uses an LLM via LangChain to propose structured test scenarios.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        # --- Directly set your API key here ---
        self.api_key = "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"


        if not self.api_key:
            raise RuntimeError(
                "❌ OPENAI_API_KEY not found. Please set self.api_key in PlannerAgent.__init__"
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key,
        )

    async def generate(
        self,
        target_url: str,
        seeds: Optional[List[str]] = None,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate test cases.
        :param target_url: Game URL
        :param seeds: Optional seed ideas (user-provided)
        :param n: Number of candidates to generate
        """
        logger.info("PlannerAgent generating %d test cases for %s", n, target_url)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a QA test planner for online puzzle games. "
                    "Generate structured test case scenarios in JSON form.",
                ),
                (
                    "user",
                    f"Target game: {target_url}\n"
                    f"Generate {n} candidate test cases. "
                    f"Seeds (optional): {seeds or 'none'}\n"
                    "Each test case must include: id, description, steps (list).",
                ),
            ]
        )

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({})
            text = response.content

            candidates = json.loads(text)
            # Ensure it's a list of dicts
            if isinstance(candidates, dict):
                candidates = [candidates]
            return candidates

        except Exception as e:
            logger.error("PlannerAgent LLM failed: %s", e)
            raise RuntimeError(
                "PlannerAgent failed to generate test cases using LLM"
            ) from e
