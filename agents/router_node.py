"""
Router Node — decides: does this query need retrieval, or can the LLM answer directly?

Routes to "direct" for:  greetings, general knowledge, math, LLM-native tasks
Routes to "rag" for:     anything needing document knowledge

This is a lightweight LLM call with structured JSON output.
"""
import json
import logging

from core.interfaces import LLMClientBase
from agents.state import AgentState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query router for a RAG system.
Decide whether a user query requires document retrieval or can be answered directly from general knowledge.

Respond with ONLY valid JSON — no explanation, no markdown:
{
  "route": "rag" | "direct",
  "reasoning": "one sentence explaining your decision"
}

Route to "rag" when:
- The query asks about specific documents, policies, reports, or proprietary content
- The query requires up-to-date or domain-specific facts not in general knowledge
- The query mentions a specific product, person, or system that may be in the knowledge base

Route to "direct" when:
- The query is a general knowledge question (history, science, math, coding)
- The query is a greeting, clarification, or meta question about the assistant
- The query can be answered accurately without any documents"""


def router_node(state: AgentState, llm: LLMClientBase) -> AgentState:
    """
    LangGraph node: Router.
    Reads:  state["query"]
    Writes: state["route"], state["router_reasoning"]
    """
    query = state["query"]
    logger.info(f"[Router] Routing query: '{query[:80]}'")

    user_prompt = f'Query: "{query}"\n\nRespond with JSON only.'

    try:
        raw = llm.generate(system_prompt=ROUTER_SYSTEM_PROMPT, user_prompt=user_prompt)

        # Strip markdown fences if model wraps in ```json
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        decision = json.loads(clean)

        route = decision.get("route", "rag")
        reasoning = decision.get("reasoning", "")

        # Fallback safety: if JSON is malformed or route is unexpected, default to RAG
        if route not in ("rag", "direct"):
            route = "rag"
            reasoning = "Defaulting to RAG due to unexpected router output."

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[Router] JSON parse failed ({e}), defaulting to 'rag'")
        route = "rag"
        reasoning = "Router fallback — defaulting to retrieval."

    logger.info(f"[Router] → {route} | {reasoning}")

    return {
        **state,
        "route": route,
        "router_reasoning": reasoning,
    }
