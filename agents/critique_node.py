"""
Critique Node — CRAG-style grounding check.

Validates that the generated answer is actually supported by retrieved context.
If not grounded → triggers a retry via conditional edge (up to MAX_RETRIES).
If grounded    → passes answer through to the final output.

This is the key enterprise quality gate — it catches hallucinations before they reach users.
"""
import json
import logging

from core.interfaces import LLMClientBase
from agents.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # Maximum retrieval retries before giving up

CRITIQUE_SYSTEM_PROMPT = """You are a strict factual grounding checker for a RAG system.

Given a question, a retrieved context, and a generated answer, determine if the answer
is fully supported by the context.

Respond with ONLY valid JSON:
{
  "grounded": true | false,
  "reasoning": "one sentence explaining your verdict",
  "issues": ["list any specific unsupported claims, or empty list if grounded"]
}

Mark as grounded=true ONLY if every factual claim in the answer can be traced to the context.
Mark as grounded=false if:
- The answer contains facts not present in the context
- The answer says "INSUFFICIENT_CONTEXT" (retrieval failed)
- The answer contradicts the context
- The answer is vague or evasive about something the context clearly covers"""


def critique_node(state: AgentState, llm: LLMClientBase) -> AgentState:
    """
    LangGraph node: Critique.
    Reads:  state["query"], state["context"], state["answer"], state["retry_count"]
    Writes: state["grounded"], state["critique_reasoning"], state["retry_count"]

    The conditional edge after this node checks state["grounded"]:
      - True  → END
      - False → back to rag_node (if retries remain) or END with disclaimer
    """
    query = state["query"]
    answer = state.get("answer", "")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)

    # Skip critique for direct (non-RAG) answers
    if state.get("route") == "direct" or state.get("grounded") is True:
        logger.info("[Critique] Skipping — direct answer path")
        return {**state, "grounded": True, "critique_reasoning": "Direct answer — no critique needed."}

    # Skip critique if no context was retrieved
    if not context:
        logger.info("[Critique] No context — marking as not grounded")
        return {
            **state,
            "grounded": False,
            "critique_reasoning": "No context was retrieved.",
            "retry_count": retry_count + 1,
        }

    logger.info(f"[Critique] Checking grounding (attempt {retry_count + 1}/{MAX_RETRIES + 1})")

    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context[:3000]}\n\n"   # Truncate context to avoid token overflow
        f"Answer: {answer}\n\n"
        f"Respond with JSON only."
    )

    grounded = False
    reasoning = ""
    issues = []

    try:
        raw = llm.generate(system_prompt=CRITIQUE_SYSTEM_PROMPT, user_prompt=user_prompt)
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)

        grounded = bool(result.get("grounded", False))
        reasoning = result.get("reasoning", "")
        issues = result.get("issues", [])

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[Critique] JSON parse failed ({e}) — assuming not grounded")
        grounded = False
        reasoning = "Critique parse error — treating as not grounded."

    if grounded:
        logger.info(f"[Critique] ✅ Grounded — {reasoning}")
    else:
        logger.warning(f"[Critique] ❌ Not grounded — {reasoning}")
        if issues:
            logger.warning(f"[Critique] Issues: {issues}")

    # If not grounded and retries remain, increment retry counter
    # The conditional edge in the graph will route back to rag_node
    new_retry_count = retry_count + 1 if not grounded else retry_count

    # If we've hit max retries, append a disclaimer to the answer
    if not grounded and new_retry_count > MAX_RETRIES:
        logger.warning(f"[Critique] Max retries ({MAX_RETRIES}) reached — appending disclaimer")
        answer_with_disclaimer = (
            f"{answer}\n\n"
            f"⚠️ Note: This answer may not be fully supported by the available documents. "
            f"Please verify with the original sources."
        )
        return {
            **state,
            "answer": answer_with_disclaimer,
            "grounded": True,           # Force exit from retry loop
            "critique_reasoning": f"Max retries reached. Last issue: {reasoning}",
            "retry_count": new_retry_count,
        }

    return {
        **state,
        "grounded": grounded,
        "critique_reasoning": reasoning,
        "retry_count": new_retry_count,
    }


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function for LangGraph.
    Called after critique_node to decide next step.

    Returns:
      "retry"  → back to rag_node
      "end"    → graph terminates
    """
    if state.get("grounded", False):
        return "end"
    if state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "end"
