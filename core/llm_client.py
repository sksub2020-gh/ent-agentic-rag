"""
Universal OpenAI-compatible LLM client.
Provider is determined entirely by config — no subclassing needed.

Switching providers = .env change only, zero code changes:
  ollama  → LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama
  openai  → LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...
  groq    → LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_API_KEY=gsk_...
  together→ LLM_BASE_URL=https://api.together.xyz/v1  LLM_API_KEY=...
"""
import logging
from openai import OpenAI

from core.interfaces import LLMClientBase
from config.settings import config

logger = logging.getLogger(__name__)


class LLMClient(LLMClientBase):
    """
    Single universal client for any OpenAI-compatible provider.
    The `provider` config field is a label for logging/tracing only —
    all providers speak the same OpenAI API contract.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            project=config.project_name,
        )
        self.model = config.llm.model
        logger.info(f"LLMClient ready → provider={config.llm.provider} model={self.model}")


    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Standard chat completion — identical across all providers."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed (provider={config.llm.provider}): {e}")
            raise

    def health_check(self) -> bool:
        """
        Verify the provider is reachable and the configured model is available.

        Ollama: models.list() returns locally pulled models — we verify the
                model is present, since a missing pull is a common mistake.
        Cloud providers (OpenAI, Groq, etc.): models.list() confirms the
                endpoint is reachable. We skip the model-presence check because
                cloud providers don't always return every model in their list
                (e.g. fine-tuned models, preview models). A failed generate()
                call will surface model errors clearly enough.
        """
        try:
            models = self.client.models.list()

            if config.llm.provider == "ollama":
                available = [m.id for m in models.data]
                if self.model not in available:
                    logger.warning(
                        f"Model '{self.model}' not pulled locally.\n"
                        f"Available: {available}\n"
                        f"Run: ollama pull {self.model}"
                    )
                    return False

            logger.info(f"Health check OK — {config.llm.provider} | model={self.model}")
            return True
        except Exception as e:
            logger.error(
                f"Health check failed [{config.llm.provider}]: {e}\n"
                f"Check LLM_BASE_URL and LLM_API_KEY in your .env"
            )
            return False

