import os
import urllib.error
import urllib.request
from pathlib import Path

from langchain_openai import ChatOpenAI


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable `{name}` is required.")
    return value


class AgentConfig:
    def __init__(
        self,
        model_name,
        base_url,
        api_key,
        review_model_name=None,
        temperature=0.1,
        max_tokens=4000,
        random_state=42,
        output_root=Path("artifacts"),
        keep_run_history=True,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.review_model_name = review_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.random_state = random_state
        self.output_root = output_root
        self.keep_run_history = keep_run_history

    @classmethod
    def from_runtime(cls, output_root=None):
        return cls(
            model_name=required_env("LLM_MODEL"),
            base_url=required_env("LLM_BASE_URL"),
            api_key=required_env("LLM_API_KEY"),
            review_model_name=os.getenv("LLM_REVIEW_MODEL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            output_root=Path(output_root or os.getenv("SMADIMO_OUTPUT_ROOT", "artifacts")),
        )

    def build_primary_model(self):
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream_usage=False,
        )

    def build_review_model(self):
        model_name = self.review_model_name or self.model_name
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=self.max_tokens,
            stream_usage=False,
        )


def ensure_llm_endpoint(config):
    models_url = config.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {config.api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"LLM endpoint returned HTTP {response.status}.")
    except urllib.error.URLError as error:
        raise RuntimeError(
            "LLM endpoint is unavailable. Verify that the configured "
            "OpenAI-compatible API is running and reachable at "
            f"{config.base_url}."
        ) from error
