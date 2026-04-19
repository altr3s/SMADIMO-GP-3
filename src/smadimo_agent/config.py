from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI


@dataclass
class AgentConfig:
    model_name: str = "qwen/qwen3.5-9b"
    review_model_name: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    random_state: int = 42
    output_root: Path = Path("artifacts")
    keep_run_history: bool = True
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"

    @classmethod
    def from_runtime(
        cls,
        model_name: Optional[str] = None,
        review_model_name: Optional[str] = None,
        output_root: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> "AgentConfig":
        return cls(
            model_name=model_name or os.getenv("LM_STUDIO_MODEL", "qwen/qwen3.5-9b"),
            review_model_name=review_model_name or os.getenv("LM_STUDIO_REVIEW_MODEL"),
            temperature=temperature
            if temperature is not None
            else float(os.getenv("LM_STUDIO_TEMPERATURE", "0.1")),
            output_root=Path(output_root or os.getenv("SMADIMO_OUTPUT_ROOT", "artifacts")),
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        )

    def build_primary_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream_usage=False,
        )

    def build_review_model(self) -> ChatOpenAI:
        model_name = self.review_model_name or self.model_name
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.0,
            max_tokens=self.max_tokens,
            stream_usage=False,
        )


def ensure_lm_studio_server(config: AgentConfig) -> None:
    models_url = config.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {config.api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"LM Studio returned HTTP {response.status}.")
    except urllib.error.URLError as error:
        raise RuntimeError(
            "LM Studio local server is unavailable. Start the LM Studio server, "
            "load the model, and verify the OpenAI-compatible API is running at "
            f"{config.base_url}."
        ) from error
