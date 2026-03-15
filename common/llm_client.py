from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from .settings import Settings


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    """Small abstraction over OpenAI API and Azure OpenAI.

    - For OpenAI: model is a model ID like 'gpt-4o-mini'
    - For Azure: model is a deployment name (e.g. 'gpt-4o-mini-prod')
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = self._build_client()

    def _build_client(self):
        provider = self.settings.llm_provider
        if provider == 'azure':
            # AzureOpenAI client from openai python SDK
            from openai import AzureOpenAI
            if self.settings.azure_use_entra:
                # Keyless auth (Entra ID)
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                return AzureOpenAI(
                    azure_endpoint=self.settings.azure_endpoint,
                    api_version=self.settings.azure_api_version,
                    azure_ad_token_provider=token_provider,
                )
            else:
                return AzureOpenAI(
                    azure_endpoint=self.settings.azure_endpoint,
                    api_version=self.settings.azure_api_version,
                    api_key=self.settings.azure_api_key,
                )

        # default: OpenAI
        from openai import OpenAI
        return OpenAI(api_key=self.settings.openai_api_key)

    def chat(self, messages: List[ChatMessage | dict[str, str]], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 1200) -> str:
        provider = self.settings.llm_provider
        if provider == 'azure':
            use_model = model or self.settings.azure_chat_deployment
        else:
            use_model = model or self.settings.openai_chat_model

        normalized_messages = []
        for m in messages:
            if isinstance(m, dict):
                normalized_messages.append({
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                })
            else:
                normalized_messages.append({"role": m.role, "content": m.content})

        resp = self._client.chat.completions.create(
            model=use_model,
            messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def embed(self, texts: Iterable[str], model: Optional[str] = None) -> list[list[float]]:
        provider = self.settings.llm_provider
        if provider == 'azure':
            use_model = model or self.settings.azure_embed_deployment
        else:
            use_model = model or self.settings.openai_embed_model

        # embeddings.create accepts input list
        resp = self._client.embeddings.create(
            model=use_model,
            input=list(texts),
        )
        return [d.embedding for d in resp.data]
