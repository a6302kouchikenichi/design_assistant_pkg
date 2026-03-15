from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # provider: 'openai' or 'azure'
    llm_provider: str = 'openai'

    # OpenAI
    openai_api_key: str | None = None

    # Azure OpenAI
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = '2024-10-21'
    # deployments (Azure uses deployment names, not model IDs)
    azure_chat_deployment: str = ''
    azure_embed_deployment: str = ''

    # Optional: Entra ID auth (recommended for production on Azure)
    azure_use_entra: bool = False

    # Default models for OpenAI
    openai_chat_model: str = 'gpt-4o-mini'
    openai_embed_model: str = 'text-embedding-3-large'


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv('LLM_PROVIDER', 'openai').lower(),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21'),
        azure_chat_deployment=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', ''),
        azure_embed_deployment=os.getenv('AZURE_OPENAI_EMBED_DEPLOYMENT', ''),
        azure_use_entra=os.getenv('AZURE_OPENAI_USE_ENTRA', '0') == '1',
        openai_chat_model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        openai_embed_model=os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-large'),
    )
