"""Ingestion-pipeline settings.

Owns every knob that shapes *what* gets written to Chroma: API keys for
upstream sources, article limits, density/co-occurrence scoring
multipliers. Scheduling intervals live in ``schedule.py`` (code, not
env) so they can be versioned alongside the pipeline.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Source API keys
    cryptocompare_api_key: str = Field(default="", alias="CRYPTOCOMPARE_API_KEY")
    coingecko_api_key: str = Field(default="", alias="COINGECKO_API_KEY")

    # Retrieval / scoring knobs used by the transforms layer.
    rag_news_limit: int = Field(default=5, alias="RAG_NEWS_LIMIT")
    rag_article_max_tokens: int = Field(default=256, alias="RAG_ARTICLE_MAX_TOKENS")
    rag_density_penalty_threshold: int = Field(
        default=300, alias="RAG_DENSITY_PENALTY_THRESHOLD"
    )
    rag_density_boost_threshold: int = Field(
        default=1000, alias="RAG_DENSITY_BOOST_THRESHOLD"
    )
    rag_density_penalty_multiplier: float = Field(
        default=0.5, alias="RAG_DENSITY_PENALTY_MULTIPLIER"
    )
    rag_density_boost_multiplier: float = Field(
        default=1.2, alias="RAG_DENSITY_BOOST_MULTIPLIER"
    )
    rag_cooccurrence_multiplier: float = Field(
        default=1.5, alias="RAG_COOCCURRENCE_MULTIPLIER"
    )

    sentiment_scoring_enabled: bool = Field(
        default=False, alias="SENTIMENT_SCORING_ENABLED"
    )
