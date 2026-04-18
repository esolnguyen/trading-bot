"""Cross-cutting storage paths shared by serve / enrich / trading-bot."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """On-disk paths shared across processes.

    - ``chroma_path``: vector DB location (enrich_knowledge writes, rag MCP
      reads). Must match across processes or the reader sees an empty DB.
    - ``data_dir``: root for OHLCV CSVs, model artifacts, persistence.
    - ``log_dir``: where process logs are written.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    def ohlcv_csv_path(self, symbol: str, interval: str) -> str:
        """Canonical OHLCV CSV path — e.g. ``data/ohlcv/btcusdt_4h.csv``."""
        sym = symbol.replace("/", "").lower()
        return f"{self.data_dir}/ohlcv/{sym}_{interval}.csv"
