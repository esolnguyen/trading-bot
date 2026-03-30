"""Token counting and cost tracking for AI model usage."""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from src.domain.analysis.stats import ProviderCostStats, SessionCosts, TokenUsageStats

try:
    import tiktoken as _tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _tiktoken = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False


class ModelPricing:
    """Loads and provides model pricing from a JSON config file."""

    _instance: Optional["ModelPricing"] = None
    _pricing: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelPricing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if ModelPricing._pricing is None:
            ModelPricing._pricing = self._load_pricing()

    def _load_pricing(self) -> Dict[str, Any]:
        # Try several relative paths
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_pricing.json"),
            os.path.join("config", "model_pricing.json"),
        ]
        for path in candidates:
            path = os.path.normpath(path)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass
        return {"google": {}, "openrouter": {}}

    def get_cost(self, provider: str, model: str, input_tokens: int,
                 output_tokens: int) -> Optional[float]:
        assert ModelPricing._pricing is not None
        provider_pricing = ModelPricing._pricing.get(provider, {})
        model_key = self._normalize_model_key(model)
        model_pricing = provider_pricing.get(model_key)
        if not model_pricing:
            for key, val in provider_pricing.items():
                if key.startswith("_"):
                    continue
                if model_key in key or key in model_key:
                    model_pricing = val
                    break
        if not model_pricing:
            return None
        input_cost = (input_tokens / 1_000_000) * model_pricing.get("input_per_million", 0)
        output_cost = (output_tokens / 1_000_000) * model_pricing.get("output_per_million", 0)
        return input_cost + output_cost

    @staticmethod
    def _normalize_model_key(model: str) -> str:
        return model.lower().replace("models/", "")


class TokenCounter:
    """Token counting and session cost tracking for AI model calls."""

    PROVIDER_COST_MESSAGES = {
        "openrouter": "Cost: ${cost:.4f}",
        "google": "Cost: ${cost:.4f} (estimated)",
        "lmstudio": "Free - local model",
        "azure": "Cost: unknown (Azure)",
    }

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if _HAS_TIKTOKEN:
            self.tokenizer = _tiktoken.get_encoding(encoding_name)
        else:
            self.tokenizer = None
        self.session_tokens: Dict[str, int] = {
            "prompt": 0, "completion": 0, "system": 0, "total": 0,
        }
        self.session_costs = SessionCosts()
        self.request_usage: Optional[TokenUsageStats] = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        # Rough fallback: ~4 chars per token
        return max(1, len(text) // 4)

    def track_prompt_tokens(self, text: str, message_type: str = "prompt") -> int:
        token_count = self.count_tokens(text)
        if message_type in self.session_tokens:
            self.session_tokens[message_type] += token_count
        else:
            self.session_tokens[message_type] = token_count
        self.session_tokens["total"] += token_count
        return token_count

    def record_api_usage(self, provider: str, prompt_tokens: int,
                         completion_tokens: int, cost: Optional[float] = None) -> None:
        self.request_usage = TokenUsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
        )
        self.session_tokens["prompt"] += prompt_tokens
        self.session_tokens["completion"] += completion_tokens
        self.session_tokens["total"] += prompt_tokens + completion_tokens
        if cost:
            if provider == "openrouter":
                self.session_costs.openrouter += cost
            elif provider == "google":
                self.session_costs.google += cost
            elif provider == "lmstudio":
                self.session_costs.lmstudio += cost

    @staticmethod
    def format_cost(cost: float) -> str:
        if cost == 0 or cost is None:
            return "Free"
        if cost < 0.0001:
            return f"${cost:.8f} ({cost * 100:.6f}¢)"
        if cost < 0.01:
            return f"${cost:.6f} ({cost * 100:.4f}¢)"
        if cost < 1:
            return f"${cost:.4f}"
        return f"${cost:.2f}"

    def process_response_usage(self, usage: Optional[Dict[str, Any]], provider: str = "unknown",
                                logger: Any = None, fallback_text: Optional[str] = None) -> None:
        if usage:
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
            cost = usage.get("cost")
            self.record_api_usage(provider, pt, ct, cost)
            if logger:
                logger.info("Tokens: %s in / %s out (total %s)", f"{pt:,}", f"{ct:,}", f"{pt+ct:,}")
                if cost is not None:
                    logger.info("Request cost: %s", self.format_cost(cost))
        elif fallback_text:
            rt = self.track_prompt_tokens(fallback_text, "completion")
            if logger:
                logger.info("Response tokens (estimated): %s", rt)

    def record_cost(self, provider: str, cost: float) -> None:
        if provider == "openrouter":
            self.session_costs.openrouter += cost
        elif provider == "google":
            self.session_costs.google += cost
        elif provider == "lmstudio":
            self.session_costs.lmstudio += cost
        if self.request_usage:
            self.request_usage.cost = cost

    def get_last_request_usage(self) -> Optional[TokenUsageStats]:
        return self.request_usage

    def get_usage_stats(self) -> Dict[str, int]:
        return self.session_tokens.copy()

    def get_session_costs(self) -> SessionCosts:
        return self.session_costs

    def get_total_session_cost(self) -> float:
        return self.session_costs.total

    def format_usage_display(self, provider: str, prompt_tokens: int,
                              completion_tokens: int, cost: Optional[float] = None) -> str:
        token_str = f"Tokens: {prompt_tokens:,} in / {completion_tokens:,} out"
        if cost is not None and provider in ("openrouter", "google"):
            cost_str = self.PROVIDER_COST_MESSAGES[provider].format(cost=cost)
        elif provider in self.PROVIDER_COST_MESSAGES:
            cost_str = self.PROVIDER_COST_MESSAGES[provider]
        else:
            cost_str = "Cost: unknown provider"
        return f"{token_str} │ {cost_str}"

    def reset_session_stats(self) -> None:
        self.session_tokens = {"prompt": 0, "completion": 0, "system": 0, "total": 0}
        self.session_costs = SessionCosts()
        self.request_usage = None

    def reset_costs_only(self) -> None:
        self.session_costs = SessionCosts()


class CostStorage:
    """Persistent storage for cumulative API costs saved to a JSON file."""

    PROVIDERS = ("openrouter", "google", "lmstudio")

    def __init__(self, file_path: str = "data/api_costs.json") -> None:
        self.file_path = file_path
        self._ensure_directory()
        self._providers: Dict[str, ProviderCostStats] = {}
        self._last_reset: Optional[str] = None
        self._lock = threading.RLock()
        self._last_save_time = 0.0
        self._dirty = False
        atexit.register(self.save)
        self._load_or_create()

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_or_create(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._last_reset = data.get("last_reset")
                for provider in self.PROVIDERS:
                    if provider in data:
                        p = data[provider]
                        self._providers[provider] = ProviderCostStats(
                            total_cost=p.get("total_cost", 0.0),
                            total_input_tokens=p.get("total_input_tokens", 0),
                            total_output_tokens=p.get("total_output_tokens", 0),
                        )
                    else:
                        self._providers[provider] = ProviderCostStats()
                return
            except (json.JSONDecodeError, OSError):
                pass
        self._init_defaults()

    def _init_defaults(self) -> None:
        for provider in self.PROVIDERS:
            self._providers[provider] = ProviderCostStats()

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            data: Dict[str, Any] = {"last_reset": self._last_reset}
            for provider, stats in self._providers.items():
                data[provider] = stats.to_dict()
            temp_path = None
            try:
                directory = os.path.dirname(self.file_path)
                with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8",
                                                  dir=directory or ".") as f:
                    temp_path = f.name
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.file_path)
                self._last_save_time = time.time()
                self._dirty = False
            except Exception as exc:
                print(f"Error saving cost storage: {exc}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    def record_usage(self, provider: str, prompt_tokens: int,
                     completion_tokens: int, cost: Optional[float] = None) -> None:
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderCostStats()
            stats = self._providers[provider]
            stats.total_input_tokens += prompt_tokens
            stats.total_output_tokens += completion_tokens
            if cost is not None:
                stats.total_cost += cost
            self._dirty = True
            if time.time() - self._last_save_time >= 5.0:
                self.save()

    def get_costs(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"last_reset": self._last_reset}
        for provider, stats in self._providers.items():
            result[provider] = stats.to_dict()
        return result

    def get_provider_costs(self, provider: str) -> ProviderCostStats:
        return self._providers.get(provider, ProviderCostStats())

    def get_total_openrouter_cost(self) -> float:
        return self._providers.get("openrouter", ProviderCostStats()).total_cost

    def reset(self) -> None:
        from datetime import datetime, timezone
        self._init_defaults()
        self._last_reset = datetime.now(timezone.utc).isoformat()
        self.save()


__all__ = ["ModelPricing", "TokenCounter", "CostStorage"]
