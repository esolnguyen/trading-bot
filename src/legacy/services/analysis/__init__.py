"""Technical and pattern analysis services.

Thin re-export shim: the compute primitives (IndicatorCalculator, PatternAnalyzer,
ChartGenerator, MultiTimeframeAnalyzer) now live under
``src.mcp_servers.shared.services`` so they can be consumed by both the trading
loop and the analysis MCP server. Everything in this package is trading-loop-
specific.
"""

from src.mcp_servers.shared.services import (
    ChartGenerator,
    IndicatorCalculator,
    MarketAggregator,
    MultiTimeframeAnalyzer,
    PatternAnalyzer,
    TechnicalAnalyzer,
)

from .context_builder import ContextBuilder
from .prompt_templates import build_system_prompt

__all__ = [
    "ChartGenerator",
    "ContextBuilder",
    "IndicatorCalculator",
    "MarketAggregator",
    "MultiTimeframeAnalyzer",
    "PatternAnalyzer",
    "TechnicalAnalyzer",
    "build_system_prompt",
]
