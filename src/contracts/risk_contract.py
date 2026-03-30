"""Protocol definition for RiskManager interface."""

from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.trading.models import RiskAssessment


class RiskManagerProtocol(Protocol):
    """Protocol defining the interface for Risk Management."""

    def calculate_entry_parameters(
        self,
        signal: str,
        current_price: float,
        capital: float,
        confidence: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: Optional[float] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> "RiskAssessment":
        """Calculate all risk parameters for a new position entry."""
        ...

    def validate_signal(self, signal: str) -> bool:
        """Validate if a signal is actionable."""
        ...


__all__ = ["RiskManagerProtocol"]
