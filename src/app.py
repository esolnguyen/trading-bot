"""Application entrypoint — routes to the LangGraph trading_bot runner.

Kept as the ``python -m src.app`` target so existing shell aliases still
work; delegates to :func:`src.trading_bot.runner.main`.
"""

from src.trading_bot.runner import main


def run() -> None:
    main()


if __name__ == "__main__":
    run()
