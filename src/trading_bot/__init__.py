"""LangChain/LangGraph-driven trading loop.

One-shot decision cycles orchestrated by a ReAct agent whose tools are
the project's MCP servers (ml, binance, analysis, rag, skills). Order
placement is deferred until the Phase 7 execution MCP lands; until
then the loop only emits structured decisions and logs them.
"""
