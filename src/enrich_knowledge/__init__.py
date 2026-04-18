"""Write-path package: Chroma ingestion + ML training.

Public surface lives in ``config`` (sub-settings) and ``runners`` (CLI
entrypoints). Library code is organised as ``rag_ingestion`` /
``ml_training`` with the three-layer shape (sources / transforms /
writers) documented in ``plan.md``.
"""
