"""Model training drivers.

Each module exposes a ``train(settings) -> TrainResult`` entrypoint.
Model classes themselves live under ``src/mcp_servers/ml/services/``
(one class, two callsites: ``.fit()`` here, ``.predict()`` in the
MCP handler). Training writes artifacts to
``<data_dir>/models/<name>_<ts>.pkl`` and flips a ``current`` symlink
after validation — never overwrites in place.
"""
