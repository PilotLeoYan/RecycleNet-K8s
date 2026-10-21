"""Helpers for MLflow tracking URI handling."""

from pathlib import Path

_SQLITE_PREFIX = "sqlite:///"


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Convert a relative SQLite tracking URI into an absolute one.

    Ray workers run in a different working directory than the driver, so a
    relative URI such as ``sqlite:///mlflow.db`` would point to a different
    (newly created) database and metrics would never reach the driver's run.

    Args:
        tracking_uri: MLflow tracking URI from configuration.

    Returns:
        URI with an absolute path for relative SQLite databases; other URIs
        are returned unchanged.
    """
    if tracking_uri.startswith(_SQLITE_PREFIX):
        db_path = tracking_uri[len(_SQLITE_PREFIX) :]
        if db_path and not db_path.startswith("/") and db_path != ":memory:":
            return f"{_SQLITE_PREFIX}{Path(db_path).resolve()}"
    return tracking_uri
