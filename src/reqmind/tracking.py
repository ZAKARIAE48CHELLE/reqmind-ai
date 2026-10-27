"""MLflow experiment tracking and MLOps utility for ReqMind AI.

Supports local file-based and SQLite tracking with automatic experiment setup,
hyperparameter logging, metric recording, model artifact storage, and figure logging.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# Enable file store compatibility for MLflow 3+
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "reqmind-classification"
DEFAULT_DB_PATH = ROOT_DIR / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def is_mlflow_available() -> bool:
    """Check if mlflow package is installed."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def setup_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
) -> str:
    """Configure MLflow tracking URI and activate target experiment.
    
    Defaults to local SQLite database at `sqlite:///mlflow.db`.
    """
    if not is_mlflow_available():
        logger.warning("MLflow is not installed; tracking is disabled.")
        return ""

    import mlflow

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            tags={"project": "ReqMind-AI", "domain": "TALN-Requirements-Engineering"},
        )
    else:
        exp_id = experiment.experiment_id
        if getattr(experiment, "lifecycle_stage", None) == "deleted":
            from mlflow.tracking import MlflowClient
            MlflowClient().restore_experiment(exp_id)
            logger.info(f"Restored deleted MLflow experiment '{experiment_name}' (ID: {exp_id})")

    mlflow.set_experiment(experiment_name)
    return exp_id


@contextmanager
def track_run(
    run_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tags: Optional[Dict[str, Any]] = None,
    nested: bool = False,
) -> Iterator[Optional[Any]]:
    """Context manager for an MLflow run. Safely falls back if MLflow is unavailable."""
    if not is_mlflow_available():
        yield None
        return

    import mlflow

    setup_tracking(experiment_name=experiment_name)
    all_tags = {"framework": "ReqMind-AI"}
    if tags:
        all_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=all_tags, nested=nested) as run:
        yield run


def log_run_params(params: Dict[str, Any]) -> None:
    """Log parameters to current active MLflow run."""
    if not is_mlflow_available():
        return
    import mlflow

    # Flatten nested dictionaries or convert complex types to string
    flat = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}
    mlflow.log_params(flat)


def log_run_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log scalar metrics to current active MLflow run."""
    if not is_mlflow_available():
        return
    import mlflow

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v), step=step)


def log_run_artifact(file_path: Path | str, artifact_path: Optional[str] = None) -> None:
    """Log an artifact file (model, plot, CSV, report) to active MLflow run."""
    if not is_mlflow_available():
        return
    import mlflow

    p = Path(file_path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_sklearn_model(model: Any, artifact_path: str = "model") -> None:
    """Log scikit-learn model or pipeline to active MLflow run."""
    if not is_mlflow_available():
        return
    import mlflow.sklearn

    try:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception as exc:
        logger.warning(f"Failed to log model via mlflow.sklearn: {exc}")
