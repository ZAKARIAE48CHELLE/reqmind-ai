"""Tests for ReqMind AI MLflow tracking module."""

import pytest
from pathlib import Path
from reqmind.tracking import setup_tracking, track_run, log_run_params, log_run_metrics, is_mlflow_available


def test_is_mlflow_available():
    """Verify MLflow availability check."""
    # When mlflow is installed, this should return True
    assert isinstance(is_mlflow_available(), bool)


def test_track_run_context_manager(tmp_path: Path):
    """Verify track_run context manager creates runs cleanly."""
    if not is_mlflow_available():
        pytest.skip("mlflow not installed")

    custom_uri = (tmp_path / "test_mlruns").as_uri()
    setup_tracking(experiment_name="test-experiment", tracking_uri=custom_uri)

    with track_run("test-run", experiment_name="test-experiment", tags={"env": "pytest"}) as run:
        assert run is not None
        log_run_params({"param_a": 10, "model": "test_clf"})
        log_run_metrics({"val_accuracy": 0.85, "val_macro_f1": 0.82})
