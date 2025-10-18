# ray_utilities/setup/test__experiment_uploader.py

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ray_utilities.setup._experiment_uploader import ExperimentUploader


class DummyArgs:
    def __init__(self, wandb=None, comet=None):
        self.wandb = wandb
        self.comet = comet


class DummyProcess:
    def __init__(self, args):
        self.args = args


@pytest.fixture
def dummy_uploader():
    class DummyUploader(ExperimentUploader):
        def __init__(self, args):
            super().__init__()
            self.args = args

    return DummyUploader


def test_upload_offline_experiments_wandb_failed(monkeypatch, dummy_uploader):
    """Test wandb upload with failed processes and no results (Tuner failed)."""
    args = DummyArgs(wandb="offline+upload")
    uploader = dummy_uploader(args)

    # Simulate wandb_upload_results returns a list of DummyProcess
    monkeypatch.setattr(
        uploader, "wandb_upload_results", lambda results, tuner, wait: [DummyProcess(["wandb", "sync", "/fake/path"])]
    )

    # Simulate _failure_aware_wait always returns non-zero (failure)
    monkeypatch.setattr(uploader, "_failure_aware_wait", lambda process, timeout, trial_id: 1)

    # Track calls to _update_failed_upload_file
    called = {}

    def fake_update_failed_upload_file(failed_processes, experiment_path, upload_to_trial):
        called["called"] = True
        called["experiment_path"] = experiment_path
        return "failed_file.txt"

    monkeypatch.setattr(uploader, "_update_failed_upload_file", fake_update_failed_upload_file)

    # Patch logger to capture error logs
    with patch.object(logging.getLogger("ray_utilities.setup._experiment_uploader"), "error") as mock_error:
        uploader.upload_offline_experiments(results=None, tuner=None)
        # Should log a warning about missing results and error about failed runs
        assert called.get("called") is True
        assert mock_error.call_count >= 1
        error_args = [call.args[0] for call in mock_error.call_args_list]
        assert any("Failed to upload the following wandb runs" in msg for msg in error_args)


def test_upload_offline_experiments_comet(monkeypatch, dummy_uploader):
    """Test comet upload with no results (Tuner failed)."""
    args = DummyArgs(comet="offline+upload")
    uploader = dummy_uploader(args)

    # Track calls to comet_upload_offline_experiments
    called = {}

    def fake_comet_upload():
        called["called"] = True

    monkeypatch.setattr(uploader, "comet_upload_offline_experiments", fake_comet_upload)

    # Patch logger to capture info logs
    with patch.object(logging.getLogger("ray_utilities.setup._experiment_uploader"), "info") as mock_info:
        uploader.upload_offline_experiments(results=None, tuner=None)
        assert called.get("called") is True
        info_args = [call.args[0] for call in mock_info.call_args_list]
        assert any("Uploading offline experiments to Comet" in msg for msg in info_args)


def test_upload_offline_experiments_wandb_and_comet(monkeypatch, dummy_uploader):
    """Test both wandb and comet upload requested, no results."""
    args = DummyArgs(wandb="offline+upload", comet="offline+upload")
    uploader = dummy_uploader(args)

    monkeypatch.setattr(
        uploader, "wandb_upload_results", lambda results, tuner, wait: [DummyProcess(["wandb", "sync", "/fake/path"])]
    )
    monkeypatch.setattr(uploader, "_failure_aware_wait", lambda process, timeout, trial_id: 1)
    monkeypatch.setattr(uploader, "_update_failed_upload_file", lambda *a, **kw: "failed_file.txt")
    monkeypatch.setattr(uploader, "comet_upload_offline_experiments", lambda: None)

    with (
        patch.object(logging.getLogger("ray_utilities.setup._experiment_uploader"), "error") as mock_error,
        patch.object(logging.getLogger("ray_utilities.setup._experiment_uploader"), "info") as mock_info,
    ):
        uploader.upload_offline_experiments(results=None, tuner=None)
        error_args = [call.args[0] for call in mock_error.call_args_list]
        info_args = [call.args[0] for call in mock_info.call_args_list]
        assert any("Failed to upload the following wandb runs" in msg for msg in error_args)
        assert any("Uploading offline experiments to Comet" in msg for msg in info_args)
