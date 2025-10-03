from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ray.tune.logger import LoggerCallback

from ray_utilities.postprocessing import save_videos

# ruff: noqa: ARG002
# pyright: reportArgumentType=false


if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial


class SaveVideoFirstCallback(LoggerCallback):
    def on_trial_result(
        self,
        iteration: int,
        trials: list["Trial"],
        trial: "Trial",
        result: dict[str, Any],
        **info,
    ):
        """Called after receiving a result from a trial.

        The search algorithm and scheduler are notified before this
        hook is called.

        Arguments:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            trial: Trial that just sent a result.
            result: Result that the trial sent.
            **info: Kwargs dict for forward compatibility.

        Saves videos to disk and provides a video_path to them
        """
        save_videos(result)
        super().on_trial_result(iteration, trials, trial, result, **info)
