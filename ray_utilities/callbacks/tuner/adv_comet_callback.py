# ruff: noqa: PLC0415

from __future__ import annotations

import logging
import math
import sys
import tempfile
import time
from typing import TYPE_CHECKING, ClassVar, Iterable, List, Literal, Optional, cast

from ray.air.integrations.comet import CometLoggerCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.tune.experiment import Trial
from ray.tune.utils import flatten_dict

from ray_utilities.callbacks.tuner._save_video_callback import SaveVideoFirstCallback
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, EPISODE_VIDEO_PREFIX
from ray_utilities.video.numpy_to_video import numpy_to_video

if TYPE_CHECKING:
    from comet_ml import Experiment, OfflineExperiment
    from numpy.typing import NDArray
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing import CometStripedVideoFilename, FlatLogMetricsDict
    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict

__all__ = [
    "AdvCometLoggerCallback",
]

_LOGGER = logging.getLogger(__name__)


class AdvCometLoggerCallback(SaveVideoFirstCallback, CometLoggerCallback):
    # Copy from parent for pylance
    """CometLoggerCallback for logging Tune results to Comet.

    Comet (https://comet.ml/site/) is a tool to manage and optimize the
    entire ML lifecycle, from experiment tracking, model optimization
    and dataset versioning to model production monitoring.

    This Ray Tune ``LoggerCallback`` sends metrics and parameters to
    Comet for tracking.

    In order to use the CometLoggerCallback you must first install Comet
    via ``pip install comet_ml``

    Then set the following environment variables
    ``export COMET_API_KEY=<Your API Key>``

    Alternatively, you can also pass in your API Key as an argument to the
    CometLoggerCallback constructor.

    ``CometLoggerCallback(api_key=<Your API Key>)``

    Args:
            online: Whether to make use of an Online or
                Offline Experiment. Defaults to True.
            tags: Tags to add to the logged Experiment.
                Defaults to None.
            save_checkpoints: If ``True``, model checkpoints will be saved to
                Comet ML as artifacts. Defaults to ``False``.
            exclude_metrics: List of metric keys to exclude from logging.
            log_cli_args: If ``True``, the command line arguments will be logged to Other.
            video_keys: List of keys to log as videos.
            log_to_other: List of keys to log to Other instead of Metrics/Hyperparameters.
                Use '/' to denote nested keys.
            **experiment_kwargs: Other keyword arguments will be passed to the
                constructor for comet_ml.Experiment (or OfflineExperiment if
                online=False).

    Please consult the Comet ML documentation for more information on the
    Experiment and OfflineExperiment classes: https://comet.ml/site/

    Example:

    .. code-block:: python

        from ray.air.integrations.comet import CometLoggerCallback
        tune.run(
            train,
            config=config
            callbacks=[CometLoggerCallback(
                True,
                ['tag1', 'tag2'],
                workspace='my_workspace',
                project_name='my_project_name'
                )]
        )

    """

    _trial_experiments: dict[Trial, Experiment | OfflineExperiment]

    _exclude_results: ClassVar[list[str]] = [
        *CometLoggerCallback._exclude_results,  # noqa: SLF001
        "cli_args/test",
        "evaluation/discrete/env_runners/episode_videos_best/video_path",
        "evaluation/discrete/env_runners/episode_videos_worst/video_path",
        "evaluation/env_runners/episode_videos_best/video_path",
        "evaluation/env_runners/episode_videos_worst/video_path",
    ]
    """Metrics that are not logged"""

    def __init__(
        self,
        *,
        online: bool = True,
        tags: Optional[List[str]] = None,
        save_checkpoints: bool = False,
        # Note: maybe want to log these in an algorithm debugger
        exclude_metrics: Optional[Iterable[str]] = None,
        # NOTE: maintain/sync in _tuner_callbacks_setup.py
        log_to_other: Optional[Iterable[str]] = (
            "comment",
            "cli_args/comment",
            "evaluation/env_runners/environments/seeds",
            "env_runners/environments/seeds",
        ),
        log_cli_args: bool = True,
        video_keys: Iterable[tuple[str, ...]] = DEFAULT_VIDEO_DICT_KEYS,  # NOTE: stored as string not list of keys
        log_pip_packages: bool = False,
        **experiment_kwargs,
    ):
        """
        Args:
            online: Whether to make use of an Online or
                Offline Experiment. Defaults to True.
            tags: Tags to add to the logged Experiment.
                Defaults to None.
            save_checkpoints: If ``True``, model checkpoints will be saved to
                Comet ML as artifacts. Defaults to ``False``.
            exclude_metrics: List of metric keys to exclude from logging.
            log_cli_args: If ``True``, the command line arguments will be logged to Other.
            video_keys: List of keys to log as videos.
            log_to_other: List of keys to log to Other instead of Metrics/Hyperparameters.
                Use '/' to denote nested keys.
            log_pip_packages: If ``True``, the installed packages will be logged, this is always ``True``
                if ``log_env_details`` is ``True``, which however is more expensive if set to ``True``.
            **experiment_kwargs: Other keyword arguments will be passed to the
                constructor for comet_ml.Experiment (or OfflineExperiment if
                online=False).
        """
        super().__init__(online=online, tags=tags, save_checkpoints=save_checkpoints, **experiment_kwargs)  # pyright: ignore[reportArgumentType]

        # Join video keys for flat dict access
        self._video_keys = video_keys
        """Video keys in their tuple form; probably without /video and /reward suffix"""
        joined_keys = ["/".join(keys) for keys in video_keys]
        # Videos are stored as dict with "video" and "reward" keys
        self._flat_video_lookup_keys = [k + "/video" if not k.endswith("/video") else k for k in joined_keys]
        """Contains only /video keys"""
        self._flat_video_keys = self._flat_video_lookup_keys + [
            k + "/reward" if not k.endswith("/reward") else k for k in joined_keys
        ]
        """Contains /video and /reward keys"""

        self._to_exclude.append("log_level")
        self._to_exclude.extend(
            [*exclude_metrics, *self._flat_video_keys] if exclude_metrics else self._flat_video_keys
        )
        """Keys that are not logged at all"""
        self._to_other.extend(log_to_other or [])
        self._cli_args = " ".join(sys.argv[1:]) if log_cli_args else None
        self._log_only_once = [
            *self._to_exclude,
            *self._to_system,
            # NOTE: These are NOT logged on log_trial_start and might not be logged on_trial_result
            # Do not add them here!
            # "env_runners/environments/seeds",
            # "evaluation/env_runners/environments/seeds",
        ]  # + all config values; but flat keys!
        if (
            "env_runners/environments/seeds" in self._log_only_once
            or "evaluation/env_runners/environments/seeds" in self._log_only_once
        ):
            _LOGGER.warning("environment seeds are not logged, remove from log_only_once")
        if "training_iteration" in self._log_only_once:
            self._log_only_once.remove("training_iteration")
            _LOGGER.debug("training_iteration must be in the results to log it, not removing it")
        self._log_pip_packages = log_pip_packages and not experiment_kwargs.get("log_env_details", False)  # noqa: RUF056
        """If log_env_details is True pip packages are already logged."""

    def _check_workspaces(self, trial: Trial) -> Literal[0, 1, 2]:
        """
        Return:
            0: If workspace is present
            1: If no workspace were found due to an exception, e.g. no internet connection.
            2: If workspace is not found in the accound
        """
        from comet_ml import API
        from comet_ml.exceptions import CometRestApiException
        from comet_ml.experiment import LOGGER as COMET_LOGGER

        try:
            api = API()
            workspaces = api.get_workspaces()
        except CometRestApiException as e:
            # Maybe offline?
            _LOGGER.warning(
                "Failed to retrieve workspaces from Comet API. Cannot check if selected workspace is valid: %s", e
            )
            return 1
        if (workspace := self.experiment_kwargs.get("workspace", None)) is not None and (
            workspace not in workspaces and workspace.lower() not in workspaces
        ):
            COMET_LOGGER.error(
                "======================================== \n"
                "Comet Workspace '%s' not found in available workspaces: %s. "
                "You need to create it first! Waiting 5s for a possible abort then using default workspace\n"
                "========================================",
                workspace,
                workspaces,
            )
            time.sleep(5)
            self.experiment_kwargs["workspace"] = None
            return 2
        return 0

    def log_trial_start(self, trial: "Trial"):
        """
        Initialize an Experiment (or OfflineExperiment if self.online=False)
        and start logging to Comet.

        Args:
            trial: Trial object.

        Overwritten method to respect ignored/refactored keys.
        nested to other keys will only have their deepest key logged.
        """
        from comet_ml import Experiment, OfflineExperiment
        from comet_ml.config import set_global_experiment

        if trial not in self._trial_experiments:
            experiment_cls = Experiment if self.online else OfflineExperiment
            experiment_kwargs = self.experiment_kwargs.copy()
            experiment_kwargs.setdefault("experiment_key", trial.trial_id)
            self._check_workspaces(trial)
            experiment = experiment_cls(**self.experiment_kwargs)
            if self._log_pip_packages:
                try:
                    experiment.set_pip_packages()
                except Exception:
                    from comet_ml.experiment import (
                        EXPERIMENT_START_FAILED_SET_PIP_PACKAGES_ERROR,
                    )

                    logging.getLogger("comet_ml.experiment").exception(EXPERIMENT_START_FAILED_SET_PIP_PACKAGES_ERROR)
            self._trial_experiments[trial] = experiment
            # Set global experiment to None to allow for multiple experiments.
            set_global_experiment(None)
        else:
            experiment = self._trial_experiments[trial]

        experiment.set_name(str(trial))
        experiment.add_tags(self.tags)
        experiment.log_other("Created from", "Ray")

        # NOTE: Keys here at not flattened, cannot use "cli_args/test" as a key
        # Unflattening only supports one level of nesting
        config = trial.config.copy()
        non_parameter_keys = self._to_exclude + self._to_other
        flat_config = flatten_dict(config)
        # get all the parent/child keys that are now in the flat config
        nested_keys = [k for k in non_parameter_keys if k in flat_config and k not in config]

        # find nested keys and
        to_other = {}
        for nested_key in nested_keys:
            k1, k2 = nested_key.split("/")
            if k1 in config and k2 in config[k1]:
                v2 = config[k1].pop(k2)
                if nested_key in self._to_other:
                    if k2 in to_other:
                        # Conflict, add to the parent key
                        to_other[nested_key] = v2
                    else:
                        to_other[k2] = v2
                if len(config[k1]) == 0:
                    config.pop(k1)

        experiment = self._trial_experiments[trial]
        experiment.log_parameters(config)
        # Log the command line arguments
        if self._cli_args:
            experiment.log_other("args", self._cli_args)
        # Log non nested config keys
        for key in self._to_other:
            if key in trial.config:
                experiment.log_other(key, trial.config[key])
        # Log nested config keys
        if to_other:
            experiment.log_others(to_other)

    def log_trial_result(self, iteration: int, trial: Trial, result: dict | AutoExtendedLogMetricsDict):
        step: int = result["training_iteration"]  # maybe result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        # Will be flattened in super anyway
        flat_result: FlatLogMetricsDict = flatten_dict(result, delimiter="/")  # type: ignore[arg-type]
        del result  # avoid using it by mistake

        videos: dict[str, NDArray | float | str] = {k: v for k in self._flat_video_keys if (v := flat_result.get(k))}

        # Remove Video keys and NaN values which can cause problems in the Metrics Tab when logged
        if trial in self._trial_experiments:
            # log_trial_start was called already, do not log parameters again
            # NOTE: # WARNING: This prevents config_updates during the run!
            log_result = {
                k: v
                for k, v in flat_result.items()
                if not (k in self._log_only_once or k in self._flat_video_keys or k.startswith("config/"))
                and (not isinstance(v, float) or not math.isnan(v))
            }  # type: ignore[assignment]
        else:
            log_result = {
                k: v
                for k, v in flat_result.items()
                if k not in self._flat_video_keys and (not isinstance(v, float) or not math.isnan(v))
            }  # type: ignore[assignment]
        # These are only once a list of int, after reduce this list is empty:
        if not log_result.get("env_runners/environments/seeds", True):
            del log_result["env_runners/environments/seeds"]
        if not log_result.get("evaluation/env_runners/environments/seeds", True):
            del log_result["evaluation/env_runners/environments/seeds"]
        # Cannot remove this
        log_result["training_iteration"] = step
        # Log normal metrics and parameters
        super().log_trial_result(iteration, trial, log_result)  # pyright: ignore[reportArgumentType]
        if videos:
            experiment = self._trial_experiments[trial]
            for video_key in self._flat_video_lookup_keys:
                video: NDArray | str | None = videos.get(video_key)  # type: ignore[assignment] # do not extract float
                if not video:
                    continue
                # turn key to evaluation_best_video, evaluation_discrete_best_video, etc.
                stripped_key: CometStripedVideoFilename = (
                    video_key.replace(ENV_RUNNER_RESULTS + "/", "").replace(EPISODE_VIDEO_PREFIX, "").replace("/", "_")
                )  # type: ignore[assignment]
                # Filename that is used for logging; not on disk
                filename = f"videos/{stripped_key}.mp4"  # e.g. step0040_best.mp4

                # Already a saved video:
                if (video_path_key := video_key.replace("/video", "/video_path")) in flat_result:
                    video = cast("str", flat_result[video_path_key])
                    logging.getLogger("comet_ml").debug("Logging video from %s", video)

                metadata = {
                    "reward": flat_result[video_key.replace("/video", "/reward")],
                    "discrete": "discrete" in video_key,
                    **(
                        {"video_path": flat_result[path_key]}
                        if (path_key := video_key.replace("/video", "/video_path")) in flat_result
                        else {}
                    ),
                }

                if isinstance(video, str):
                    experiment.log_video(video, name=filename, step=step, metadata=metadata)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="temp_dir") as temp:
                        # os.makedirs(os.path.dirname(filename), exist_ok=True)
                        numpy_to_video(video, video_filename=temp.name)
                        experiment.log_video(
                            temp.name,
                            name=filename,
                            step=step,
                            metadata=metadata,
                        )
            experiment.log_other("hasVideo", value=True)
