from typing import TYPE_CHECKING, Any, ClassVar

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

    from ray_utilities.typing import AlgorithmReturnData


class AdvWandbLoggerCallback(WandbLoggerCallback):
    AUTO_CONFIG_KEYS: ClassVar[list[str]] = list({*WandbLoggerCallback.AUTO_CONFIG_KEYS, "trainable_name"})

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config: dict[str, Any] = _clean_log(config)  # pyright: ignore[reportAssignmentType]
        config = {key: value for key, value in config.items() if key not in self.excludes}
        # --- New Code --- : Remove nested keys
        for nested_key in filter(lambda x: "/" in x, self.excludes):
            key, sub_key = nested_key.split("/")
            if key in config:
                config[key].pop(sub_key, None)
        assert "num_jobs" not in config["cli_args"]
        assert "test" not in config["cli_args"]
        # --- End New Code

        wandb_init_kwargs = {
            "id": trial_id,
            "name": trial_name,
            "resume": False,
            "reinit": True,
            "allow_val_change": True,
            "group": wandb_group,
            "project": wandb_project,
            "config": config,
        }
        wandb_init_kwargs.update(self.kwargs)

        self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)

    def log_trial_result(self, iteration: int, trial: "Trial", result: "AlgorithmReturnData"):  # noqa: ARG002 # pyright: ignore[reportIncompatibleMethodOverride]
        """Called each time a trial reports a result."""
        if trial not in self._trial_logging_actors:
            self.log_trial_start(trial)
        if not self.log_config:
            # Config will be logged once log_trial_start
            result.pop("config", None)  # pyright: ignore[reportCallIssue,reportArgumentType] # pyright bug

        result = _clean_log(result)  # pyright: ignore[reportAssignmentType]
        self._trial_queues[trial].put((_QueueItem.RESULT, result))
