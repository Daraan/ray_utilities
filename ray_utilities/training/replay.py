from __future__ import annotations

import json
import pathlib
from inspect import isclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Collection, Literal, Optional, cast
from unittest import mock

from typing_extensions import Self

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import RE_GET_TRIAL_ID, trial_name_creator
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import get_args_and_config

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
    from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict


class ReplayTrainable(DefaultTrainable["DefaultArgumentParser", Any, Any]):
    replay_file: pathlib.Path

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        replay_file: Optional[str | pathlib.Path] = None,
        start_iteration: Optional[int] = None,
        start_step: Optional[int] = None,
        algorithm_overrides: dict[str, Any] | None = None,
        model_config: dict[str, Any] | DefaultModelConfig | None = None,
        loggers: Collection[Literal["comet", "wandb"]] = (
            # "comet",
            "wandb",
        ),
        # For trial
        trainable_name: str,
        trial_id: str,
        trial_name_creator=trial_name_creator,
        **kwargs,
    ):
        if replay_file is not None:
            self.replay_file = pathlib.Path(replay_file)
        if not config:
            with self.replay_file.open("r") as f:
                first_line = f.readline()
                first_result = json.loads(first_line)
                config = first_result["config"]

        self._replay_iteration = 0
        if start_step is not None:
            if start_iteration is not None:
                raise ValueError("Cannot specify both start_step and start_iteration.")
            for start_iteration, step in enumerate(self._replay_step()):  # noqa: B007, PLR1704
                if step["current_step"] >= (start_step or 0):
                    break
        self._first_result = next(self._replay_step())
        self._start_iteration = start_iteration or 0
        super().__init__(config, algorithm_overrides=algorithm_overrides, model_config=model_config, **kwargs)
        from ray.tune.experiment.trial import Trial

        self._mock_trial = Trial(
            trainable_name=trainable_name,
            config=config,
            trial_name_creator=trial_name_creator,
            trial_id=trial_id,
            stub=True,
        )
        mock_storage = SimpleNamespace(trial_driver_staging_path=self.replay_file.parent, trial_working_directory="na")
        print("local path will be", self.replay_file.parent)
        self._mock_trial.storage = mock_storage
        self._callbacks = self._make_callbacks(loggers)

    @classmethod
    def define(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        setup_cls: type[ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]],
        replay_file: str | pathlib.Path,
        **kwargs,
    ) -> type[Self]:
        Trainable: type[Self] = super().define(setup_cls, **kwargs)  # pyright: ignore[reportAssignmentType]
        Trainable.replay_file = pathlib.Path(replay_file)
        return Trainable

    def _setup_trainable(self, hparams: dict[str, Any], *, create_algorithm: bool = False):
        algorithm = mock.MagicMock() if create_algorithm else None
        if algorithm is not None:
            algorithm.config.num_learners = 1
        args, config = get_args_and_config(
            hparams,
            setup=self._setup,
            setup_class=self.setup_class if isclass(self.setup_class) else None,
            model_config=self._model_config,
        )
        self._replay_generator = self._replay_step()
        if self._start_iteration > 0:
            i = -1
            for i in range(self._start_iteration):  # noqa: B007
                next(self._replay_generator)
                self._iteration += 1
                self._iterations_since_restore += 1
            print("Skipped to iteration", i + 1)
        return args, config, algorithm, None

    def _make_callbacks(
        self, loggers: Collection[Literal["comet", "wandb"]]
    ) -> dict[str, AdvCometLoggerCallback | AdvWandbLoggerCallback]:
        # Let tuner restore callbacks?
        self._setup.args.wandb = "offline+upload@end"
        tuner_setup = TunerSetup(
            setup=self._setup,
            eval_metric=self._setup.args.metric,
            eval_metric_order=self._setup.args.mode,
            add_iteration_stopper=self._setup._tuner_add_iteration_stopper(),
            trial_name_creator=self._setup._tune_trial_name_creator,
        )
        callbacks = {}
        if "wandb" in loggers:
            callbacks["wandb"] = wandb_callback = tuner_setup.create_wandb_logger()
            wandb_callback.on_trial_start(0, [self._mock_trial], self._mock_trial)
        if "comet" in loggers:
            callbacks["comet"] = comet_callback = tuner_setup.create_comet_logger()
            comet_callback.on_trial_start(0, [self._mock_trial], self._mock_trial)
        return callbacks

    def create_wandb_run(self, mode="offline"):
        import wandb  # noqa: PLC0415

        logger = cast("AdvWandbLoggerCallback", self._callbacks["wandb"])

        self._first_result["config"]

        config = self._first_result["config"]
        config.pop("callbacks", None)  # Remove callbacks
        config.pop("log_level", None)

        trial_dir = self.replay_file.parent
        experiment_dir = trial_dir.parent
        if logger.group is None:
            logger.group = experiment_dir.name.rsplit("-", 1)[0]
        if self.replay_file.name == "result.json":
            trial_id = RE_GET_TRIAL_ID.search(args.replay_file).groupdict("trial_id")["trial_id"]
        else:
            # named result-fork.trial_id.json
            trial_id = args.replay_file.stem.split("-")[-1]

        wandb_init_kwargs = {
            "id": trial_id,  # change if forked? e.g. + forked_from
            "name": None if logger.kwargs.get("resume") else trial_dir.name,
            "reinit": "default",  # bool is deprecated
            "allow_val_change": True,
            "group": logger.group,
            "project": logger.project,
            "config": config,
            # possibly fork / resume
        }
        if FORK_FROM in config:
            wandb_init_kwargs["fork_from"] = config[FORK_FROM]
            wandb_init_kwargs["name"] = logger.make_forked_trial_name(trial_dir.name, config[FORK_FROM])

        # if we resume do no set trial name to not overwrite it
        wandb_init_kwargs.update(logger.kwargs)
        if self._first_result["config"].get("fork_from"):
            wandb_init_kwargs.setdefault("tags", []).append("forked")
        wandb_init_kwargs["tags"].append("replayed")
        wandb_init_kwargs.pop("mode", None)
        self.wandb_run = wandb.init(**wandb_init_kwargs, mode=mode, dir=trial_dir)
        return self.wandb_run

    def _replay_step(self):
        with open(self.replay_file, "r") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Could not decode line in replay file {self.replay_file}: '{line}'")
                    raise
                self._replay_iteration += 1

    def step(self):
        return next(self._replay_generator)

    def end_replay(self):
        for callback in self._callbacks.values():
            callback.on_experiment_end([self._mock_trial])

    def train(self) -> AutoExtendedLogMetricsDict:
        try:
            result = super().train()
        except StopIteration:
            self.end_replay()
            return {}  # pyright: ignore[reportReturnType]
        if isinstance(result, list):
            results = result
        else:
            results = [result]
        for result in results:
            print(
                "Replayed iteration:",
                self.iteration,
                "current_step",
                result.get("current_step", "n/a"),
            )
            print("Result keys:", list(result.keys()))
            for callback in self._callbacks.values():
                callback.on_trial_result(self.iteration, [], trial=self._mock_trial, result=result)

        return result  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    import argparse
    import sys

    from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

    parser = argparse.ArgumentParser()
    parser.add_argument("replay_file", type=str)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--trainable_name", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    args, extra = parser.parse_known_args()
    replay_path = pathlib.Path(args.replay_file)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file {replay_path} does not exist.")
    with replay_path.open("r") as f:
        first_line = f.readline()
        first_result = json.loads(first_line)
        config = first_result["config"]
        if not args.trainable_name:
            args.trainable_name = first_result["config"].get("trainable_name", "ReplayTrainable")
        if not args.group:
            args.group = config["experiment_group"]
        if not args.project:
            args.project = config.get("experiment_name")
            if args.project is None:
                project, experiment_id = replay_path.parts[replay_path.parts.index("experiments") + 1].rsplit("-", 1)
                args.project = project
        trial_id = config.get("trial_id")
        if trial_id is None:
            if replay_path.name == "result.json":
                trial_id = RE_GET_TRIAL_ID.search(args.replay_file).groupdict("trial_id")["trial_id"]
            else:
                # named result-fork.trial_id.json
                trial_id = replay_path.stem.split("-")[-1]

    PPOMLPSetup.PROJECT = args.project
    PPOMLPSetup.group_name = args.group  # pyright: ignore[reportAttributeAccessIssue]
    Trainable: type[ReplayTrainable] = ReplayTrainable.define(
        PPOMLPSetup,
        replay_file=args.replay_file,
    )
    sys.argv = sys.argv[:1] + extra
    trainable = Trainable(
        replay_file=replay_path,
        config=config,
        trial_id=trial_id,
        trainable_name=args.trainable_name,
        start_iteration=args.start,
    )
    while True:
        result = trainable.train()
        if not result:
            break
    print("Replay finished.")
    # run = trainable.create_wandb_run()
    # for result in trainable._replay_step():
    #    trainable.wandb_run.log(result)
    # trainable.wandb_run.finish()
