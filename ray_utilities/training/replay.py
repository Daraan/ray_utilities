from __future__ import annotations

import json
import pathlib
from inspect import isclass
from typing import TYPE_CHECKING, Any, Collection, Literal, Optional, Sequence, cast
from unittest import mock

from typing_extensions import Self

from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
from ray_utilities.callbacks.wandb import wandb_api
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import RE_GET_TRIAL_ID
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import get_args_and_config

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase


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
        loggers: Collection[Literal["comet", "wandb"]] = ("comet", "wandb"),
        **kwargs,
    ):
        if replay_file is not None:
            self.replay_file = pathlib.Path(replay_file)

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
        self._calbacks = self._make_callbacks(loggers)

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
            for _ in range(self._start_iteration):
                next(self._replay_generator)
        return args, config, algorithm, None

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

    def _make_callbacks(
        self, loggers: Collection[Literal["comet", "wandb"]]
    ) -> dict[str, AdvCometLoggerCallback | AdvWandbLoggerCallback]:
        # Let tuner restore callbacks?
        tuner_setup = TunerSetup(
            setup=self._setup,
            eval_metric=self._setup.args.metric,
            eval_metric_order=self._setup.args.mode,
            add_iteration_stopper=self._setup._tuner_add_iteration_stopper(),
            trial_name_creator=self._setup._tune_trial_name_creator,
        )
        callbacks = {}
        if "wandb" in loggers:
            callbacks["wandb"] = tuner_setup.create_wandb_logger()
        if "comet" in loggers:
            callbacks["comet"] = tuner_setup.create_comet_logger()
        return callbacks

    def create_wandb_run(self, mode="offline"):
        import wandb  # noqa: PLC0415

        logger = cast("AdvWandbLoggerCallback", self._calbacks["wandb"])

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


if __name__ == "__main__":
    import argparse
    from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("replay_file", type=str)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--group", type=str, default=None)
    args, extra = parser.parse_known_args()
    PPOMLPSetup.PROJECT = args.project
    PPOMLPSetup.group_name = args.group  # pyright: ignore[reportAttributeAccessIssue]

    Trainable: type[ReplayTrainable] = ReplayTrainable.define(
        PPOMLPSetup,
        replay_file=args.replay_file,
    )
    sys.argv = sys.argv[:1] + extra
    trainable = Trainable(replay_file=args.replay_file)
    run = trainable.create_wandb_run()
    for result in trainable._replay_step():
        trainable.wandb_run.log(result)
    trainable.wandb_run.finish()
