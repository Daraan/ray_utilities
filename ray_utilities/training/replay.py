from __future__ import annotations

import json
import pathlib
from inspect import isclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Collection, Literal, Optional, cast
from unittest import mock

from ray.tune.experiment.trial import Trial
from tap import Tap
from typing_extensions import Self, deprecated

from ray_utilities.callbacks.wandb import RunNotFound, WandbUploaderMixin, wandb_api
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import RE_GET_TRIAL_ID, ExperimentKey, trial_name_creator
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import get_args_and_config

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    import wandb
    from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
    from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.config.parser.default_argument_parser import OnlineLoggingOption
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing import ForkFromData
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
        upload_mode: OnlineLoggingOption = "offline+upload@end",
        wandb_run: Optional[wandb.Run] = None,
        **kwargs,
    ):
        if replay_file is not None:
            self.replay_file = pathlib.Path(replay_file)
        if not config:
            with self.replay_file.open("r") as f:
                first_line = f.readline()
                first_result = json.loads(first_line)
                config = first_result["config"]
        assert config is not None
        self._tags = []
        if wandb_run is not None:
            self._tags = wandb_run.tags
            self._comment = wandb_run.notes
        else:
            self._comment = config.get("comment", "")
        self._wandb_upload_mode = upload_mode

        self._replay_iteration = 0
        if start_step is not None:
            if start_iteration is not None:
                raise ValueError("Cannot specify both start_step and start_iteration.")
            for start_iteration, step in enumerate(self._replay_step()):  # noqa: B007, PLR1704
                if step["current_step"] >= (start_step or 0):
                    break
        # self._first_result = next(self._replay_step())
        self._start_iteration = start_iteration or 0
        super().__init__(config, algorithm_overrides=algorithm_overrides, model_config=model_config, **kwargs)
        self._setup.args.comment = self._comment or None
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
        self._mock_trial.storage = mock_storage  # pyright: ignore[reportAttributeAccessIssue]
        self._callbacks = self._make_callbacks(
            loggers,
        )

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
        self._setup.args.wandb = self._wandb_upload_mode
        # ordered unique tags
        self._setup.args.tags = list(
            (dict.fromkeys(self._setup.args.tags, True) | dict.fromkeys(self._tags or [], True)).keys()
        )
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

    @deprecated("use tune methods instead")
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
        # query online for tags

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

    __last_step_print: int = -8000

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
            current_step = result.get("current_step", None)  # pyright: ignore[reportAttributeAccessIssue]
            if self.iteration % 10 == 0 and (current_step is None or (current_step - self.__last_step_print) >= 8000):
                print(
                    "Replayed iteration:",
                    self.iteration,
                    "current_step",
                    current_step or "n/a",
                )
                self.__last_step_print = current_step or -8000
            for callback in self._callbacks.values():
                callback.on_trial_result(self.iteration, [], trial=self._mock_trial, result=result)

        return result  # pyright: ignore[reportReturnType]


_API = None
if __name__ == "__main__":

    def gather_whole_experiment(path: pathlib.Path):
        return list(path.glob("**/result*.json"))

    def config_from_result_file(replay_path: pathlib.Path, *, is_fork: bool = False, fork_step: int | None = None):
        if not is_fork:
            with replay_path.open("r") as f:
                first_line = f.readline()
                first_result = json.loads(first_line)
                return first_result["config"]
        # Iterate until FORK_FROM is found with matching step
        if fork_step is None:
            raise ValueError("fork_step must be specified for forked experiments.")
        config = None
        return_next = False
        with replay_path.open("r") as f:
            for line in f:
                result = json.loads(line)
                if "current_step" not in result:
                    raise ValueError("Result does not contain current_step.")
                if result["current_step"] < fork_step:
                    continue
                config = result["config"]
                if FORK_FROM in config:
                    # Check if fork_step matches
                    fork_data = cast("ForkFromData", config[FORK_FROM])
                    if fork_data.get("parent_env_steps") == fork_step:
                        return config  # found correct
                    parent_data = fork_data["parent_time"]
                    if parent_data[0] == "current_step" and parent_data[1] == fork_step:
                        return config
                    if result["current_step"] == fork_step:
                        return_next = True
                if return_next:
                    return config
        raise ValueError(f"Could not find FORK_FROM at step {fork_step} in {replay_path}. .json incomplete?")

    import argparse
    import logging
    import sys

    import ray_utilities.constants
    from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

    logger = logging.getLogger(__name__)

    class Parser(Tap):
        replay_file: str
        experiment: bool = False  # Replay the whole experiment
        start: Optional[int] = None
        trainable_name: Optional[str] = None
        project: Optional[str] = None
        group: Optional[str] = None
        new_id: Optional[str | Literal[True]] = None
        ignore_old_wandb_paths: bool = False

    parser = argparse.ArgumentParser()
    parser.add_argument("replay_file", type=str)
    parser.add_argument("--experiment", action="store_true", default=False, help="Replay the whole experiment")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--trainable_name", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--new_id", nargs="?", const=True, type=str, default=None)
    parser.add_argument("--ignore-old-wandb-paths", action="store_true", default=False)
    args, extra = parser.parse_known_args()
    args = cast("Parser", args)
    _API = wandb_api()
    if args.experiment:
        replay_files = gather_whole_experiment(pathlib.Path(args.replay_file))
    else:
        replay_files = [pathlib.Path(args.replay_file)]
    if args.new_id is True:
        new_trial_id_base = Trial.generate_id()

    if args.ignore_old_wandb_paths:
        # Depending in the situation we do not want to re-upload the old wandb_paths.
        # Need to know which are old
        old_wandb_paths = {
            wandb_run for replay_path in replay_files for wandb_run in (replay_path.parent / "wandb").glob("*run*")
        }

    failures = []
    uploader = WandbUploaderMixin()
    import ray

    ray.init()

    for replay_path in replay_files:
        # input("press for next")
        if not replay_path.exists():
            raise FileNotFoundError(f"Replay file {replay_path} does not exist.")
        with replay_path.open("r") as f:
            first_line = f.readline()
            try:
                first_result = json.loads(first_line)
            except json.JSONDecodeError:
                logger.error(f"Could not decode first line of {replay_path}: '{first_line}'")  # noqa: G004
                failures.append(replay_path)
                continue
            run_id = first_result["config"]["run_id"]
            logger.info(f"For replay changing RUN_ID to {run_id}")  # noqa: G004
            ray_utilities.constants._RUN_ID = run_id
            if ExperimentKey.FORK_SEPARATOR in replay_path.name:
                # Determine parent step
                parent_steps = replay_path.stem.split(ExperimentKey.FORK_SEPARATOR)[-1].split(
                    ExperimentKey.STEP_SEPARATOR
                )[-1]
                import base62

                parent_steps = base62.decode(parent_steps)
                try:
                    config = config_from_result_file(replay_path, is_fork=True, fork_step=parent_steps)
                except ValueError:
                    logger.exception("Could not find config for %s. .json incomplete?", replay_path)
                    failures.append(replay_path)
                    continue
            else:
                config = first_result["config"]
            if args.ignore_old_wandb_paths:
                config["wandb"] = "offline"
            if not args.trainable_name:
                args.trainable_name = first_result["config"].get("trainable_name", "ReplayTrainable")
            if not args.group:
                args.group = config["experiment_group"]
            if not args.project:
                args.project = config.get("experiment_name")
                if args.project is None:
                    project, experiment_id = replay_path.parts[replay_path.parts.index("experiments") + 1].rsplit(
                        "-", 1
                    )
                    args.project = project

            trial_id = config.get("trial_id")
            if trial_id is None:
                if replay_path.name == "result.json":
                    trial_id = RE_GET_TRIAL_ID.search(str(replay_path)).groupdict("trial_id")["trial_id"]
                else:
                    # named result-fork.trial_id.json
                    trial_id = replay_path.stem.split("-")[-1]
            # Before we replay it verify if it does need replay (last step + all intermediate steps exist)
            # region verify
            uploader = WandbUploaderMixin()
            uploader.project = args.project
            verify_failures = uploader.verify_wandb_uploads(
                experiment_id=config["run_id"], output_dir=replay_path.parent.parent.parent, single_experiment=trial_id
            )
            if not verify_failures or all(
                not (
                    isinstance(run, RunNotFound) or isinstance(failure, Exception) or any(not f.minor for f in failure)
                )
                for run, failure in verify_failures.items()
            ):
                logger.info("Replay of {replay_path} not needed, all steps uploaded.")
                if args.new_id and args.new_id is not True and args.new_id.startswith("clone"):
                    choice = ""
                    while choice.lower() not in ("y", "n"):
                        choice = input(
                            f"Replay of {replay_path} not needed, all steps uploaded. "
                            f"Do you want to clone it anyway under new ID {args.new_id}? (y/n): "
                        )
                    if choice.lower() == "y":
                        pass
                    else:
                        continue
                else:
                    continue

            # endregion
            original_trial_id = trial_id
            if args.new_id is not None:
                if args.new_id is True:
                    if "_" in trial_id:
                        # carry over the counter
                        trial_id = f"{new_trial_id_base}_{trial_id.split('_', 1)[1]}"  # pyright: ignore[reportPossiblyUnboundVariable]
                    elif ExperimentKey.FORK_SEPARATOR in trial_id:
                        # this is hard if we want to keep the fork info
                        original_id_base = trial_id.split(ExperimentKey.FORK_SEPARATOR, 1)[0].split(
                            ExperimentKey.RUN_ID_SEPARATOR, 1
                        )[1]
                        trial_id = trial_id.replace(original_id_base, new_trial_id_base)  # pyright: ignore[reportPossiblyUnboundVariable]
                elif args.new_id.startswith("clone"):
                    trial_id = original_trial_id + "_" + args.new_id
                else:
                    new_trial_id = args.new_id
                    if "_" not in trial_id and ExperimentKey.FORK_SEPARATOR not in trial_id:
                        # carry over counter if a simple id was passed
                        trial_id = f"{new_trial_id}_{trial_id.split('_', 1)[1]}"
                    else:
                        trial_id = new_trial_id

        PPOMLPSetup.PROJECT = args.project
        PPOMLPSetup.group_name = args.group  # pyright: ignore[reportAttributeAccessIssue]
        Trainable: type[ReplayTrainable] = ReplayTrainable.define(
            PPOMLPSetup,
            replay_file=args.replay_file,
        )
        sys.argv = sys.argv[:1] + extra
        api = _API or wandb_api()
        try:
            run: wandb.Run | None = api.run(f"{args.project}/{original_trial_id}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not fetch wandb run for {args.project}/{original_trial_id}: {e}")  # noqa: G004
            run = None
        if ExperimentKey.FORK_SEPARATOR in replay_path.name:
            # need to determine if we fork existing parent or also cloned parent
            fork_data = cast("ForkFromData", config[FORK_FROM])
            if args.new_id and args.new_id is not True and args.new_id.startswith("clone") and args.experiment:
                # need to update fork_from to new cloned
                fork_data["parent_fork_id"] = fork_data["parent_fork_id"] + "_" + args.new_id
            fork_data["fork_id_this_trial"] = trial_id
            # define ID for this trial. NOTE: Might not work with comet as too long.
            config["experiment_key"] = trial_id

            # TODO: Should write this data into the fork file
            # Data might also be missing
            # fork_csv = pd.read_csv(replay_path.parent / "forked_from.csv")

        trainable = Trainable(
            replay_file=replay_path,
            config=config,
            trial_id=trial_id,
            trainable_name=args.trainable_name,
            start_iteration=args.start,
            # NOTE: If we upload at end this blocks, but no upload block below
            upload_mode="offline+upload@end" if not args.ignore_old_wandb_paths else "offline",
            wandb_run=run,
            trial_name_creator=(lambda _trial, run=run: run.name) if run is not None else trial_name_creator,
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
    if args.ignore_old_wandb_paths:
        import base62

        # Depending in the situation we do not want to re-upload the old wandb_paths.
        # Filter out old wandb_paths from the new ones
        new_wandb_paths = {
            wandb_run
            for replay_path in replay_files
            for wandb_run in (replay_path.parent / "wandb").glob("*run*")
            if wandb_run not in old_wandb_paths  # pyright: ignore[reportPossiblyUnboundVariable]
        }
        # Upload new wandb runs

        # There is no real dependency ordering here, have no upload groups here, sort in an intelligent way
        new_wandb_paths = list(new_wandb_paths)

        basic_runs = [p for p in new_wandb_paths if ExperimentKey.FORK_SEPARATOR not in p.name]
        uploader = WandbUploaderMixin()
        uploader.project = args.project
        if basic_runs:
            uploader.upload_paths(basic_runs, wait=True, use_tqdm=True)
        other = [p for p in new_wandb_paths if p not in basic_runs]
        # NOTE: Required parent could still end up in a parallel upload, hope wait and retry is enough to catch up.
        other.sort(
            key=lambda p: (
                ExperimentKey.FORK_SEPARATOR in p.name,
                ExperimentKey.COUNT_SEPARATOR not in p.name
                or base62.decode(p.stem.split(ExperimentKey.COUNT_SEPARATOR)[-1].split("_")[0].strip(" -_")),
            )
        )
        uploader.upload_paths(other, parallel_uploads=2, wait=True, use_tqdm=True)
    if failures:
        logger.error("Failures during replay of the following files:")
        for failure in failures:
            print(failure)
