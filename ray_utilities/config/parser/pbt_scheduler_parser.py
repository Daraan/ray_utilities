"""Module for custom Ray Tune schedulers."""

from __future__ import annotations

import inspect
import logging
from ast import literal_eval
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional, TypeAlias, TypeVar

from ray.tune.schedulers import PopulationBasedTraining
from tap import to_tap_class
from random import Random

from ray_utilities.config.parser._common import GoalParser
from ray_utilities.config.parser.subcommand import SubcommandMixin
from ray_utilities.constants import CURRENT_STEP, DEFAULT_EVAL_METRIC

if TYPE_CHECKING:
    from ray.tune.search.sample import Domain

    from ray_utilities.config.parser.default_argument_parser import DefaultResourceArgParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase

logger = logging.getLogger(__name__)

ParentT = TypeVar("ParentT", bound="DefaultResourceArgParser | None")

_HPMutationsType: TypeAlias = dict[str, "dict[Any, Any] | list[Any] | tuple[Any, ...] | Callable[[], Any] | Domain"]

# TODO: Pack into a common submodule without circular dependencies
_T = TypeVar("_T")
NotAModelParameter = Annotated[_T, "NotAModelParameter"]
NeverRestore = Annotated[_T, "NeverRestore"]
AlwaysRestore = Annotated[_T, "AlwaysRestore"]

__default_seed_options = [42, 128, 0, 480, 798]


def get_default_seed_options() -> list[int]:
    """Returns a mutable list of default seed options."""
    try:
        from experiments.create_tune_parameters import seed_options  # noqa: PLC0415, cylclic import
    except ModuleNotFoundError:
        seed_options = __default_seed_options
    return seed_options


def _to_hyperparam_mutations(string: str) -> _HPMutationsType:
    """Convert a string representation of hyperparameter mutations to a dictionary."""
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Invalid hyperparam_mutations format, must be parsable by :func:`ast.literal_eval`: {string}. "
            "Alternatively set it at runtime."
        ) from e


class PopulationBasedTrainingParser(GoalParser, to_tap_class(PopulationBasedTraining), SubcommandMixin[ParentT]):
    """
    Attributes:
        time_attr: The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        perturbation_interval: Models will be considered for
            perturbation at this interval of `time_attr`. Note that
            perturbation incurs checkpoint overhead, so you shouldn't set this
            to be too frequent.
        burn_in_period: Models will not be considered for
            perturbation before this interval of `time_attr` has passed. This
            guarantees that models are trained for at least a certain amount
            of time or timesteps before being perturbed.
        hyperparam_mutations: Hyperparams to mutate. The format is
            as follows: for each key, either a list, function,
            or a tune search space object (tune.loguniform, tune.uniform,
            etc.) can be provided. A list specifies an allowed set of
            categorical values. A function or tune search space object
            specifies the distribution of a continuous parameter. You must
            use tune.choice, tune.uniform, tune.loguniform, etc.. Arbitrary
            tune.sample_from objects are not supported.
            A key can also hold a dict for nested hyperparameters.
            You must specify at least one of `hyperparam_mutations` or
            `custom_explore_fn`.
            Tune will sample the search space provided by
            `hyperparam_mutations` for the initial hyperparameter values if the
            corresponding hyperparameters are not present in a trial's initial `config`.
        quantile_fraction: Parameters are transferred from the top
            `quantile_fraction` fraction of trials to the bottom
            `quantile_fraction` fraction. Needs to be between 0 and 0.5.
            Setting it to 0 essentially implies doing no exploitation at all.
        resample_probability: The probability of resampling from the
            original distribution when applying `hyperparam_mutations`. If not
            resampled, the value will be perturbed by a factor chosen from
            `perturbation_factors` if continuous, or changed to an adjacent value
            if discrete.
        perturbation_factors: Scaling factors to choose between when mutating
            a continuous hyperparameter.
        custom_explore_fn: You can also specify a custom exploration
            function. This function is invoked as `f(config)` after built-in
            perturbations from `hyperparam_mutations` are applied, and should
            return `config` updated as needed. You must specify at least one of
            `hyperparam_mutations` or `custom_explore_fn`.
        log_config: Whether to log the ray config of each model to
            local_dir at each exploit. Allows config schedule to be
            reconstructed.
        require_attrs: Whether to require time_attr and metric to appear
            in result for every iteration. If True, error will be raised
            if these values are not present in trial result.
        synch: If False, will use asynchronous implementation of
            PBT. Trial perturbations occur every perturbation_interval for each
            trial independently. If True, will use synchronous implementation
            of PBT. Perturbations will occur only after all trials are
            synced at the same time_attr every perturbation_interval.
            Defaults to False. See Appendix A.1 here
            https://arxiv.org/pdf/1711.09846.pdf.
        grouped: If True, use GroupedTopPBTTrialScheduler which groups trials
            by configuration and applies PBT quantile logic at the group level
            rather than individual trials. Defaults to False.
    """

    hyperparam_mutations: Optional[_HPMutationsType] = None
    require_attrs: NotAModelParameter[bool] = True
    synch: NotAModelParameter[bool] = True
    log_config: NotAModelParameter[NeverRestore[bool]] = True
    time_attr: str = CURRENT_STEP
    quantile_fraction: NotAModelParameter[float] = 0.1
    perturbation_interval: int | float = 8192 * 14  # (114688) Total should be divisible by total steps
    resample_probability: NotAModelParameter[float] = 1.0  # always resample

    # custom_args, remove before passing to PopulationBasedTraining
    use_native_pbt: NotAModelParameter[AlwaysRestore[bool]] = False
    """Do not use TopPBTTrialScheduler"""

    grouped: NotAModelParameter[AlwaysRestore[bool]] = False
    """Use GroupedTopPBTTrialScheduler for group-based PBT"""

    group_size: NotAModelParameter[AlwaysRestore[int]] = 3
    """Number of trials with same config (differing only in seed) per group. Used with grouped PBT."""

    def set_hyperparam_mutations(self, mutations: _HPMutationsType | None) -> None:
        if mutations is None:
            self.hyperparam_mutations = None
            return
        mutations = mutations.copy()
        if "batch_size" in mutations:
            mutations["train_batch_size_per_learner"] = mutations.pop("batch_size")
            logger.debug("Renaming 'batch_size' hyperparam mutation to 'train_batch_size_per_learner'")
        self.hyperparam_mutations = mutations

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--hyperparam_mutations",
            type=_to_hyperparam_mutations,
            default=None,
            help="Hyperparameter mutations for PopulationBasedTraining.",
        )
        self.add_argument(
            "--perturbation_interval",
            type=lambda x: float(x) if "." in x else int(x),
        )
        # As long as Sentinel cannot be pickled do not add it as default
        # self.add_argument(
        #    "--metric",
        #    type=_resolve_default_metric,
        #    default=DEFAULT_EVAL_METRIC,
        #    help="The metric to be optimized, by default the evaluation metric.",
        # )

    def __init__(
        self,
        *args,
        underscores_to_dashes=False,
        explicit_bool=False,
        config_files=None,
        **kwargs,
    ) -> None:
        # TODO: config files if not available on remote
        super().__init__(
            *args,
            underscores_to_dashes=underscores_to_dashes,
            explicit_bool=explicit_bool,
            config_files=config_files,
            **kwargs,
        )

        # HACK: to_tap_class does not support overrides of default values
        # see https://github.com/swansonk14/typed-argument-parser/issues/166
        def replace_action(arg_name: str, new_default: Any) -> None:
            for action in self._actions:
                if action.dest == arg_name:
                    action.default = new_default
                    break

        # HACK: as noted above, use this class and parent to override defaults
        for var, val in (vars(GoalParser) | vars(PopulationBasedTrainingParser)).items():
            # On python < 3.11 Sentinel passes callable() check
            if val is DEFAULT_EVAL_METRIC or not (
                var.startswith("_") or callable(val) or isinstance(val, (staticmethod, classmethod, property))
            ):
                replace_action(var, val)
        assert (action := next(a for a in self._actions if a.dest == "time_attr")).default == CURRENT_STEP, (
            f"got {action.default}"
        )

    def get_seed_options(self, amount: Optional[int] = None) -> list[int]:
        """
        Returns a deterministic seed sequence for the given amount of seeds.
        This function is used to generate seed options for grouped PBT.

        Build from :func:`get_default_seed_options`.
        """
        amount = amount if amount is not None else self.group_size
        seeds = get_default_seed_options()
        if len(seeds) < amount:
            rng = Random(seeds[-1])
            while len(seeds) < amount:
                new_seed = rng.randint(0, 2**16)
                if new_seed not in seeds:
                    seeds.append(new_seed)
        elif len(seeds) > amount:
            seeds = seeds[:amount]
        return seeds

    def to_scheduler(self, setup: Optional[ExperimentSetupBase] = None) -> PopulationBasedTraining:
        if not self._parsed:
            # When used as subparser we should not end up here
            args = self.parse_args(known_only=True).as_dict()
        else:
            args = self.as_dict()
        args.pop("hyperparam_mutations", None)  # will be set below
        # Set by Tuner, unset
        args["mode"] = None
        args["metric"] = None
        # non-scheduler args
        use_native = args.pop("use_native_pbt")
        grouped = args.pop("grouped", False)

        if self.resample_probability >= 1.0 and self.hyperparam_mutations is None:
            raise ValueError("hyperparam_mutations must be set if resample_probability is 1.0")
        assert not TYPE_CHECKING or self.hyperparam_mutations is not None  # ray has implicit optional
        args = {arg: val for arg, val in args.items() if arg in inspect.signature(PopulationBasedTraining).parameters}

        if use_native:
            return PopulationBasedTraining(**args, hyperparam_mutations=self.hyperparam_mutations)

        num_samples = self.parent.num_samples if self.parent else 1  # pyright: ignore[reportOptionalMemberAccess]

        if grouped:
            from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import (  # noqa: PLC0415
                GroupedTopPBTTrialScheduler,
            )

            if "seed" not in self.hyperparam_mutations and num_samples > 1:
                self.hyperparam_mutations["seed"] = {"grid_search": get_default_seed_options()[:num_samples]}
                logger.warning(
                    "Using GroupedTopPBTTrialScheduler without 'seed' in hyperparam_mutations may lead to "
                    "identical configurations in multiple groups. Consider adding 'seed' to hyperparam_mutations. "
                    "Forcing 'seed' mutation with options: %s",
                    self.hyperparam_mutations["seed"],
                )
                # Important we need this in the parameter space, less in the mutations!
            if setup and "seed" not in setup.param_space:
                setup.param_space["seed"] = {"grid_search": get_default_seed_options()[:num_samples]}
                logger.debug(
                    "Adding 'seed' to experiment param_space with options: %s",
                    setup.param_space["seed"],
                )

            logger.info("Using GroupedTopPBTTrialScheduler for group-based PBT")
            # TODO: Note: Grouped PBT is not compatible with *continuous* sampling, must be all grid_search
            # Or a custom VariantGenerator must be implemented that repeats samples for the group_size
            return GroupedTopPBTTrialScheduler(
                **args,
                hyperparam_mutations=self.hyperparam_mutations,
                num_samples=num_samples,
                group_size=self.group_size,
                prune_late_trials=True,
            )

        from ray_utilities.tune.scheduler.top_pbt_scheduler import TopPBTTrialScheduler  # noqa: PLC0415

        return TopPBTTrialScheduler(
            **args, hyperparam_mutations=self.hyperparam_mutations, num_samples=num_samples, prune_late_trials=True
        )

    # See _ScalingPopulationBasedTrainingParser.process_args to leverage RLLib arguments as well
