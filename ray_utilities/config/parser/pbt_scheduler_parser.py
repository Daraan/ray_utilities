"""Module for custom Ray Tune schedulers."""

from __future__ import annotations

import logging
from ast import literal_eval
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeAlias

from ray.tune.schedulers import PopulationBasedTraining
from tap import to_tap_class

if TYPE_CHECKING:
    from ray.tune.search.sample import Domain

logger = logging.getLogger(__name__)

_HPMutationsType: TypeAlias = dict[str, "dict[Any, Any] | list[Any] | tuple[Any, ...] | Callable[[], Any] | Domain"]


def _to_hyperparam_mutations(string: str) -> _HPMutationsType:
    """Convert a string representation of hyperparameter mutations to a dictionary."""
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Invalid hyperparam_mutations format, must be parsable by :func:`ast.literal_eval`: {string}. "
            "Alternatively set it at runtime."
        ) from e


class PopulationBasedTrainingParser(to_tap_class(PopulationBasedTraining)):
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
    """

    mode: Literal["min", "max"] = "max"
    """One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."""
    hyperparam_mutations: Optional[_HPMutationsType] = None
    require_attrs: bool = True
    synch: bool = True
    log_config: bool = True
    time_attr: str = "current_step"
    quantile_fraction: float = 0.1
    perturbation_interval: float = 100_000
    resample_probability: float = 1.0  # always resample

    def set_hyperparam_mutations(self, mutations: _HPMutationsType | None) -> None:
        self.hyperparam_mutations = mutations

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--hyperparam-mutations",
            type=_to_hyperparam_mutations,
            default=None,
            help="Hyperparameter mutations for PopulationBasedTraining.",
        )

    def __init__(
        self,
        *args,
        underscores_to_dashes=False,
        explicit_bool=False,
        config_files=None,
        **kwargs,
    ) -> None:
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

        for var, val in vars(PopulationBasedTrainingParser).items():
            if not (var.startswith("_") or callable(val) or isinstance(val, (staticmethod, classmethod, property))):
                replace_action(var, val)
        assert (action := next(a for a in self._actions if a.dest == "time_attr")).default == "current_step", (
            f"got {action.default}"
        )

    def to_scheduler(self) -> PopulationBasedTraining:
        if not self._parsed:
            args = self.parse_args().as_dict()
        else:
            args = self.as_dict()
        if self.resample_probability >= 1.0 and self.hyperparam_mutations is None:
            raise ValueError("hyperparam_mutations must be set if resample_probability is 1.0")
        assert not TYPE_CHECKING or self.hyperparam_mutations is not None  # ray has implicit optional
        return PopulationBasedTraining(
            **args,
            hyperparam_mutations=self.hyperparam_mutations,
        )
