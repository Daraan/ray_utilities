"""Extension mixins for dynamic configuration in Ray RLlib experiment setups.

This module provides mixin classes that extend the base experiment setup functionality
with dynamic configuration capabilities. These mixins can be composed with base setup
classes to add features like dynamic buffer sizing, batch size adjustment, and
adaptive evaluation intervals during training.

The mixins are designed to work together and with the base
:class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase` class through
multiple inheritance, providing modular functionality that can be mixed and matched
based on experiment requirements.

Key Components:
    - :class:`SetupWithDynamicBuffer`: Dynamic experience buffer sizing
    - :class:`SetupWithDynamicBatchSize`: Dynamic batch size and gradient accumulation
    - :class:`SetupForDynamicTuning`: Base class for dynamic configuration mixins

These extensions integrate with Ray Tune's parameter spaces and RLlib callbacks
to provide adaptive behavior during training and hyperparameter optimization.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast

import optuna
from ray import tune
from typing_extensions import Self

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import add_dynamic_eval_callback_if_missing
from ray_utilities.setup.experiment_base import (
    AlgorithmType_co,
    ConfigType_co,
    ExperimentSetupBase,
    ParserType_co,
    SetupCheckpointDict,
)
from ray_utilities.tune import update_hyperparameters

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.tune.search.sample import Domain

    from ray_utilities.typing import ParameterSpace

_logger = logging.getLogger(__name__)


def optuna_dist_to_ray_distribution(dist: optuna.distributions.BaseDistribution) -> Domain:
    """Convert an Optuna distribution to a Ray Tune distribution.

    This method converts an Optuna distribution object into a corresponding
    Ray Tune distribution that can be used for hyperparameter tuning.

    Args:
        dist: An Optuna distribution object.

    Returns:
        A Ray Tune distribution corresponding to the provided Optuna distribution.

    Raises:
        ValueError: If the distribution type is not supported or cannot be converted.
    """
    if isinstance(dist, optuna.distributions.FloatDistribution):
        lower = dist.low
        upper = dist.high
        step = dist.step
        log = dist.log

        if log:
            if step is not None:
                return tune.qloguniform(lower, upper, step)
            return tune.loguniform(lower, upper)

        if step is not None:
            return tune.quniform(lower, upper, step)
        return tune.uniform(lower, upper)

    if isinstance(dist, optuna.distributions.IntDistribution):
        lower = dist.low
        upper = dist.high
        step = dist.step
        log = dist.log
        # NOTE: step is at least one in optuna distributions
        # Optunas upper bound is inclusive, Ray Tune's is exclusive, json we also use exclusive upper bound

        if log:
            if step > 1:  # NOTE: Step is at least 1
                return tune.qlograndint(lower, upper + 1, step)
            return tune.lograndint(lower, upper + 1)

        if step > 1:
            # Ray Tune's qrandint expects exclusive upper bound for integers
            return tune.qrandint(lower, upper + 1, step)

        # Ray Tune's randint uses exclusive upper bound
        return tune.randint(lower, upper + 1)

    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return tune.choice(dist.choices)

    # grid_search, sample_from (included function) and normal distribution, randn, qrandn cannot be expressed by
    # Optuna interface.

    raise TypeError(
        f"Unsupported Optuna distribution type: {type(dist).__name__}. "
        "Supported types are FloatDistribution, IntDistribution, and CategoricalDistribution."
    )


def dict_to_ray_distributions(
    dist_dict: dict[str, dict[str, Any]] | dict[Literal["grid_search"], Sequence[Any]],
) -> ParameterSpace[Any]:
    """Convert a dictionary of Optuna distributions to Ray Tune distributions.

    This function takes a dictionary where keys are parameter names and values
    are Optuna distribution objects, and converts each distribution to its
    corresponding Ray Tune distribution.

    Args:
        dist_dict: A dictionary mapping parameter names to Optuna distribution objects.

    Returns:
        A dictionary mapping parameter names to Ray Tune distributions.
    """
    if "grid_search" in dist_dict:
        if len(dist_dict) != 1:
            _logger.warning("A grid_search dict should only contain the grid_search key, ignoring others keys")
        return {"grid_search": cast("Sequence[Any]", dist_dict["grid_search"])}
    try:
        return optuna_dist_to_ray_distribution(optuna.distributions.json_to_distribution(json.dumps(dist_dict)))
    except (ValueError, TypeError, KeyError):
        # Assume key, value match tune functions
        key, value = next(iter(dist_dict.items()))
        return getattr(tune, key)(**value)  # pyright: ignore[reportCallIssue]


def load_distributions_from_json(
    json_dict: dict[str, Any] | Path | str,
) -> dict[str, ParameterSpace[Any]]:
    if isinstance(json_dict, str):
        json_dict = Path(json_dict)
    if isinstance(json_dict, Path):
        json_path = json_dict
        for attempt in range(5):
            try:
                with json_path.open("r") as f:
                    json_dict = cast("dict[str, Any]", json.load(f))
                break
            except (json.JSONDecodeError, OSError) as e:
                if attempt < 4:
                    _logger.warning(
                        "Failed to load JSON from %s (attempt %d/5): %s. Retrying in 1s...",
                        json_path.resolve(),
                        attempt + 1,
                        e,
                    )
                    time.sleep(1)
                else:
                    raise
        assert isinstance(json_dict, dict)
    return {k: dict_to_ray_distributions(v) for k, v in json_dict.items()}


class TunableSetupMixin(ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]):
    TUNE_PARAMETER_FILE = "experiments/tune_parameters.json"

    def __init__(
        self,
        args: Optional[Sequence[str]] = None,
        *,
        config_files=None,
        load_args=None,
        init_config: bool = True,
        init_param_space: bool = True,
        init_trainable: bool = True,
        parse_args: bool = True,
        trial_name_creator=None,
        change_log_level: Optional[bool] = True,
    ):
        if self.__restored__:
            _logger.debug("Not calling set_tune_parameters as instance is being restored.")
        else:
            self.tune_parameters: dict[str, ParameterSpace[Any] | optuna.distributions.BaseDistribution] = {}
            self.set_tune_parameters()
        super().__init__(args, config_files=config_files, load_args=load_args, init_config=init_config,
                         init_param_space=init_param_space, init_trainable=init_trainable, parse_args=parse_args,
                         trial_name_creator=trial_name_creator, change_log_level=change_log_level)  # fmt: skip

    def add_tune_parameter(self, name: str, param_space: ParameterSpace[Any]) -> None:
        """Add a parameter to the tuning space.

        This method allows adding a new parameter to the tuning space
        dynamically. If the parameter already exists, it will be overwritten.

        Args:
            name: The name of the parameter to add.
            param_space: The Ray Tune parameter space defining the values
                to sample for this parameter.
        """
        self.tune_parameters[name] = param_space
        # TODO: Unfinished, turn to ray Domain

    def get_state(self) -> SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co]:
        state = super().get_state()
        state["tune_parameters"] = self.tune_parameters
        return state

    @classmethod
    def from_saved(
        cls,
        data: SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co],
        *,
        load_class: bool = False,
        init_trainable: bool = True,
        init_config: Optional[bool] = None,
        load_config_files: bool = True,
    ) -> Self:
        new = super().from_saved(
            data,
            load_class=load_class,
            init_trainable=init_trainable,
            init_config=init_config,
            load_config_files=load_config_files,
        )
        new.tune_parameters = data.get("tune_parameters") or {}
        return new

    def create_param_space(self) -> dict[str, Any]:
        param_space = super().create_param_space()
        if not self.args.tune:
            return param_space
        update_hyperparameters(
            self.param_space,
            self.tune_parameters,
            self.args.tune or [],
            num_grid_samples=self.args.num_samples,
            train_batch_size_per_learner=self.args.train_batch_size_per_learner,
        )
        return param_space

    def _load_optuna_from_json(self, json_dict: dict[str, Any]) -> None:
        for param, entry in json_dict.items():
            if isinstance(entry, dict):
                try:
                    optuna_dist = optuna.distributions.json_to_distribution(json.dumps(entry))
                except ValueError as e:
                    _logger.warning(
                        "Could not parse Optuna distribution for parameter '%s' from JSON entry %s: %s",
                        param,
                        entry,
                        e,
                    )
                else:
                    json_dict[param] = optuna_dist

    def set_tune_parameters(self, json_dict: Optional[dict[str, Any] | Path] = None) -> None:
        """Add parameters from a JSON dictionary to the tuning space.

        This method allows adding multiple parameters to the tuning space
        from a JSON-like dictionary. Each key-value pair in the dictionary
        represents a parameter name and its corresponding Ray Tune parameter
        space.

        Args:
            json_dict: A dictionary where keys are parameter names and values
                are Ray Tune parameter spaces.
                If not provided loads the default from :attr:`TUNE_PARAMETER_FILE`.
        """
        # Does not support callable / Domain objects
        if json_dict is None:
            json_dict = Path(self.TUNE_PARAMETER_FILE)
        if isinstance(json_dict, Path):
            if not json_dict.exists():
                _logger.warning(
                    "Tuning parameter file %s does not exist. No tuning parameters were added.",
                    json_dict.resolve(),
                )
                return
            json_dict = json.loads(json_dict.read_text())
            assert isinstance(json_dict, dict)
        for name, param_space in json_dict.items():
            self.add_tune_parameter(name, load_distributions_from_json({name: param_space})[name])


class SetupForDynamicTuning(ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]):
    ...
    # base class that can be expanded in the future


class SetupWithDynamicBuffer(SetupForDynamicTuning[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """Mixin class for dynamic experience buffer sizing in RLlib experiments.

    This mixin adds the capability to dynamically adjust the experience buffer
    (rollout) size during training or hyperparameter optimization. It provides
    both predefined parameter spaces for tuning and automatic callback integration
    for adaptive buffer management.

    The mixin automatically adds the :class:`~ray_utilities.callbacks.algorithm.dynamic_buffer_callback.DynamicBufferUpdate`
    callback when dynamic buffer sizing is enabled, and includes evaluation
    interval adjustment to work properly with the dynamic buffer updates.

    Features:
        - Predefined rollout size parameter space for Ray Tune optimization
        - Automatic callback registration for dynamic buffer updates
        - Integration with dynamic evaluation intervals
        - Supports both grid search and adaptive parameter selection

    Class Attributes:
        rollout_size_sample_space: Ray Tune parameter space with common rollout sizes
            ranging from 32 to 8192 steps, suitable for grid search optimization.

    Note:
        This mixin should be used before other setups that add
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`
        to avoid error logs about missing configuration keys.

    Example:
        class MySetup(SetupWithDynamicBuffer, ExperimentSetupBase):
            config_class = PPOConfig
            algo_class = PPO

    See Also:
        :class:`~ray_utilities.callbacks.algorithm.dynamic_buffer_callback.DynamicBufferUpdate`: Buffer update callback
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`: Evaluation callback
        :class:`SetupWithDynamicBatchSize`: Companion mixin for batch size dynamics
    """

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # Can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks


class SetupWithDynamicBatchSize(SetupForDynamicTuning[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """Mixin class for dynamic batch size adjustment through gradient
    accumulation.

    This mixin enables dynamic batch size control during training by
    utilizing gradient accumulation rather than directly modifying the
    ``train_batch_size_per_learner``. It integrates the
    :class:`~ray_utilities.callbacks.algorithm.dynamic_batch_size.DynamicGradientAccumulation`
    callback when dynamic batch sizing is enabled so the effective batch
    size can be adjusted during training or tuning.

    Features:
        - Dynamic batch size control via gradient accumulation
        - Predefined batch size parameter space for Ray Tune optimization
        - Automatic callback registration for gradient accumulation
        - Integration with dynamic evaluation intervals

    Note:
        Use :class:`SetupWithDynamicBuffer` for direct tuning of rollout
        sizes. This mixin controls effective batch size via gradient
        accumulation.

    Warning:
        The ``batch_size`` tuning values refer to effective batch sizes
        achieved via gradient accumulation, not direct accumulation
        multipliers.

    Examples:
        .. code-block:: python

            class MySetup(SetupWithDynamicBatchSize, ExperimentSetupBase):
                config_class = PPOConfig
                algo_class = PPO

                def create_config(self, args):
                    config = super().create_config(args)
                    if args.dynamic_batch:
                        # Gradient accumulation will be handled automatically
                        pass
                    return config

    See Also:
        :class:`~ray_utilities.callbacks.algorithm.dynamic_batch_size.DynamicGradientAccumulation`:
            Gradient accumulation callback
        :class:`SetupWithDynamicBuffer`:
            Companion mixin for buffer size dynamics
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`:
            Evaluation callback
    """

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for
        callbacks = super()._get_callbacks_from_args(args)
        # TODO: Add dynamic minibatch scaling option
        if args.dynamic_batch:
            callbacks.append(DynamicGradientAccumulation)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks
