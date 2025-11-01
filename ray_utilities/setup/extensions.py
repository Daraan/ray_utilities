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
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

import optuna
from ray import tune
from typing_extensions import Self, deprecated

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
                _logger.warning(
                    "Ray Tune does not support both log scaling and quantization "
                    "for FloatDistribution. Dropping quantization (step=%s).",
                    step,
                )
            return tune.loguniform(lower, upper)

        if step is not None:
            return tune.quniform(lower, upper, step)

        return tune.uniform(lower, upper)

    if isinstance(dist, optuna.distributions.IntDistribution):
        lower = dist.low
        upper = dist.high
        step = dist.step
        log = dist.log

        if log:
            # Ray Tune's lograndint expects exclusive upper bound
            # Optuna enforces step=1 when log=True for IntDistribution
            return tune.lograndint(lower, upper + 1)

        if step > 1:
            # Ray Tune's qrandint expects exclusive upper bound for integers
            return tune.qrandint(lower, upper + 1, step)

        # Ray Tune's randint uses exclusive upper bound
        return tune.randint(lower, upper + 1)

    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return tune.choice(dist.choices)

    raise TypeError(
        f"Unsupported Optuna distribution type: {type(dist).__name__}. "
        "Supported types are FloatDistribution, IntDistribution, and CategoricalDistribution."
    )


def dict_to_ray_distributions(
    dist_dict: dict[str, dict],
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
        return {"grid_search": dist_dict["grid_search"]}  # type: ignore[return-value]
    return optuna_dist_to_ray_distribution(optuna.distributions.json_to_distribution(json.dumps(dist_dict)))


def load_distributions_from_json(
    json_dict: dict[str, Any] | Path,
) -> dict[str, ParameterSpace[Any]]:
    if isinstance(json_dict, str):
        json_dict = Path(json_dict)
    if isinstance(json_dict, Path):
        with json_dict.open("r") as f:
            json_dict = cast("dict[str, Any]", json.load(f))
    return {k: dict_to_ray_distributions(v) for k, v in json_dict.items()}


class TunableSetupMixin(ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]):
    TUNE_PARAMETER_FILE = "experiments/tune_parameters.json"

    def __init__(self, *args, **kwargs):
        if self.__restored__:
            return
        self.tune_parameters: dict[str, ParameterSpace[Any] | optuna.distributions.BaseDistribution] = {}
        super().__init__(*args, **kwargs)
        self.set_tune_parameters()

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
        new.tune_parameters = data.get("tune_parameters", {})
        return new

    def create_param_space(self) -> dict[str, Any]:
        param_space = super().create_param_space()
        if not self.args.tune:
            return param_space
        for parameter in self.args.tune:
            if parameter in self.tune_parameters:
                param_space[parameter] = self.tune_parameters[parameter]
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
            self.add_tune_parameter(name, param_space)


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

    rollout_size_sample_space: ClassVar[ParameterSpace[int]] = tune.grid_search(
        [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 2048 * 3, 8192]  # 4096 * 3, 16384]
    )

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # Can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks

    @classmethod
    def _create_dynamic_buffer_params(cls):
        return deepcopy(cls.rollout_size_sample_space)

    def create_param_space(self) -> dict[str, Any]:
        self._set_dynamic_parameters_to_tune()  # sets _dynamic_parameters_to_tune
        if not self.args.tune or not (
            (add_all := "all" in self._dynamic_parameters_to_tune) or "rollout_size" in self._dynamic_parameters_to_tune
        ):
            return super().create_param_space()
        if not add_all:
            self._dynamic_parameters_to_tune.remove(
                "rollout_size"
            )  # remove before calling super().create_param_space()
        param_space = super().create_param_space()
        # TODO: # FIXME "rollout_size" is not used anywhere
        # however train_batch_size_per_learner is used with the DynamicBatchSize Setup
        # which uses in ints dynamic variant gradient accumulation.
        param_space["rollout_size"] = self._create_dynamic_buffer_params()
        return param_space


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

    batch_size_sample_space: ClassVar[ParameterSpace[int]] = tune.grid_search(
        [64, 128, 256, 512, 1024, 2048, 3072, 4096, 2048 * 3, 8192]  # 4096 * 3, 16384]
    )
    """
    Tune parameter space with batch sizes from 32 to 16384.
    """

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_batch:
            callbacks.append(DynamicGradientAccumulation)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks

    @classmethod
    def _create_dynamic_batch_size_params(cls):
        """Create parameter space for dynamic batch size tuning.

        Returns a deep copy of the class's batch size sample space for use in
        hyperparameter optimization. The returned parameter space contains
        batch size values that will be achieved through gradient accumulation.

        Returns:
            A Ray Tune parameter space containing batch size values for optimization.

        Note:
            The returned parameters represent effective batch sizes achieved through
            gradient accumulation, not the gradient accumulation multiplier values directly.

        Warning:
            This method adds ``batch_size`` parameters to the tuning space, not
            values for direct gradient accumulation control.
        """
        # TODO: control this somehow via args
        return deepcopy(cls.batch_size_sample_space)

    @deprecated("Use less hidden methods to set the dynamic parameters to tune.")
    def create_param_space(self) -> dict[str, Any]:
        self._set_dynamic_parameters_to_tune()  # sets _dynamic_parameters_to_tune
        if not self.args.tune or not (
            (add_all := "all" in self._dynamic_parameters_to_tune) or "batch_size" in self._dynamic_parameters_to_tune
        ):
            return super().create_param_space()
        if not add_all:
            self._dynamic_parameters_to_tune.remove("batch_size")  # remove before calling super().create_param_space()
        # TODO: That in the dynamic variant gradient_accumulation is used and here train_batch_size_per_learner
        # is contradictory naming
        param_space = super().create_param_space()
        param_space["train_batch_size_per_learner"] = self._create_dynamic_batch_size_params()
        return param_space
