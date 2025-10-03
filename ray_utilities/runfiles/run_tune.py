"""High-level utilities for running Ray Tune experiments and local debugging.

This module provides the main entry points for executing Ray RLlib experiments,
both through Ray Tune for distributed hyperparameter optimization and locally
for debugging and testing. It handles the setup, execution, and result processing
of machine learning experiments.

The module bridges experiment setup classes with actual execution, providing
both production-ready distributed training capabilities and development-friendly
local execution modes for rapid iteration and debugging.

Key Components:
    - :func:`run_tune`: Main function for running distributed experiments with Ray Tune
    - Local execution utilities for debugging without Ray Tune overhead
    - Automatic parameter sampling and test mode support
    - Integration with experiment setup classes and trainable functions

This module is typically used as the final step in experiment configuration,
after setting up algorithms, hyperparameters, and training configurations.
"""

from __future__ import annotations

import logging
import sys
import time
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.misc import raise_tune_errors
from ray_utilities.random import seed_everything
from ray_utilities.training.default_class import TrainableBase

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.result_grid import ResultGrid

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup import ExperimentSetupBase
    from ray_utilities.training.default_class import TrainableBase
    from ray_utilities.typing import TestModeCallable
    from ray_utilities.typing.trainable_return import TrainableReturnData

logger = logging.getLogger(__name__)

_SetupT = TypeVar("_SetupT", bound="ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]")
"""A TypeVar for ExperimentSetupBase with specific type parameters."""


def _run_without_tuner(
    setup: _SetupT,
    trainable: type[TrainableBase[Any, Any, Any]] | Callable[[dict], TrainableReturnData],
    test_mode_func: Optional[TestModeCallable[_SetupT]] = None,
) -> TrainableReturnData:
    """Test and debug mode function that does not run a Tuner instance but locally."""
    # will spew some warnings about train.report
    func_name = getattr(test_mode_func, "__name__", repr(test_mode_func)) if test_mode_func else trainable.__name__
    print(f"-- FULL TEST MODE running {func_name} --")
    logger.info("-- FULL TEST MODE --")
    import ray.tune.search.sample  # noqa: PLC0415 # import lazy

    # Sample the parameters when not entering via tune
    params = {
        k: v.sample() if isinstance(v, ray.tune.search.sample.Domain) else v for k, v in setup.param_space.items()
    }
    setup.param_space.update(params)
    # Possibly set RAY_DEBUG=legacy
    if isclass(trainable):
        # If trainable is a class, instantiate it with the sampled parameters
        trainable_instance = trainable(setup.sample_params())
        logger.warning(
            "[TESTING] Using a Trainable class, without a Tuner relying on a stopper or 'done' return value."
        )

        tuner = setup.create_tuner()
        assert tuner._local_tuner
        stopper = tuner._local_tuner.get_run_config().stop
        while True:
            result = trainable_instance.train()
            if callable(stopper):
                # If stop is a callable, call it with the result
                if stopper("NA", result):  # pyright: ignore[reportArgumentType]
                    break
            # If stop is not a callable, check if it is reached
            elif result.get("done", False):
                break
        from ray_utilities.postprocessing import create_log_metrics  # noqa: PLC0415, circular import

        if "config" in result:
            result["config"].setdefault("_train_batch_size_per_learner", setup.config.train_batch_size_per_learner)
        return create_log_metrics(result, log_stats=setup.args.log_stats)
    if test_mode_func:
        return test_mode_func(trainable, setup)
    return trainable(setup.sample_params())


def run_tune(
    setup: _SetupT | type[_SetupT],
    test_mode_func: Optional[TestModeCallable[_SetupT]] = None,
    *,
    raise_errors: bool = True,
) -> TrainableReturnData | ResultGrid:
    """Execute hyperparameter tuning for Ray RLlib experiments.

    This function orchestrates the complete tuning workflow, including experiment setup,
    test mode handling, seed configuration, and result management. It provides the main
    entry point for running hyperparameter optimization experiments.

    Args:
        setup: Experiment setup instance or class. If a class is provided, it will be
            instantiated with default parameters. Must inherit from :class:`ExperimentSetupBase`.
        test_mode_func: Optional function to execute in test mode instead of full tuning.
            Should accept the trainable and setup as arguments and return results.
        raise_errors: Whether to raise exceptions encountered during tuning. If False,
            errors are logged but execution continues.

    Returns:
        Ray Tune results from the hyperparameter optimization process, or test results
        if running in test mode. Returns a :class:`ray.tune.ResultGrid` for full tuning
        or :class:`TrainableReturnData` for test mode execution.

    Examples:
        Basic usage with a setup class:

        >>> from ray_utilities.setup import PPOSetup
        >>> results = run_tune(PPOSetup)

        With a configured setup instance:

        >>> setup = PPOSetup()
        >>> setup.config.lr = 0.001
        >>> results = run_tune(setup)

        Using test mode:

        >>> def test_func(trainable, setup):
        ...     return trainable(setup.sample_params())
        >>> results = run_tune(PPOSetup, test_mode_func=test_func)

    Note:
        - Test mode is activated when ``args.test=True`` and ``args.not_parallel=True``
        - Offline experiment tracking uploads are handled automatically after completion.
          When ``args.comet`` or ``args.wandb`` are set to ``offline+upload``.
        - Seeds are automatically configured across Ray, torch, and numpy when specified

    See Also:
        :class:`ray_utilities.setup.ExperimentSetupBase`: Base class for experiment setups
        :func:`_run_without_tuner`: Test mode execution function
        :func:`ray.tune.run`: Underlying Ray Tune execution
    """
    # full parser example see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61

    if isinstance(setup, type):
        setup = setup()
    args = setup.get_args()
    if args.seed is not None:
        logger.debug("Setting seed to %s", args.seed)
        _next_seed = seed_everything(env=None, seed=args.seed, torch_manual=True, torch_deterministic=True)
        if setup.config.seed != args.seed:
            with setup.open_config():  # config is frozen
                setup.config.seed = args.seed
    trainable = setup.trainable or setup.create_trainable()

    # -- Test --
    if args.test and args.not_parallel:
        return _run_without_tuner(setup=setup, trainable=trainable, test_mode_func=test_mode_func)
    # Use tune.with_parameters to pass large objects to the trainable

    results = None
    fit_error = None
    fallback_error = None
    tuner = setup.create_tuner()
    try:
        results = tuner.fit()
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            # In case we do not want to upload results
            logger.info(
                "Tuning interrupted. Will try to upload gathered results now. "
                "Waiting 2 sec before continuing... "
                "Press Ctrl + C again to exit immediately."
            )
            time.sleep(2)
        fit_error = e
        logger.error("Error occurred during tuning: %s", fit_error)
        try:
            results = tuner.get_results()
        except (RuntimeError, Exception) as e2:
            logger.error("Error occurred while getting results: %s", e)
            fallback_error = e2
    if not results:
        logger.warning("No results returned from the tuner.")
    setup.upload_offline_experiments(results, tuner)
    if raise_errors and results is not None:
        raise_tune_errors(results)
    if fallback_error and fit_error:
        # Use Base in case of KeyboardInterrupt
        raise BaseExceptionGroup(
            f"Encountered an error {fit_error} and could not call get_results {fallback_error!r}",
            [fit_error, fallback_error],
        )
    if fit_error:
        raise fit_error
    assert results is not None

    return results


if __name__ == "__main__":
    # For testing purposes, run the function with a dummy setup
    from ray_utilities.setup import PPOSetup

    dummy_setup = PPOSetup()
    run_tune(dummy_setup)
