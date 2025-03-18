from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, TypeVar

from ray.tune.result_grid import ResultGrid

from ray_utilities import seed_everything

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig, Algorithm
    from ray.tune.result_grid import ResultGrid

    from ray_utilities.config import DefaultArgumentParser, ExperimentSetupBase
    from ray_utilities.typing import TestModeCallable
    from ray_utilities.typing.trainable_return import TrainableReturnData

logger = logging.getLogger(__name__)

_SetupT = TypeVar("_SetupT", bound="ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]")


def run_tune(
    setup: _SetupT | type[_SetupT], test_mode_func: Optional[TestModeCallable[_SetupT]] = None
) -> TrainableReturnData | ResultGrid:
    """
    Runs the tuning process for a given experiment setup.

    Args:
        setup: The experiment setup containing the configuration and trainable.
        test_mode_func: A callable function to execute in test mode.
            This function should take the trainable and args as parameters.

    Returns:
        ResultGrid: The results of the tuning process.

    Notes:
        - If `args.test` is True and `args.not_parallel` is True, the function will run the `test_mode_func`,
          without parallelization and tuner setup.
        - Offline experiments will be uploaded after the tuning process.
          NOT FOR WANDB currently!
    """
    # full parser example see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61

    if isinstance(setup, type):
        setup = setup()
    args = setup.get_args()
    if args.seed is not None:
        logger.debug("Setting seed to %s", args.seed)
        seed_everything(env=None, seed=args.seed, torch_manual=True, torch_deterministic=True)
        setup.config.seed = args.seed
    trainable = setup.trainable

    # -- Test --
    if args.test and args.not_parallel and test_mode_func:
        # will spew some warnings about train.report
        func_name = getattr(test_mode_func, "__name__", repr(test_mode_func))
        print(f"-- FULL TEST MODE running {func_name} --")
        logger.info("-- FULL TEST MODE --")
        # Possibly set RAY_DEBUG=legacy
        return test_mode_func(trainable, setup)

    # Use tune.with_parameters to pass large objects to the trainable
    tuner = setup.create_tuner()
    results = tuner.fit()
    setup.upload_offline_experiments()
    return results
