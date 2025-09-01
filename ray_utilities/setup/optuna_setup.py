"""
See Also:
    https://docs.ray.io/en/latest/tune/examples/optuna_example.html


.. code-block:: python

    param_space = {
        "x": tune.grid_search([1, 2, 3]),  # not compatible with OptunaSearch
        "y": tune.choice([1, 2, 3]),
    }
    search, stopper = create_search_algo(
        "MyStudy",
        hparams=param_space,
        metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
        mode="max",
        storage=None,
        seed=None,  # required
        pruner=None,  # pass your own Pruner or True for MedianPruner
        # Accepts all other kwargs of OptunaSearch
    )
    compatible_param_space = clean_grid_search_for_optuna(param_space)
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            search_alg=search,
        ),
        run_config=tune.RunConfig(stop=stopper),
        param_space=compatible_param_space,
    )
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Literal, Optional, overload

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import Stopper
from ray.tune.utils import flatten_dict
from ray.tune.search import sample

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN

_logger = logging.getLogger(__name__)


class OptunaSearchWithPruner(OptunaSearch, Stopper):
    def __init__(self, *args, pruner: Optional[optuna.pruners.BasePruner] = None, **kwargs):
        # forward compatibility with OptunaSearch
        if "pruner" in inspect.signature(super().__init__).parameters:
            kwargs["pruner"] = pruner
            _logger.info("OptunaSearch accepts prune parameter, OptunaSearchWithPrune can likely be removed.")
        super().__init__(*args, **kwargs)
        self._pruner = pruner or optuna.pruners.MedianPruner()
        if self._ot_study:  # only if space is used
            self._ot_study.pruner = self._pruner
            self._pruner_set = True
        else:
            self._pruner_set = False
        self._reported_metric_this_step = False
        """Tracks if super().on_trial_result was called from __call__ (Stopper) or later on_trial_result."""

    # Searcher interface:

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        # Allows lazy setup of the pruner
        success = super().set_search_properties(metric, mode, config, **spec)
        if self._ot_study and not self._pruner_set:
            # Set the pruner for the study if it exists
            self._ot_study.pruner = self._pruner
            self._pruner_set = True
        return success

    def on_trial_result(self, trial_id, result: dict[str, Any]) -> None:
        """
        Args:
            trial_id: The ID of the trial.
            result: The result of the trial, which is a flattened dict of the one passed to `tune.report`.
        """
        # When working as a Stopper, the result needs to be reported earlier
        # skipping to report them again
        # Note: This result is a flat_result and the metric is present (even if None)
        if not self._reported_metric_this_step:
            super().on_trial_result(trial_id, result)
        self._reported_metric_this_step = False  # reset for next Stopper.__call__

    # Stopper interface:

    def stop_all(self):
        """Returns true if the experiment should be terminated."""
        return False

    def __call__(self, trial_id: str, result: dict[str, dict[str, Any] | Any]) -> bool:
        """
        Stopper method called by the tuner

        Attention:
            `result` is the result passed to `tune.report`, it is NOT a copy
            and unlike `on_trial_result` is not flattened
        """
        # NOTE:
        # NOTE: on_trial_result is executed AFTER this call, but for should_prune need to report it first
        try:
            super().on_trial_result(trial_id, flatten_dict(result))
        except KeyError as e:
            self._reported_metric_this_step = False  # TODO could set this to current iteration
            _logger.debug(
                "KeyError in Stopper OptunaSearchWithPruner.__call__: %s. "
                "Likely the tracked metric is not present before the first evaluation.",
                e,
            )
            return False
        self._reported_metric_this_step = True  # TODO could set this to current iteration
        trial: optuna.trial.Trial = self._ot_trials[trial_id]
        if trial.should_prune():
            _logger.info("Optuna pruning trial %s", trial_id)
            trial = self._ot_trials[trial_id]
            return True  # TODO: Can we report this to Optuna as well? as raising PruneTrial is not possible
        return False


@overload
def create_search_algo(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: optuna.pruners.BasePruner | Literal[True],
    **kwargs,
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> tuple[OptunaSearchWithPruner, Stopper]: ...


@overload
def create_search_algo(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: Literal[False] = False,
    **kwargs,
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> tuple[OptunaSearch, None]: ...


def create_search_algo(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: optuna.pruners.BasePruner | bool = False,
    **kwargs,
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> tuple[OptunaSearch | OptunaSearchWithPruner, Optional[Stopper]]:
    """
    To be used with TuneConfig in Ray Tune.

    .. code-block:: python

        search, stopper = create_search_algo(
            "MyStudy",
            hparams=param_space,
            metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
            mode="max",
            initial_params=None,
            storage=None,
            seed=None,  # making it required for now to get reproducible results
            pruner=None,
            # evaluated_rewards=None,  # experimental feature
        )

        # When using tune.grid_search in the param_space, it needs to be cleaned first.
        compatible_param_space = clean_grid_search_for_optuna(param_space)
        tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                search_alg=algo,  # <---
                num_samples=num_samples,
            ),
            run_config=tune.RunConfig(
                stop=stopper  # <--- When using pruner
            ),
            param_space=compatible_param_space,
        )
        tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)

    Args:
        hparams: Config to use for hyperparameter tuning. Can contain search spaces and dicts with
            "grid_search" keys.
        max_concurrent: Maximum number of trials at the same time.
        initial_params: Initial parameters to start the search with.
        pruner: If True, uses a MedianPruner, if a pruner is passed, it will be used.
            Otherwise the default of ray's `OptunaSearch` is used.
        kwargs: Forwarded to OptunaSearch, for example `evaluated_rewards`.
    """
    grid_values = {}
    if hparams:
        grid_values = {k: v["grid_search"] for k, v in hparams.items() if isinstance(v, dict) and "grid_search" in v}
    # when using grid search need to use a grid sampler here and adjust other parameters accordingly
    if grid_values:
        if hparams and any(
            isinstance(v, sample.Domain) and (not isinstance(v, sample.Categorical) or len(v.categories) > 1)
            for v in hparams.values()
        ):
            _logger.warning(
                "Grid search is not compatible with sampled parameters: "
                "found continuous distributions or > 1 non-grid Categoricals. "
                "Turning categoricals to grid and sampling ONE value for continuous parameters. "
                "This warning is common if '--tune' and '--env_seeding_strategy sequential' is used. "
                "In this case change the env_seeding_strategy to 'same', 'constant' or 'random'."
            )
            for k, v in hparams.items():
                if isinstance(v, sample.Domain) and not isinstance(v, sample.Categorical):
                    # sample ONE value for continuous parameters
                    hparams[k] = tune.choice([v.sample()])
                    _logger.debug("Converted continuous parameter %s to a single choice: %s", k, hparams[k])
                elif isinstance(v, sample.Categorical) and len(v.categories) > 1:
                    # add single category to grid search
                    grid_values[k] = v.categories
                    _logger.debug("Converted categorical parameter %s to grid search: %s", k, grid_values[k])
        sampler = optuna.samplers.GridSampler(grid_values, seed=seed)
        # TODO: This covers grid but what if I want TPESampler as well?
    else:
        sampler = None  # TPE sampler
    if not pruner:
        searcher = OptunaSearch(
            study_name=study_name,
            points_to_evaluate=initial_params,
            mode=mode,
            metric=metric,
            storage=storage,
            seed=seed,
            sampler=sampler if grid_values else None,
            **kwargs,
        )
        stopper = None
        return searcher, stopper
    if pruner is True:
        pruner = optuna.pruners.MedianPruner()
    searcher = OptunaSearchWithPruner(
        study_name=study_name,
        points_to_evaluate=initial_params,
        mode=mode,
        metric=metric,
        storage=storage,
        seed=seed,
        sampler=sampler if grid_values else None,
        pruner=pruner,
    )
    stopper = searcher
    return searcher, stopper


def clean_grid_search_for_optuna(
    param_space: dict[str, Any | dict[Literal["grid_search"], Any]] | None = None,
) -> dict[str, Any]:
    """
    Removes grid_search parameters from the search space and replaced them with `tune.choice`.
    A grid search will be performed like expected when the search algorithm uses a
    `optuna.samplers.GridSampler` with the respective values.
    """
    if param_space is None:
        return {}
    return {
        k: tune.choice(v["grid_search"]) if isinstance(v, dict) and "grid_search" in v else v
        for k, v in param_space.items()
    }
