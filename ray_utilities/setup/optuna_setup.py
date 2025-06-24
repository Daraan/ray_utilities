"""
See Also:
    https://docs.ray.io/en/latest/tune/examples/optuna_example.html
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from ray.tune.search.optuna import OptunaSearch

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
import optuna


def __example_conditional_search_space(trial: optuna.Trial) -> Optional[dict[str, Any]]:
    """Define-by-run function to construct a conditional search space.

    Ensure no actual computation takes place here. That should go into
    the trainable passed to ``Tuner()`` (in this example, that's
    ``objective``).

    For more information, see https://optuna.readthedocs.io/en/stable\
    /tutorial/10_key_features/002_configurations.html

    Args:
        trial: Optuna Trial object

    Returns:
        Dict containing constant parameters or None

    Use OptunaSearch(space=_conditional_search_space) and omit

    tuner = tune.Tuner(
        ...,
        # param_space=search_space, # do not use in this case
    )
    """
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # Define-by-run allows for conditional search spaces.
    if activation == "relu":
        trial.suggest_float("width", 0, 20)
        trial.suggest_float("height", -100, 100)
    else:
        trial.suggest_float("width", -1, 21)
        trial.suggest_float("height", -101, 101)

    # Return all constants in a dictionary.
    return {"steps": 100}


def create_search_algo(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric=EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,  # making it required for now to get reproducible results
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> OptunaSearch:
    """
    To be used with TuneConfig in Ray Tune.

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            search_alg=algo, # <---
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

    Args:
        max_concurrent: Maximum number of trials at the same time.
        initial_params: Initial parameters to start the search with.
    """
    grid_values = {}
    if hparams:
        grid_values = {k: v["grid_search"] for k, v in hparams.items() if isinstance(v, dict) and "grid_search" in v}
    if grid_values:
        sampler = optuna.samplers.GridSampler(grid_values)
        # TODO: This covers grid but what if I want TPESampler as well?
    else:
        sampler = None
    algo = OptunaSearch(
        study_name=study_name,
        points_to_evaluate=initial_params,
        mode=mode,
        metric=metric,
        storage=storage,
        seed=seed,
        sampler=sampler if grid_values else None,
    )
    # when using grid search need to use a grid sampler here
    return algo
