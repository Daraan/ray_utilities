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
import weakref
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Dict, Literal, Optional, cast, overload

from frozendict import frozendict
from ray import tune
from ray.tune.search import sample
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import Stopper
from ray.tune.utils import flatten_dict

from ray_utilities.constants import DEFAULT_EVAL_METRIC, EVAL_METRIC_RETURN_MEAN, NEW_LOG_EVAL_METRIC
from ray_utilities.misc import deep_freeze, new_log_format_used
from ray_utilities.nice_logger import ImportantLogger

try:
    from ray.tune.experiment.config_parser import _create_trial_from_spec as _original_create_trial_from_spec
except ImportError:
    _original_create_trial_from_spec = None

try:
    import optuna
except ModuleNotFoundError:
    if TYPE_CHECKING:
        import optuna
    else:
        optuna = cast("Any", ModuleNotFoundError)

# Needs to be pickable
_NOT_PROVIDED: Literal["_NOT_PROVIDED"] = "_NOT_PROVIDED"


_logger = logging.getLogger(__name__)

_original_limiter_on_trial_complete = ConcurrencyLimiter.on_trial_complete


_patched_trial_mappings_new_to_old: dict[str, str] = {}
_patched_trial_mappings_old_to_new: dict[str, str] = {}

_patched_searchers: weakref.WeakSet["OptunaSearch | OptunaSearchWithPruner"] = weakref.WeakSet()
_monkey_patch_applied = False


# need to be able to pickle this function
def _patch_searcher_on_trial_complete(
    trial_id: str,
    result: Optional[Dict] = None,
    error: bool = False,  # noqa: FBT001, FBT002
    *args,
    org_function: Callable[Concatenate[str, Optional[Dict], bool, ...], None] | None = None,
    searcher: OptunaSearch | OptunaSearchWithPruner | None = None,
    **kwargs,
) -> None:
    """
    Handles trial completion for OptunaSearch and OptunaSearchWithPruner, ensuring correct trial ID mapping
    when monkey-patching trial completion logic for compatibility with systems expecting unique trial IDs.

    Args:
        searcher: The OptunaSearch or OptunaSearchWithPruner instance.
        org_function: The original on_trial_complete function to call.
        trial_id: The ID of the trial being completed.
        result: Optional result dictionary for the trial.
        error: Boolean indicating if the trial ended with an error.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    if trial_id in _patched_trial_mappings_old_to_new:
        if org_function is None:
            _logger.error("org_function should not be None here, this should have been added by partial")
        else:
            try:
                org_function(_patched_trial_mappings_old_to_new[trial_id], result, error, *args, **kwargs)
            except KeyError:
                pass
            else:
                return
    try:
        assert org_function is not None, "org_function should not be None here, this should have been added by partial"
        org_function(trial_id, result, error, *args, **kwargs)
    except RecursionError:
        assert org_function is not None
        nested_func = org_function == _patch_searcher_on_trial_complete
        nested_func_matches_name = org_function.__name__ == "_patch_searcher_on_trial_complete"
        _logger.error(
            "RecursionError occurred in _patch_searcher_on_trial_complete, Calling on_trial_complete via parent class."
            "patcher is calling itself (partial has been used) (same function %s, same name %s) %s",
            nested_func,
            nested_func_matches_name,
            error,
        )
    except AssertionError as e:
        if org_function is not None:
            _logger.error("AssertionError in _patch_searcher_on_trial_complete: %s", e)
            raise
        _logger.error("org_function is None in _patch_searcher_on_trial_complete: %s", e)
    else:
        return
    if searcher is not None:
        type(searcher).on_trial_complete(searcher, trial_id, result, error, *args, **kwargs)  # pyright: ignore[reportArgumentType]
        return
    _logger.error("searcher argument in patched function is None, this should have been added by partial")


def patch_limiter_on_trial_complete(
    self: ConcurrencyLimiter,
    trial_id: str,
    result: Optional[Dict] = None,
    error: bool = False,  # noqa: FBT001, FBT002
    *args,
    **kwargs,  # noqa: FBT001, FBT002
):
    if (
        trial_id not in self.live_trials
        and (swapped_trial_id := _patched_trial_mappings_new_to_old.get(trial_id, None)) is not None
    ):
        trial_id = swapped_trial_id  # exchange back to original trial_id
    return _original_limiter_on_trial_complete(self, trial_id, result, error, *args, **kwargs)


def _monkey_patch_trial_creation(searcher: OptunaSearch | OptunaSearchWithPruner) -> None:
    """
    When using a SearchAlgo trial_ids are not suffixed with a count.
    This monkey patch adds a count to the trial_id to make it unique and compatible with
    systems that expect unique trial_ids like comet_ml.
    """
    if searcher in _patched_searchers:
        # Patch a different searcher?
        _logger.warning(
            "Monkey patch for OptunaSearch trial creation already applied This function should not be called twice. "
            "Patching the passed searcher only."
        )
    if _original_create_trial_from_spec:  # and not _monkey_patch_applied:
        _logger.info("Patching ray.tune.search.search_generator to yield trials in <trial_id>_<count> format.")
        import ray.tune.search.search_generator  # noqa: PLC0415

        count = 0

        def patched_create_trial_from_spec(*args, **kwargs):
            """Insert count into trial_id for it to have a <trial_id>_<count> format."""
            if _original_create_trial_from_spec is None:
                return None  # This should never been called then
            trial = _original_create_trial_from_spec(*args, **kwargs)
            if "_" in kwargs.get("trial_id", "_"):
                return trial
            try:
                nonlocal count
                if not hasattr(searcher, "_base_trial_id") or searcher._base_trial_id is None:  # pyright: ignore[reportAttributeAccessIssue]
                    searcher._base_trial_id = kwargs.get("trial_id", trial.trial_id)  # pyright: ignore[reportAttributeAccessIssue]
                    _logger.info(
                        "Setting base_trial_id to %s. Will use for all trials in this experiment",
                        searcher._base_trial_id,  # pyright: ignore[reportAttributeAccessIssue]
                    )
                else:
                    _logger.info(
                        "Discarding trial id %s for trial_id with count %s_<some_count>",
                        trial.trial_id,
                        searcher._base_trial_id,  # pyright: ignore[reportAttributeAccessIssue]
                    )
                new_trial_id = f"{searcher._base_trial_id}_{count:05d}"  # pyright: ignore[reportAttributeAccessIssue]
                _patched_trial_mappings_new_to_old[new_trial_id] = trial.trial_id
                _patched_trial_mappings_old_to_new[trial.trial_id] = new_trial_id
                trial.trial_id = new_trial_id
                if trial.trial_name_creator:
                    trial.custom_trial_name = trial.trial_name_creator(trial)
                if trial.trial_dirname_creator:
                    trial.custom_dirname = trial.trial_dirname_creator(trial)
                count += 1
            except Exception:
                _logger.exception("Error while patching trial_id:")
                raise
            return trial

        ray.tune.search.search_generator._create_trial_from_spec = patched_create_trial_from_spec  # pyright: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]

    # patch_trial_cleanup
    if ConcurrencyLimiter.on_trial_complete == _original_limiter_on_trial_complete:
        ConcurrencyLimiter.on_trial_complete = patch_limiter_on_trial_complete

    # But searcher needs updated id again
    if searcher not in _patched_searchers:
        original_searcher_complete = searcher.on_trial_complete
        if isinstance(original_searcher_complete, partial) and (
            original_searcher_complete.func == _patch_searcher_on_trial_complete
            or original_searcher_complete.func.__name__ == "_patch_searcher_on_trial_complete"
        ):
            _logger.warning("OptunaSearch.on_trial_complete already patched. This should not happen.")
        else:
            searcher.on_trial_complete = partial(
                _patch_searcher_on_trial_complete, org_function=original_searcher_complete, searcher=searcher
            )
    _patched_searchers.add(searcher)


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


class OptunaSearchWithPruner(OptunaSearch, Stopper):
    def __init__(
        self,
        *args,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        initial_seed: int | Literal["_NOT_PROVIDED"] | None = _NOT_PROVIDED,
        **kwargs,
    ):
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
        self._base_trial_id = None
        self._suggested_params: set[frozendict] = set()
        self._initial_seed = initial_seed

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
            if trial_id not in self._ot_trials:
                _logger.error(
                    "Trial id %s not found in Optuna trials when reporting result. "
                    "This might be related to restored experiments. Valid ids are: %s",
                    trial_id,
                    list(self._ot_trials.keys()),
                )
                self._reported_metric_this_step = False
                return
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
                "Likely the tracked metric is not present before the first evaluation. "
                "or this class is used as a stopper without using it as a searcher at the same time.",
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

    def suggest(self, trial_id: str) -> Dict | None:
        if self._base_trial_id is None:
            self._base_trial_id = trial_id.rsplit("_", 1)[0]
            trial_id = self._base_trial_id + "_00000"
        else:
            # discard trial id and use counter
            new_trial_id = self._base_trial_id + f"_{len(self._ot_trials):05d}"
            _logger.info("Discarding suggested trial_id %s for and replacing with %s", trial_id, new_trial_id)
            trial_id = new_trial_id
        suggestion = super().suggest(trial_id)
        # Check minibatch <= batch_size constraint
        if suggestion is not None and "minibatch_size" in suggestion:
            if "train_batch_size_per_learner" not in suggestion:
                ImportantLogger.important_warning(
                    _logger,
                    "Sampling minibatch_size without knowing train_batch_size_per_learner. Constraint minibatch_size <= batch_size",
                )
            elif suggestion["minibatch_size"] > suggestion["train_batch_size_per_learner"]:
                return None  # skip for this step
        if suggestion is not None and self._initial_seed is not None:
            frozen_suggestion = deep_freeze(suggestion)
            counter = 1
            while frozen_suggestion in self._suggested_params:
                # modify env seed
                if self._initial_seed == _NOT_PROVIDED:
                    self._initial_seed = 42
                suggestion["env_seed"] = (self._initial_seed, counter)
                counter += 1
                frozen_suggestion = deep_freeze(suggestion)
            if counter > 1:
                _logger.info("Search suggested param multiple times, adjusting env_seed to %s", suggestion["env_seed"])
            self._suggested_params.add(frozendict(suggestion))
        return suggestion


@overload
def create_optuna_searcher(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric: str = EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: optuna.pruners.BasePruner | Literal[True],
    pruner_warmup_steps: int = 128,
    pruner_min_trials: int = 3,
    **kwargs,
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> tuple[OptunaSearchWithPruner, Stopper]: ...


@overload
def create_optuna_searcher(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric: str = EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: Literal[False] = False,
    **kwargs,
    # evaluated_rewards: Optional[list[float]] = None,  # experimental feature
) -> tuple[OptunaSearch, None]: ...


def create_optuna_searcher(
    study_name: str,
    *,
    hparams: Optional[dict[str, Any | dict[Literal["grid_search"], Any]]],
    metric: Optional[str | DEFAULT_EVAL_METRIC] = EVAL_METRIC_RETURN_MEAN,  # flattened key
    mode: str | list[str] | None = "max",
    initial_params: Optional[list[dict[str, Any]]] = None,
    storage: Optional[optuna.storages.BaseStorage] = None,
    seed: int | None,
    pruner: optuna.pruners.BasePruner | bool = False,
    pruner_warmup_steps: int = 128,
    pruner_min_trials: int = 3,
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
    if metric is DEFAULT_EVAL_METRIC:
        metric = NEW_LOG_EVAL_METRIC if new_log_format_used() else EVAL_METRIC_RETURN_MEAN
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
        _monkey_patch_trial_creation(searcher)
        searcher._base_trial_id = None  # pyright: ignore[reportAttributeAccessIssue]
        return searcher, stopper
    if pruner is True:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruner_warmup_steps, n_min_trials=pruner_min_trials)
    from ray_utilities.tune.searcher.optuna_searcher import OptunaSearchWithPruner  # noqa: PLC0415

    searcher = OptunaSearchWithPruner(
        study_name=study_name,
        points_to_evaluate=initial_params,
        mode=mode,
        metric=metric,
        storage=storage,
        seed=None if sampler else seed,
        sampler=sampler if grid_values else None,
        pruner=pruner,
    )
    stopper = searcher
    _monkey_patch_trial_creation(searcher)
    return searcher, stopper
