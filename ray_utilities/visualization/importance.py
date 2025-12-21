from __future__ import annotations

import logging
import math
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain, product
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import optuna.importance
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import Literal

from experiments.create_tune_parameters import write_distributions_to_json
from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.misc import cast_numpy_numbers, round_floats
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.testing_utils import remote_breakpoint
from ray_utilities.visualization._common import Placeholder
from ray_utilities.visualization.data import (
    SubmissionRun,
    check_metric_backport,
    clean_placeholder_keys,
    combine_df,
    get_and_check_group_stat,
    get_epoch_stats,
    get_run_directories_from_submission,
    get_running_experiments,
    get_runs_from_submission,
    ifill,
    load_excludes,
    load_run_data,
    save_run_data,
)
from ray_utilities.visualization.data import ifill as data_ifill

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
except ImportError:
    RandomForestRegressor = None  # type: ignore[assignment]
    permutation_importance = None  # type: ignore[assignment]
try:
    from numpy import trapezoid as _np_trapezoid
except ImportError:
    from numpy import trapz as _np_trapezoid  # type: ignore[assignment]

ImportanceMethod = str
RegressorFactory = Callable[[], Any]
MetricKey = str | tuple[Any, ...]


logger = logging.getLogger(__name__)

_REPORT_INTERVAL = 32
global DEBUG
DEBUG = False

from functools import partial

remote_breakpoint = partial(remote_breakpoint, port=5681)


@dataclass
class EpochSummary:
    """Container holding per-epoch aggregates and metadata."""

    table: pd.DataFrame
    feature_columns: list[str]
    metric_columns: dict[str, str]
    metadata_columns: list[str] = field(default_factory=list)


@dataclass
class ImportanceResult:
    """Stores the fitted model and importance scores for one experiment."""

    summary: EpochSummary
    model: Any
    feature_matrix: pd.DataFrame
    target: pd.Series
    feature_importances: pd.Series
    permutation_importances: pd.DataFrame | None
    category_mappings: dict[str, dict[int, Any]]


def experiment_importance(
    experiment_paths: str | Path | Sequence[str | Path],
    metric: MetricKey,
    *,
    method: ImportanceMethod = "permutation",
    target: str = "last",
    model_factory: RegressorFactory | None = None,
    n_repeats: int = 16,
    random_state: int | None = None,
    min_group_size: int = 4,
) -> dict[str, ImportanceResult]:
    """Compute hyperparameter importance scores for one or many PBT experiments.

    Args:
        experiment_paths: Single experiment path or iterable of experiment directories.
        metric: Column name (or tuple key) that should be summarised per epoch.
        method: Importance method to compute (``"permutation"``, ``"feature"`` or ``"both"``).
        target: Epoch summary column to model (``"mean"``, ``"last"``, ``"delta"``, ``"auc"``).
        model_factory: Factory returning an unfitted sklearn regressor. Defaults to
            :class:`~sklearn.ensemble.RandomForestRegressor` when available.
        n_repeats: Number of shuffles used for permutation importances.
        random_state: Optional random state for reproducibility.
        min_group_size: Minimum number of rows required per ``pbt_epoch`` group.

    Returns:
        Mapping from experiment path (string) to :class:`ImportanceResult` instances.
    """
    paths = _normalise_paths(experiment_paths)
    results: dict[str, ImportanceResult] = {}
    combined_records: list[pd.DataFrame] = []
    reference_summary: EpochSummary | None = None

    for path in paths:
        logger.debug("Processing experiment at %s", path)
        # We do not want load trials that are currently running
        run_frames = load_run_data(path)
        if not run_frames:
            logger.warning("No runs found for experiment at %s", path)
            continue
        if not isinstance(run_frames, pd.DataFrame):
            combined_df = combine_df(run_frames)
            save_run_data(path, combined_df)
        else:
            combined_df = run_frames
        summary = summarise_epochs(combined_df, metric, min_group_size=min_group_size)
        if summary.table.empty:
            logger.warning("Skipping %s - no epoch summaries left after filtering", path)
            continue
        try:
            target_column = summary.metric_columns[target]
        except KeyError as err:  # pragma: no cover - defensive
            raise ValueError(f"Unknown target '{target}'. Available: {list(summary.metric_columns)}") from err

        importance = _compute_importances(
            summary,
            target_column,
            method=method,
            model_factory=model_factory,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        results[str(path)] = importance
        combined_records.append(summary.table)
        reference_summary = importance.summary

    if len(combined_records) > 1 and reference_summary is not None:
        logger.debug("Computing global importance across %s experiments", len(combined_records))
        combined_table = pd.concat(combined_records, axis=0, ignore_index=True)
        combined_features = [column for column in reference_summary.feature_columns if column in combined_table.columns]
        combined_summary = EpochSummary(
            table=combined_table,
            feature_columns=combined_features,
            metric_columns=reference_summary.metric_columns,
            metadata_columns=reference_summary.metadata_columns,
        )
        importance_all = _compute_importances(
            combined_summary,
            combined_summary.metric_columns[target],
            method=method,
            model_factory=model_factory,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        results["__all__"] = importance_all

    return results


def summarise_epochs(
    df: pd.DataFrame,
    metric: MetricKey,
    *,
    min_group_size: int = 4,
) -> EpochSummary:
    """Create per-epoch aggregates and hyperparameter snapshots.

    Args:
        df: Combined DataFrame returned by :func:`combine_df`.
        metric: Column key referencing the reward/metric column.
        min_group_size: Minimum rows required to keep a ``pbt_epoch`` segment.

    Returns:
        :class:`EpochSummary` describing the dataset used for importance modelling.
    """
    if df.empty:
        return EpochSummary(pd.DataFrame(), [], {})

    depth = df.columns.nlevels

    def _ifill(*cols: Any) -> tuple[Any, ...]:
        return data_ifill(*cols, n=depth)

    metric_key = _normalise_column_key(df, metric)
    metric_label = _flatten_column_name(metric_key)
    metric_column_name = f"{metric_label}"

    try:
        pbt_epoch_key = _ifill("config", "pbt_epoch")
    except Exception as exc:  # pragma: no cover - defensive
        raise KeyError("Combined frame is missing the 'pbt_epoch' column") from exc

    if pbt_epoch_key not in df.columns:
        raise KeyError("Combined frame is missing the 'pbt_epoch' column")

    if metric_key not in df.columns:
        raise KeyError(f"Metric column '{metric}' not found in combined dataframe")

    group_df = pd.DataFrame(
        {
            "run_id": df.index.get_level_values("run_id"),
            "pbt_epoch": df[pbt_epoch_key].to_numpy(),
            "current_step": df["current_step"].to_numpy(),
            metric_column_name: df[metric_key].to_numpy(),
        },
        index=df.index,
    )

    hyperparam_columns = _collect_hyperparameters(df)
    feature_columns: list[str] = []
    for column, label in hyperparam_columns:
        group_df[label] = df[column].to_numpy()
        feature_columns.append(label)

    group_df = group_df.dropna(subset=["pbt_epoch", metric_column_name])

    grouped = group_df.groupby(["run_id", "pbt_epoch"], sort=True, group_keys=False)
    records: list[dict[str, Any]] = []

    for (run_id, epoch), segment in grouped:
        if len(segment) < min_group_size:
            continue
        ordered_segment = segment.sort_values("current_step")
        values = ordered_segment[metric_column_name].astype(float)
        steps = ordered_segment["current_step"].astype(float)
        if values.isna().all():
            continue
        first_value = values.iloc[0]
        last_value = values.iloc[-1]
        mean_value = float(values.mean())
        area = float(_np_trapezoid(values.to_numpy(), x=steps.to_numpy()))
        record = {
            "run_id": run_id,
            "pbt_epoch": int(epoch),
            "step_start": float(steps.iloc[0]),
            "step_end": float(steps.iloc[-1]),
            "step_interval": float(steps.iloc[-1] - steps.iloc[0]),
            f"{metric_label}__first": float(first_value),
            f"{metric_label}__last": float(last_value),
            f"{metric_label}__mean": mean_value,
            f"{metric_label}__delta": float(last_value - first_value),
            f"{metric_label}__auc": area,
            "samples": len(segment),
        }
        for _column, label in hyperparam_columns:
            value = ordered_segment[label].dropna()
            if value.empty:
                record[label] = np.nan
                continue
            first = value.iloc[0]
            if _is_scalar(first):
                record[label] = first
            else:
                record[label] = str(first)
        records.append(record)

    summary_frame = pd.DataFrame.from_records(records)
    metadata_columns = ["run_id", "pbt_epoch", "step_start", "step_end", "step_interval", "samples"]
    metric_columns = {
        "first": f"{metric_label}__first",
        "last": f"{metric_label}__last",
        "mean": f"{metric_label}__mean",
        "delta": f"{metric_label}__delta",
        "auc": f"{metric_label}__auc",
    }

    return EpochSummary(summary_frame, feature_columns, metric_columns, metadata_columns)


def _compute_importances(
    summary: EpochSummary,
    target_column: str,
    *,
    method: ImportanceMethod = "permutation",
    model_factory: RegressorFactory | None = None,
    n_repeats: int = 16,
    random_state: int | None = None,
) -> ImportanceResult:
    if summary.table.empty:
        raise ValueError("Cannot compute importances on an empty summary table")

    available_features = summary.feature_columns
    if not available_features:
        raise ValueError("Summary table contains no hyperparameter columns to analyse")

    raw_features = summary.table[available_features]
    target = summary.table[target_column]

    encoded_features, category_mappings = _encode_features(raw_features)
    frame = pd.concat([encoded_features, target], axis=1).dropna()
    if frame.empty:
        raise ValueError("No rows left after dropping NaNs - check metric selection")

    y = frame[target_column]
    X = frame.drop(columns=[target_column])

    if model_factory is None:
        if RandomForestRegressor is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "scikit-learn is required for importance computation. Install it or provide a custom model factory."
            )

        def _default_model_factory() -> Any:
            assert RandomForestRegressor is not None  # for type checkers
            return RandomForestRegressor(
                n_estimators=256,
                random_state=random_state,
                n_jobs=-1,
                min_samples_leaf=2,
            )

        model_factory = _default_model_factory

    model = model_factory()
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns, name="feature_importance")
    else:
        feature_importances = pd.Series(index=X.columns, dtype=float, name="feature_importance")

    perm_result: pd.DataFrame | None = None
    if method in {"permutation", "both"}:
        if permutation_importance is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "scikit-learn is required for permutation importances. Install it or set method='feature'."
            )
        perm_raw = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        perm: Any = perm_raw
        perm_result = pd.DataFrame(
            {
                "importances_mean": perm.importances_mean,
                "importances_std": perm.importances_std,
            },
            index=X.columns,
        )

    return ImportanceResult(
        summary=summary,
        model=model,
        feature_matrix=X,
        target=y,
        feature_importances=feature_importances,
        permutation_importances=perm_result,
        category_mappings=category_mappings,
    )


def _normalise_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _normalise_column_key(df: pd.DataFrame, key: MetricKey) -> MetricKey:
    if key in df.columns:
        return key
    depth = df.columns.nlevels
    if isinstance(key, tuple):
        padded = key + (Placeholder,) * max(0, depth - len(key))
        if padded in df.columns:
            return padded
    flattened = _flatten_column_name(key)
    for column in df.columns:
        if _flatten_column_name(column) == flattened:
            return column
    raise KeyError(f"Column '{key}' not found in combined dataframe")


def _collect_hyperparameters(df: pd.DataFrame) -> list[tuple[MetricKey, str]]:
    hyperparams: list[tuple[MetricKey, str]] = []
    for column in df.columns:
        if not isinstance(column, tuple):
            continue
        if len(column) < 3:
            continue
        if column[0] != "config" or column[1] != "cli_args":
            continue
        label = _flatten_column_name(column)
        hyperparams.append((column, label))
    return hyperparams


def _flatten_column_name(column: MetricKey) -> str:
    if not isinstance(column, tuple):
        return str(column)
    parts = [str(part) for part in column if part is not Placeholder]
    return ".".join(parts)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str, np.number)) or value is None


def _encode_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[int, Any]]]:
    encoded = pd.DataFrame(index=features.index)
    category_mappings: dict[str, dict[int, Any]] = {}

    for column in features.columns:
        series = features[column]
        if pd.api.types.is_bool_dtype(series):
            encoded[column] = series.astype(int)
            continue
        if pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
            continue
        categorical = series.astype("category")
        codes = categorical.cat.codes.replace({-1: np.nan}).astype(float)
        encoded[column] = codes
        category_mappings[column] = {int(code): value for code, value in enumerate(categorical.cat.categories)}

    constant_columns: list[str] = []
    for column in list(encoded.columns):
        series = encoded[column]
        non_na = series.dropna()
        if non_na.empty:
            constant_columns.append(column)
            continue
        if bool((non_na == non_na.iloc[0]).all()):
            constant_columns.append(column)
    if constant_columns:
        encoded = encoded.drop(columns=constant_columns)
        for column in constant_columns:
            category_mappings.pop(column, None)
        logger.debug("Dropping constant hyperparameters: %s", constant_columns)

    return encoded, category_mappings


# ----------

import os

import optuna


def _get_storage(experiment_path: str | Path):
    experiment_path = Path(experiment_path)
    if experiment_path.is_dir():
        db_path = f"{experiment_path}/optuna_study.db"
    else:
        db_path = str(experiment_path)
    storage_str = f"sqlite:///{db_path}"
    try:
        storage = optuna.storages.RDBStorage(
            url=storage_str,
            engine_kwargs={"connect_args": {"timeout": 10}},
        )
    except Exception:
        logger.error("Could not oben RDBStorage %s", db_path)
        raise
    logger.info("Created/Opened db %s", db_path)
    return storage


def make_study_name(base: str = "pbt_study", env: str | None = None, step: int | None = None, suffix: str = "") -> str:
    study_name = base
    if env is not None:
        study_name += f"_env={env}"
    if step is not None:
        study_name += f"_step={step}"
    else:
        study_name += "_global"
    study_name += suffix
    return study_name


def get_optuna_study(
    experiment_path: str | Path | optuna.storages.BaseStorage,
    env: str | None,
    step: int | None = None,
    suffix: str = "",
    *,
    load_if_exists=True,
    clear_study: bool | Literal["all"] | Sequence[str] = False,
):
    if isinstance(experiment_path, (Path, str)):
        storage = _get_storage(experiment_path)
    else:
        storage = experiment_path
    study_name = make_study_name("pbt_study", env, step, suffix)
    if clear_study:
        if clear_study == "all":
            clear_study = True
        elif isinstance(clear_study, (Sequence, Collection)):  # includes single strings
            if "step=" in study_name:
                step_part = study_name.split("step=")[-1].split("_")[0]

                study_name_no_step = re.sub(f"step={step_part}_?", "", study_name).rstrip("_")
                study_name = study_name_no_step
            if study_name.removesuffix(suffix) in clear_study or study_name in clear_study:
                clear_study = True
            else:
                clear_study = False
        if clear_study is True:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                logger.info("Study with name %s does not exist, cannot delete it.", study_name)
            else:
                logger.info("Deleted and recreating study %s", study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=load_if_exists, direction="maximize"
    )
    logger.debug("Created optuna study %s", study_name)
    TRIAL_CACHE[study] = {t.user_attrs.get("identifier") for t in study.get_trials()}  # pyright: ignore[reportArgumentType]
    TRIAL_CACHE[study].discard(None)  # pyright: ignore[reportArgumentType]
    return study


def get_experiment_track_study(storage: optuna.storages.BaseStorage):
    tracking_study = optuna.create_study(study_name="_experiment_tracker", storage=storage, load_if_exists=True)
    STORED_EXPERIMENTS.update(t.user_attrs["run_id"] for t in tracking_study.get_trials())
    return tracking_study


def add_finished_experiment(run_id: str, tracking_study: optuna.Study):
    tracker = optuna.create_trial(state=optuna.trial.TrialState.COMPLETE, value=1, user_attrs={"run_id": run_id})
    tracking_study.add_trial(tracker)
    STORED_EXPERIMENTS.add(run_id)


def clear_tracking_study(storage: optuna.storages.BaseStorage):
    clear_study(storage, study_name="_experiment_tracker")


def clear_study(storage, study_name: str):
    optuna.delete_study(study_name=study_name, storage=storage)


def create_finished_trial(
    metric_result: float,
    identifier: str,
    params: dict[str, Any],
    intermediate_values: dict[int, float] | None = None,
    distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
):
    params = params.copy()
    user_attrs = {
        "identifier": identifier,
        "vf_share_layers": params.pop("vf_share_layers"),
        "use_kl_loss": params.pop("use_kl_loss"),
    }
    assert isinstance(user_attrs["vf_share_layers"], bool)
    assert isinstance(user_attrs["use_kl_loss"], bool)
    try:
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=metric_result,
            params=params,
            intermediate_values=intermediate_values,
            distributions=distributions,
            user_attrs=user_attrs,
        )
    except ValueError as ve:
        if not distributions or "not in" not in str(ve):
            raise
        # likely because we changed the distribution
        # need to figure out to which dist the value belongs
        # Only works for Categorical:
        wrong_metrics = {k: v for k, v in params.items() if k in distributions and v not in distributions[k].choices}
        for k, v in wrong_metrics.items():
            new_choices = (v, *distributions[k].choices)
            try:
                new_choices = tuple(sorted(new_choices))
            except Exception:  # noqa: BLE001
                pass
            distributions[k].choices = new_choices
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=metric_result,
            params=params,
            intermediate_values=intermediate_values,
            distributions=distributions,
            user_attrs=user_attrs,
        )
    return trial


GLOBAL_STUDY = "global"

TRIAL_CACHE: dict[optuna.Study, set[str]] = {}
STORED_EXPERIMENTS: set[str] = set()


def maybe_add_trials_to_study(
    trial: optuna.trial.FrozenTrial | Iterable[optuna.trial.FrozenTrial], study: optuna.Study, *, force: bool = False
):
    if isinstance(trial, Iterable) and not force:
        trial = [t for t in trial if t.user_attrs["identifier"] not in TRIAL_CACHE[study]]
    if isinstance(trial, Iterable):
        study.add_trials(trial)
        return

    if force or trial.user_attrs["identifier"] not in TRIAL_CACHE[study]:
        study.add_trial(trial)
    else:
        logger.info(
            "Trial with identifier '%s' already existed in the study did not add it again",
            trial.user_attrs["identifier"],
        )


__DEBUG_SKIP_EXISTING = False


def optuna_create_studies(
    *experiment_paths,
    database_path: str | Path | None = None,
    env: str | None,
    study_each_epoch: bool | None = False,
    dir_depth=1,
    metric="episode_reward_mean",
    group_stat: str | None = None,
    distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
    load_if_exists: bool = True,
    submission_file_path: Path | None = None,
    disable_checks: bool = True,
    clear_experiment_cache: bool = False,
    clear_study: Sequence[str] | Literal[False] = False,
    excludes: Collection[str] = (),
    submission_study: str | None = None,
) -> dict[Any, optuna.Study]:
    assert load_if_exists, "Not loading existing is unstable"
    metric_to_check = metric
    if disable_checks:
        _disable_distribution_check()
    studies: dict[Literal["global"] | int, optuna.Study] = {}
    pbt_centered_studies: dict[optuna.Study, optuna.Study] = {}
    submission_studies: dict[optuna.Study, optuna.Study] = {}
    submission_studies_centered: dict[optuna.Study, optuna.Study] = {}
    storage = _get_storage(database_path or experiment_paths[0])
    if clear_experiment_cache:
        clear_tracking_study(storage)
        STORED_EXPERIMENTS.clear()
    tracking_study = get_experiment_track_study(storage)
    if clear_experiment_cache:
        assert not STORED_EXPERIMENTS
    if not study_each_epoch:
        study = get_optuna_study(storage, env, load_if_exists=load_if_exists, clear_study=clear_study)
        studies[GLOBAL_STUDY] = study
        if submission_study:
            submission_studies[studies[GLOBAL_STUDY]] = get_optuna_study(
                storage, env, suffix=f"_{submission_study}", clear_study=clear_study, load_if_exists=load_if_exists
            )
    if study_each_epoch is not False:
        # Create a study for each epoch at step 8192 * 4 until max_step
        for epoch in range(1, 32 + 1):
            step = epoch * 8192 * 4
            studies[step] = get_optuna_study(storage, env, step, clear_study=clear_study)
            pbt_centered_studies[studies[step]] = get_optuna_study(
                storage, env, step, suffix="_centered", clear_study=clear_study
            )
            if submission_study:
                submission_studies[studies[step]] = get_optuna_study(
                    storage, env, step, suffix=f"_{submission_study}", clear_study=clear_study
                )
                submission_studies_centered[studies[step]] = get_optuna_study(
                    storage,
                    env,
                    step,
                    suffix=f"_{submission_study}_centered",
                    clear_study=clear_study,
                )

        # This is not sufficient for the 1/8 perturbation intervals
        for epoch in range(1, 8 + 1):
            step = epoch * 147456
            if step not in studies:
                studies[step] = get_optuna_study(storage, env, step, clear_study=clear_study)
                pbt_centered_studies[studies[step]] = get_optuna_study(
                    storage, env, step, suffix="_centered", clear_study=clear_study
                )
                if submission_study:
                    submission_studies[studies[step]] = get_optuna_study(
                        storage, env, step, suffix=f"_{submission_study}", clear_study=clear_study
                    )
                    submission_studies_centered[studies[step]] = get_optuna_study(
                        storage,
                        env,
                        step,
                        suffix=f"_{submission_study}_centered",
                        clear_study=clear_study,
                    )

    def get_all_studies():
        for key, main_study in studies.items():
            # key is "global" or step
            if isinstance(key, int):
                key = f"step={key}"  # noqa
                if env and env not in key:
                    key = f"{env}_{key}"  # noqa
            elif env and env not in key:
                key = f"{env}_{key}"  # noqa
            yield (key, main_study)
            if main_study in pbt_centered_studies:
                yield (key + "_centered", pbt_centered_studies[main_study])
            if main_study in submission_studies:
                yield (key + f"_{submission_study}", submission_studies[main_study])
            if main_study in submission_studies_centered:
                yield (key + f"_{submission_study}_centered", submission_studies_centered[main_study])

    for experiment_path in experiment_paths:
        logger.info("Checking path for experiments: %s", experiment_path)
        if dir_depth == 0:
            experiment_subdirs = [Path(experiment_path)]
        else:
            experiment_subdirs = Path(experiment_path).glob(f"*/{'/'.join(['*'] * dir_depth)}")
        running_experiments = set()
        if submission_file_path:
            running_experiments = get_running_experiments(submission_file_path, include_restore=True)
        for experiment_dir in list(experiment_subdirs):
            if any(excl in str(experiment_dir) for excl in excludes) or "cometml" in str(experiment_dir):
                if "cometml" not in str(experiment_dir):
                    logger.info(f"Excluding experiment path {experiment_dir} due to exclude patterns.")  # noqa: G004
                continue

            no_errors = True
            experiment_id = experiment_dir.name.split("-")[-1]
            if experiment_id in running_experiments:
                logger.info("skipping running experiment: %s", experiment_dir.name)
                continue
            if experiment_id in STORED_EXPERIMENTS:
                logger.info("Skipping run %s as the id is already stored", experiment_id)
                global __DEBUG_SKIP_EXISTING
                if not DEBUG or __DEBUG_SKIP_EXISTING:
                    continue
                __DEBUG_SKIP_EXISTING = True

            logger.info("Checking experiment path: %s", experiment_dir)
            # Check if this directory is an experiment directory
            marker_files = {".validate_storage_marker", "pbt_global.txt", "tuner.pkl"}
            dir_contents = [p.name for p in experiment_dir.glob("*")]
            if not any(marker_file in dir_contents for marker_file in marker_files):
                logger.warning("Directory %s does not appear to be a valid experiment directory.", experiment_dir)
                continue
            if env and env not in experiment_dir.name:
                logger.debug("Skpping experiment %s as env %s not in name", experiment_dir, env)
                continue

            logger.debug("Processing experiment directory: %s", experiment_dir)

            trials_to_add = {study: [] for study in studies.values()} | {
                study: [] for study in pbt_centered_studies.values()
            }
            if submission_study:
                trials_to_add |= {study: [] for study in submission_studies.values()}
                trials_to_add |= {study: [] for study in submission_studies_centered.values()}
            try:
                # Load run data for this experiment
                run_frames = load_run_data(experiment_dir)
                if len(run_frames) == 0:
                    logger.warning("No runs found for experiment at %s", experiment_dir)
                    continue
                if not isinstance(run_frames, pd.DataFrame):
                    df = combine_df(run_frames)
                    save_run_data(experiment_dir, df)
                else:
                    df = run_frames
                _group_stat, df = get_and_check_group_stat(df, group_stat=None, group_by=("pbt_group_key",))

                metric = check_metric_backport(df, metric_to_check)
                # Get tools to center epochs
                _metric_values, _epoch_end_steps, (value_shifter, value_shifter_with_first) = get_epoch_stats(
                    df, metric, individual_runs=False
                )

                # Get vf_share_layers info and use_kl_loss info and check that they are all equal
                vf_share_layers = df.config.get("vf_share_layers", df.config.cli_args.get("vf_share_layers", True))
                if not isinstance(vf_share_layers, bool):
                    assert (vf_share_layers.iloc[0] == vf_share_layers).all().item()
                    vf_share_layers = vf_share_layers.iloc[0].item()
                use_kl_loss = df.config.get("use_kl_loss", df.config.cli_args.get("use_kl_loss", False))
                if not isinstance(use_kl_loss, bool):
                    _does_kl_match = (use_kl_loss.iloc[0] == use_kl_loss).all()
                    assert _does_kl_match.item() if hasattr(_does_kl_match, "item") else _does_kl_match
                    use_kl_loss = use_kl_loss.iloc[0].item()

                # Create trials for each run_id
                # TODO: possibly groupkey -> take mean
                # pbt_epoch should be always present when using combine_df
                try:
                    max_epoch = df.config.pbt_epoch.max().item()
                except AttributeError:
                    if "pbt_epoch" in df.config:
                        raise  # some other error
                    run_frames = load_run_data(experiment_dir, use_cache=False)
                    df = combine_df(run_frames)
                    if "pbt_epoch" not in df.config:
                        remote_breakpoint()
                        raise
                    save_run_data(experiment_dir, df)
                    max_epoch = df.config.pbt_epoch.max().item()
                if study_each_epoch is True:
                    if max_epoch == 0:
                        logger.info("Experiment %s has only a single epoch skipping", experiment_path)
                        continue
                    iterator = df.groupby(("pbt_epoch", "pbt_group_key"))
                elif study_each_epoch is False or (study_each_epoch is None and max_epoch == 0):
                    iterator = df.groupby("run_id")
                # but do not do for baseliens that have only one epoch or short runs
                else:
                    iterator = chain(
                        df.groupby("run_id"),
                        df.groupby(
                            [
                                ifill("config", "pbt_epoch", n=df.columns.nlevels),
                                ifill("config", "pbt_group_key", n=df.columns.nlevels),
                            ]
                        ),
                    )
                # TODO: to have a more local evaluation over PBT epochs standardise the results
                # for that subtract the result of the parent run
                group_key: Literal["run_id"] | tuple[Literal["pbt_epoch"], Literal["pbt_group_key"]]
                for i, (group_key, group) in enumerate(iterator):  # pyright: ignore[reportAssignmentType]
                    if group.empty:
                        continue
                    per_pbt_epoch = False
                    if isinstance(group_key, tuple):
                        run_id = group_key[1]
                        per_pbt_epoch = True
                        # run_id = group_key[1]
                        # No run id if we group by pbt_group_key
                        trial_identifier = "-".join(group.index.get_level_values("run_id").unique().sort_values())
                        group_numeric = group.select_dtypes(include=["number", "bool"])[
                            (group.current_step == group.current_step.max()).values  # noqa: PD011
                        ]
                        final_row = group_numeric.mean()
                        # if "grad_clip" in df.config:
                        #    # is the None value cast to NaN? -> cast below
                        #    remote_breakpoint()
                        #    assert "grad_clip" in final_row.config
                    else:
                        run_id = group_key
                        trial_identifier = run_id
                        final_row = group.infer_objects(copy=False).ffill().iloc[-1]
                    # Extract final metric value (last row)

                    # Extract hyperparameters from config
                    params = clean_placeholder_keys(final_row.to_dict(), flatten=True)
                    if distributions:
                        for key in distributions.keys():
                            if key not in params:
                                params[key] = final_row.config.get(
                                    key, final_row.config.get("cli_args", {}).get(key, "NOT_FOUND")
                                )
                                if isinstance(params[key], (pd.DataFrame, pd.Series)) and params[key].size == 1:  # pyright: ignore[reportAttributeAccessIssue]
                                    params[key] = params[key].item()  # pyright: ignore[reportAttributeAccessIssue]
                                if params[key] == "NOT_FOUND":
                                    params[key] = getattr(DefaultArgumentParser, key)
                        params = {k: v for k, v in params.items() if not isinstance(v, str) or v != "NOT_FOUND"}
                    # Clean floating point errors from params
                    params = cast_numpy_numbers(params)
                    params = round_floats(params)
                    if group_stat == "lr":
                        assert 0.00047713000000000003 not in params.values()
                    params["vf_share_layers"] = vf_share_layers
                    params["use_kl_loss"] = use_kl_loss
                    if "grad_clip" in params and pd.isna(params["grad_clip"]):
                        params["grad_clip"] = None
                    assert params["grad_clip"] is None or not math.isnan(params["grad_clip"])
                    # Clean placeholders
                    if metric not in final_row:
                        raise KeyError(f"Metric column '{metric}' not found in combined dataframe")  # noqa: TRY301
                    try:
                        metric_result = float(final_row[metric].iloc[0])
                    except Exception:
                        metric_result = float(final_row[metric])
                    if per_pbt_epoch:
                        try:
                            # need to either substract the parent metric, of if run was continued value from previous epoch
                            # For individual runs:
                            if False:
                                if not final_row.config.fork_from.parent_fork_id.isna().any():
                                    shift_values = value_shifter_with_first.loc[
                                        final_row.config.pbt_epoch.item(),
                                        final_row.config.fork_from.parent_fork_id.item(),
                                    ]
                                    assert (
                                        shift_values["current_step"].item()
                                        == final_row.config.fork_from.parent_env_steps.item()
                                    )
                                    metric_centered = (metric_result - shift_values[metric]).item()
                                else:
                                    # first epoch oder continued from self
                                    # FIXME: run_id is NOT the experiment_key
                                    shift_values = value_shifter.loc[final_row.config.pbt_epoch.item(), run_id]
                                    assert (shift_values["current_step"].item() < final_row.current_step).all()
                                    metric_centered = (metric_result - shift_values[metric]).item()
                            else:
                                # for group
                                try:
                                    shift_values = value_shifter_with_first.loc[final_row.config.pbt_epoch.item()]
                                except KeyError:
                                    # maybe need to remove parquet file and recreate it
                                    remote_breakpoint()
                                    raise
                                if metric in shift_values:
                                    metric_centered = (metric_result - shift_values[metric]).item()
                                elif isinstance(metric, str):
                                    raise KeyError(metric, "not in shift values")
                                else:  # some Sequece
                                    metric_centered = (metric_result - shift_values["-".join(metric)]).item()
                            centered_trial = create_finished_trial(
                                metric_result=metric_centered,
                                identifier=trial_identifier + "_centered",
                                params={
                                    k: v
                                    for k, v in params.items()
                                    if k in distributions or k in ("vf_share_layers", "use_kl_loss")
                                }
                                if distributions
                                else params,
                                distributions=distributions,
                                # intermediate_values=intermediate_values
                            )
                        except Exception:
                            logger.exception("Could not create centered trial for %s", trial_identifier)
                            remote_breakpoint()
                    # Create and add trial to study
                    trial = create_finished_trial(
                        metric_result=metric_result,
                        identifier=trial_identifier,
                        params={
                            k: v
                            for k, v in params.items()
                            if k in distributions or k in ("vf_share_layers", "use_kl_loss")
                        }
                        if distributions
                        else params,
                        distributions=distributions,
                        # intermediate_values=intermediate_values
                    )
                    if not per_pbt_epoch:
                        trials_to_add[studies[GLOBAL_STUDY]].append(trial)
                        if submission_study:
                            trials_to_add[submission_studies[studies[GLOBAL_STUDY]]].append(trial)
                    else:
                        # TODO: will not work if trial ended too early
                        current_step = int(final_row.current_step.iloc[0])
                        if current_step in studies:
                            study_to_add = studies[current_step]
                        else:
                            # do we have perturbation interval
                            perturbation_interval = df.attrs.get("perturbation_interval")
                            if perturbation_interval is not None:
                                try:
                                    study_to_add = studies[perturbation_interval * group_key[0]]
                                except (KeyError, IndexError):
                                    study_to_add = studies[perturbation_interval * (final_row.config.pbt_epoch + 1)]
                            else:
                                # if current_step is not slightly above <= 2048 on another key ceil it
                                study_keys = sorted([step for step in studies.keys() if isinstance(step, int)])
                                closest_key_idx = np.argmin([abs(current_step - step) for step in study_keys])
                                closest_key = study_keys[closest_key_idx]
                                upper_bound = closest_key + final_row.get(
                                    "batch_size", final_row.config.get("train_batch_size_per_learner", 2048)
                                )
                                if isinstance(upper_bound, pd.Series):
                                    upper_bound: float = upper_bound.iloc[-1]
                                final_epoch = final_row.config.get("pbt_epoch", 0)
                                total_steps = final_row.config.get("cli_args", {}).get("total_steps").item()
                                if total_steps is not None:
                                    total_steps = int(total_steps)
                                if closest_key > current_step or current_step <= upper_bound:
                                    if closest_key == study_keys[-2] and total_steps == study_keys[-1]:
                                        # special case where we are close to -2 but wanted to train to -1
                                        logger.warning(
                                            "Trial with steps %s did not reach the exact end of %s but has total_steps %s putting it into that epoch.",
                                            current_step,
                                            study_keys[-1],
                                            total_steps,
                                        )
                                        study_to_add = studies[total_steps]
                                    else:
                                        study_to_add = studies[closest_key]
                                else:
                                    # Known to be trained a few steps over
                                    if "tws47f25121306216cda4" in trial_identifier and current_step >= 1048576:
                                        study_to_add = studies[1048576]

                                    logger.warning("Could not determine epoch of %s at step %s", run_id, current_step)
                                    # is in the lower half between two keys - ceil up
                                    try:
                                        study_to_add = studies[study_keys[closest_key_idx + 1]]
                                    except (KeyError, IndexError):
                                        if current_step > study_keys[-1]:
                                            # was trained to long add to max
                                            logger.warning(
                                                "Trial was likely trained to long adding it to epoch matching step %s",
                                                study_keys[-1],
                                            )
                                            study_to_add = studies[study_keys[-1]]
                                        else:
                                            # if it is between the second largest and largest add to the largest.
                                            # Problem there is the total_steps 1_179_648 case and the 1_048_576 of 8 / 32 epochs
                                            # and possibly old trials not fitting in both.
                                            if total_steps is not None and total_steps in studies:
                                                logger.warning(
                                                    "Trial with steps %s did not reach the exact end of %s but has total_steps %s putting it into that epoch.",
                                                    current_step,
                                                    study_keys[-1],
                                                    total_steps,
                                                )
                                                if current_step == 744448:
                                                    remote_breakpoint()
                                                if closest_key < total_steps:
                                                    # Why did this then fail with a KeyError?
                                                    remote_breakpoint()
                                                    study_to_add = ...
                                                else:
                                                    study_to_add = studies[int(total_steps)]
                                            elif current_step > study_keys[-2] + 8192 * 2:
                                                logger.warning(
                                                    "Trial with steps %s did not reach the exact end of %s and is > %s putting it into the last epoch.",
                                                    current_step,
                                                    study_keys[-1],
                                                    study_keys[-2],
                                                )
                                                study_to_add = studies[max(study_keys)]
                                            elif current_step > study_keys[-2] and final_epoch >= 6:
                                                # Do we have a total steps info?
                                                if 6 <= final_epoch <= 12:
                                                    study_to_add = studies[study_keys[-1]]
                                                else:
                                                    # ~32 epoch case
                                                    study_to_add = studies[study_keys[-2]]
                                            else:
                                                # should not happen
                                                logger.error(
                                                    "Cannot determine pbt epoch / current_step of trial %s with at step %s.",
                                                    run_id,
                                                    current_step,
                                                )
                                                study_to_add = None
                                                no_errors = False
                                    else:
                                        # No KeyError
                                        logger.warning(
                                            "Could not determine pbt epoch / current_step of trial %s with at step %s. "
                                            "Added it to next step. %s",
                                            run_id,
                                            current_step,
                                            study_keys[closest_key_idx + 1],
                                        )
                        if study_to_add is not None:
                            trials_to_add[study_to_add].append(trial)
                            trials_to_add[pbt_centered_studies[study_to_add]].append(centered_trial)
                            if submission_study:
                                trials_to_add[submission_studies[study_to_add]].append(trial)
                                trials_to_add[submission_studies_centered[study_to_add]].append(centered_trial)

                    if i % _REPORT_INTERVAL == 0:
                        if per_pbt_epoch:
                            logger.debug(
                                "Adding trial for run %s pbt_epoch %s with metric %s - and %d more trials",
                                run_id,
                                group_key[0],
                                metric_result,
                                _REPORT_INTERVAL,
                            )
                        else:
                            logger.debug(
                                "Adding trial for run %s with metric %s - and %d more trials",
                                run_id,
                                metric_result,
                                _REPORT_INTERVAL,
                            )

            except Exception as e:
                logger.exception("Failed to process experiment at %s: %r", experiment_dir, e)
                no_errors = False
                if "does does not match id in file name" in str(e):
                    # likely incomplete offline data
                    continue
                remote_breakpoint()
                continue
            logger.info("Adding trials from experiment %s to studies", experiment_dir)
            for study, trials in trials_to_add.items():
                maybe_add_trials_to_study(trials, study)
            if no_errors and study_each_epoch is None:  # add it as finished when we checked both types.
                add_finished_experiment(experiment_id, tracking_study)

    return dict(get_all_studies())


PARAMS_TO_CHECK = {
    "batch_size",
    "lr",
    "gamma",
}


def _fix_distributions(study: optuna.Study, params: Collection[str]):
    if hasattr(study, "_fixed_trials"):  # backup for when we adjust what get_trials should
        return study._fixed_trials  # pyright: ignore
    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    distributions: dict[str, set[optuna.distributions.BaseDistribution]] = {}
    for trial in completed_trials:
        trial_distributions = trial.distributions
        for param in params:
            if param in trial_distributions:
                continue
            ...  # Need all values
            raise KeyError(f"Parameter {param} not found in trial distributions")
        for name, distribution in trial_distributions.items():
            if name not in distributions:
                distributions[name] = set()
            distributions[name].add(distribution)
    for name, dists in distributions.items():
        # merge them
        if len(dists) <= 1:
            continue
        logger.info("Parameter %s has multiple distributions: %s", name, dists)
        # Merge
        choices = set()
        for dist in dists:
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                choices.update(round_floats(dist.choices))
            elif isinstance(dist, optuna.distributions.FloatDistribution):
                choices.update(round_floats([dist.low, dist.high]))
            elif isinstance(dist, optuna.distributions.IntDistribution):
                choices.update(range(dist.low, dist.high + 1))
            else:
                logger.warning("Cannot merge distribution of type %s for parameter %s", type(dist), name)
        if all(isinstance(c, (int, float)) for c in choices):
            if all(isinstance(c, int) for c in choices):
                new_dist = optuna.distributions.IntDistribution(
                    low=round_floats(min(choices)), high=round_floats(max(choices))
                )
            else:
                new_dist = optuna.distributions.FloatDistribution(
                    low=round_floats(min(choices)), high=round_floats(max(choices))
                )
        else:
            if None in choices:
                choices.discard(None)
                choices = (*sorted(round_floats(choices)), None)
                new_dist = optuna.distributions.CategoricalDistribution(choices=choices)
            else:
                new_dist = optuna.distributions.CategoricalDistribution(choices=tuple(sorted(round_floats(choices))))
        logger.debug("Setting merged distribution for parameter %s: %s", name, new_dist)
        # Now set it for all trials
        for trial in completed_trials:
            trial.distributions[name] = new_dist
    study._fixed_trials = completed_trials
    return completed_trials


def __sort_columns(col):
    if isinstance(col, pd.Index):
        return tuple(__sort_columns(c) for c in col)

    if isinstance(col, str):
        try:
            return int(col)
        except ValueError:
            return float("inf")
    if isinstance(col, tuple):
        return tuple(__sort_columns(c) for c in col)
    return col


def __try_step_cast(col):
    if isinstance(col, str):
        try:
            return int(col)
        except ValueError:
            pass
    if isinstance(col, tuple):
        return tuple(__try_step_cast(c) for c in col)
    return col


def _get_importances(study, evaluator, params, *, use_kl_loss: bool | None = None, vf_share_layers: bool | None = None):
    if use_kl_loss is None and vf_share_layers is None:
        if hasattr(study, "_fixed_trials"):
            study.get_trials = lambda *args, **kwargs: study._fixed_trials  # noqa
        try:
            importances = optuna.importance.get_param_importances(study, evaluator=evaluator, params=params)
        except ValueError as ve:
            if "dynamic search" in str(ve):
                study = deepcopy(study)  # noqa: PLW2901
                completed_trials = _fix_distributions(study, params or PARAMS_TO_CHECK)
                study.get_trials = lambda *args, **kwargs: completed_trials  # noqa
                importances = optuna.importance.get_param_importances(study, evaluator=evaluator, params=params)
            else:
                raise
        else:
            completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        return importances, completed_trials
    if hasattr(study, "_fixed_trials"):
        completed_trials = study._fixed_trials  # noqa
    else:
        study = deepcopy(study)  # noqa: PLW2901
        completed_trials = _fix_distributions(study, params or PARAMS_TO_CHECK)
    # Filter trials based on vf_share_layers and use_kl_loss
    # NOTE: currently there is not combination of use_kl_loss and not vf_share_layers in our experiments
    if vf_share_layers is not None:
        completed_trials = [
            t
            for t in completed_trials
            if t.user_attrs["vf_share_layers"] == vf_share_layers  # If this raises a KeyError recreate the study :(
        ]
    if use_kl_loss is not None:
        completed_trials = [
            t
            for t in completed_trials
            if t.user_attrs["use_kl_loss"] == use_kl_loss  # If this raises a KeyError recreate the study :(
        ]
    if not completed_trials:
        logger.warning(
            "No completed trials found for study %s with filtering vf_share_layers=%s and use_kl_loss=%s",
            study.study_name,
            vf_share_layers,
            use_kl_loss,
        )
        return {}, []
    study.get_trials = lambda *args, **kwargs: completed_trials  # noqa
    importances = optuna.importance.get_param_importances(study, evaluator=evaluator, params=params)
    return importances, completed_trials


def _analyze_single_study(
    key: Any,
    study: optuna.Study,
    evaluators: dict[str, Any],
    params: list[str] | None,
) -> list[dict[str, Any]]:
    """Analyze one study and return result rows."""
    rows: list[dict[str, Any]] = []
    logger.info("Analyzing study for key: %s", key)
    completed_trials = None
    try:
        for filter_kl, filter_vf_share in product([None, True, False], [None, True, False]):
            for evaluator_name, evaluator in evaluators.items():
                evaluator = deepcopy(evaluator)
                importances, completed_trials = _get_importances(
                    study, evaluator, params, use_kl_loss=filter_kl, vf_share_layers=filter_vf_share
                )
                if not completed_trials:
                    continue
                for param, importance in importances.items():
                    store_key = key
                    if isinstance(store_key, int):
                        # should normally have env name in key now
                        step = store_key
                        centered = False
                    else:
                        centered = "centered" in store_key
                        store_key = store_key.replace("_centered", "")

                        match = re.search(r"step=(\d+)", store_key)
                        if match:
                            step = match.groups()[0]
                            store_key = re.sub(r"_?step=\d+", "", store_key)
                        else:
                            step = "global"
                            store_key = store_key.replace("_global", "")
                    if filter_kl is True:
                        store_key = f"{store_key}_kl_loss"
                    elif filter_kl is False:
                        store_key = f"{store_key}_no_kl_loss"
                    if filter_vf_share is True:
                        store_key = f"{store_key}_shared_encoder"
                    elif filter_vf_share is False:
                        store_key = f"{store_key}_shared_encoder"
                    rows.append(
                        {
                            "param": param,
                            "importance": importance,
                            "evaluator_name": evaluator_name,
                            "key": store_key,
                            "centered": centered,
                            "step": step,
                            "number_of_trials": len(completed_trials),
                            "study_name": study.study_name,
                        }
                    )
                sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
                report_str = (
                    f"Hyperparameter importances for study {study.study_name} "
                    f"(filtered for kl_loss: {filter_kl}, shared encoder: {filter_vf_share}) ({evaluator_name}) "
                    f"with {len(completed_trials)} trials:\n"
                )
                for param, importance in sorted_importances:
                    report_str += f"  {param}: {importance:.5f}\n"
                logger.info("%s", report_str)
    except Exception as e:
        logger.exception("Failed to analyze study for key %s: %r", key, e)
        try:
            if len(study.get_trials()) == 0:
                pass
            else:
                remote_breakpoint()
        except Exception:
            remote_breakpoint()
    return rows


def optuna_analyze_studies(
    studies: dict[Any, optuna.Study],
    output_path: str | Path | None,
    params: list[str] | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame | None:
    from optuna.importance import (
        FanovaImportanceEvaluator,
        MeanDecreaseImpurityImportanceEvaluator,
        PedAnovaImportanceEvaluator,
    )

    evaluators = {
        "MeanDecreaseImpurity": MeanDecreaseImpurityImportanceEvaluator(),
        "PedAnovaLocal": PedAnovaImportanceEvaluator(evaluate_on_local=True),  # default
        "Fanova(default)": FanovaImportanceEvaluator(),  # default
        # "MeanDecreaseImpurity8": MeanDecreaseImpurityImportanceEvaluator(n_trees=12, max_depth=12),
        # "PedAnovaGlobal": PedAnovaImportanceEvaluator(evaluate_on_local=False),
        # "Fanova8": FanovaImportanceEvaluator(n_trees=12, max_depth=12),
    }

    all_results: list[dict[str, Any]] = []

    # Parallelize per-study processing
    workers = max_workers or max(1, min((os.cpu_count() or 6), len(studies), 22))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_single_study, key, study, evaluators, params): key
            for key, study in studies.items()
        }
        for future in as_completed(futures):
            rows = future.result()
            if rows:
                all_results.extend(rows)

    # Create combined DataFrame with multilevel columns
    if not all_results:
        return None
    results_df = pd.DataFrame(all_results)

    pivot_df = results_df.pivot_table(
        index="param", columns=["evaluator_name", "key", "centered", "step"], values="importance", fill_value=0.0
    )
    pivot_df.columns = pivot_df.columns.map(__try_step_cast)
    pivot_df.sort_index(axis=1, key=__sort_columns, inplace=True)

    if output_path is not None:
        # should not save to a unique file when we write only for submissions
        try:
            save_path = Path(output_path)
            if save_path.is_dir():
                save_path = save_path / "hyperparameter_importance_combined.csv"
            else:
                raise ValueError("output_path must be a directory to save importances")
            pivot_df.to_csv(save_path, index=True)
            pivot_df.to_parquet(save_path.with_suffix(".parquet"))
            logger.info("Saved combined importances to %s", save_path)
        except Exception:
            logger.exception("Could not export combined importances to %s", output_path)
    return pivot_df


_check_distribution_compatibility = optuna.distributions.check_distribution_compatibility


def _disable_distribution_check():
    # TODO: Alternatively move all trials to a new study with the changed distribution
    optuna.distributions.check_distribution_compatibility = lambda a, b: None


import seaborn as sns


def plot_importance_studies(
    studies: dict[Any, optuna.Study] | pd.DataFrame, output_path: str | Path, params: list[str] | None = None
) -> None:
    # note there is also import optuna.visualization as optuna_vis
    if isinstance(studies, dict):
        # remote_breakpoint()
        study_results = optuna_analyze_studies(studies, output_path, params=params)
    else:
        study_results = studies
    if study_results is None:
        return
    if study_results.columns.nlevels > 1:
        importances = {
            evaluator_name: study_results[evaluator_name] for evaluator_name in study_results.columns.levels[0]
        }
    else:
        importances = {"_": study_results}
    written_paths = set()
    for evaluator_name, importance_df in importances.items():
        for key in importance_df.columns.get_level_values("key").unique():
            key_df = importance_df[key]
            for centered in key_df.columns.get_level_values("centered").unique():
                fig = None
                try:
                    centered_df = key_df[centered]

                    # Create a figure with proper size and colormap
                    centered_df = centered_df.sort_index(axis=1, key=__sort_columns)  # noqa: PLW2901

                    # Cannot center global study so only assert for False
                    if centered is False:
                        assert "global" == centered_df.columns[-1], (
                            f"Expected last column to be 'global' not {centered_df.columns[-1]}"
                        )
                    add_global = "global" in centered_df.columns
                    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(centered_df))))
                    # Plot heatmap
                    cmap = sns.color_palette("magma", as_cmap=True)
                    # Find the max value in each column and create a mask for those cells
                    max_mask = centered_df.eq(centered_df.max())
                    # Plot heatmap
                    sns.heatmap(
                        centered_df,
                        annot=False,
                        xticklabels=True,
                        ax=ax,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                        yticklabels=True,
                        linewidths=0.5,
                        linecolor="gray",
                    )
                    # Set every 8th x ticklabel in fraction of 1e6 (e.g., 100000 -> 0.1)
                    xticks = ax.get_xticks()
                    xlabels = centered_df.columns
                    new_labels = []
                    for i, col in enumerate(xlabels):
                        if col == "global":
                            continue
                        if i % 8 == 0 or i == len(xlabels) - (2 if add_global else 1):
                            try:
                                val = int(col)
                                label = f"{val / 1e6:.1f}"
                            except Exception:
                                label = str(col)
                            new_labels.append(label)
                        else:
                            new_labels.append("")
                    if add_global:
                        new_labels.append("G")
                    ax.set_xticklabels(new_labels, rotation=45, ha="center")
                    ytick_labels = ax.get_yticklabels()
                    ax.set_yticklabels(ytick_labels, rotation=22, va="top")
                    ax.set(ylabel=None)

                    # Draw a black rectangle around the max value in each column
                    for col_idx, col in enumerate(centered_df.columns):
                        if max_mask[col].all() or sum(max_mask[col]) <= 1:  # do not draw if all 0 or only 1 value
                            continue
                        max_rows = centered_df.index[max_mask[col]]
                        for row in max_rows:
                            row_idx = centered_df.index.get_loc(row)
                            # Rectangle: (x, y) is top left, width=1, height=1
                            rect = plt.Rectangle(
                                (col_idx, row_idx),
                                1,
                                1,
                                fill=False,
                                edgecolor="white",
                                linewidth=2,
                                zorder=10,
                            )
                            ax.add_patch(rect)
                    plt.title(f"Hyperparameter Importances ({evaluator_name})")
                    plt.xlabel("Step")
                    file_path = Path(
                        output_path
                    ) / f"hyperparameter_importance_{key}-centered={centered}-{evaluator_name}.pdf".replace(
                        "(default)", ""
                    )
                    assert file_path not in written_paths, "File path already written: %s" % file_path
                    fig.savefig(
                        file_path,
                        format="pdf",
                        bbox_inches="tight",
                    )
                    written_paths.add(file_path)
                    logger.info("Saved heatmap for %s key centered=%s at '%s", key, centered, file_path)
                except Exception:
                    logger.error("Could not plot heatmap for %s key centered=%s", key, centered)
                    remote_breakpoint()
                finally:
                    if fig:
                        plt.close(fig)


if __name__ == "__main__":
    import argparse

    from experiments.create_tune_parameters import default_distributions
    from ray_utilities import nice_logger

    os.chdir(Path(__file__).parent.parent.parent)

    parser = argparse.ArgumentParser(description="Analyze hyperparameter importance for RL experiments.")
    parser.add_argument(
        "envs",
        nargs="*",
        help="List of environment names to analyze (e.g., Humanoid-v5 Hopper-v5).",
    )
    parser.add_argument(
        "--clear-experiment-cache",
        action="store_true",
        help="Clear the experiment tracking study before running.",
    )
    parser.add_argument(
        "--clear-study",
        nargs="?",
        const="all",
        default=False,
        help="Clear the experiment tracking study before running.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[
            "outputs/shared/experiments/Default-mlp-%s",
            "outputs/shared_backup/needs_sync/Default-mlp-%s",
            "outputs/shared_backup/Default-mlp-%s",
        ],
        help="Experiment directories to analyze (default: three HumanoidStandup-v5 paths).",
    )
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing analysis files.")
    args = parser.parse_args()
    # possibly for repair sqlite3 "outputs/shared/optuna_hp_study.db" ".dump" | sqlite3 new.db

    logger = nice_logger(logger, logging.INFO)
    distributions = load_distributions_from_json(write_distributions_to_json(default_distributions), as_optuna=True)
    distributions.pop("test", None)
    distributions.pop("minibatch_scale", None)
    if "vf_clip_param" in distributions and 0.0 not in distributions["vf_clip_param"].choices:
        # We changed this distribution
        distributions["vf_clip_param"].choices = (0.0, *distributions["vf_clip_param"].choices)
    # TODO: Need to separate submission
    # - use_kl
    # - vf_share_layers
    # (fine)
    # Make studies for each submission file.
    # And all/rest into a common study.
    # TODO: Subtract start value of epoch to get actual improvement
    if "DEBUG" in os.environ or "test" in sys.argv:
        DEBUG = True
        paths = ("outputs/shared/experiments/Default-mlp-Acrobot-v1",)
        env = "Acrobot-v1"
        paths = [
            "outputs/shared/experiments/Default-mlp-HumanoidStandup-v5",
            "outputs/shared_backup/needs_sync/Default-mlp-HumanoidStandup-v5",
            "outputs/shared_backup/Default-mlp-HumanoidStandup-v5",
        ]
        env = "HumanoidStandup-v5"
        studies = optuna_create_studies(
            *paths,
            # TESTING
            database_path="outputs/shared/optuna_hp_study.db",
            env=env,
            dir_depth=1,
            distributions=distributions,
            load_if_exists=True,
            study_each_epoch=None,
            submission_file_path=Path("experiments"),
            clear_experiment_cache=args.clear_experiment_cache,
            excludes=load_excludes(),
            # TESTING
        )
        studies = optuna_analyze_studies(studies, output_path=paths[0], params=list(distributions.keys()))
        # Want to plot for each env.
        plot_importance_studies(studies, output_path=paths[0], params=list(distributions.keys()))
        sys.exit(0)
    # clear_study = args.clear_study
    clear_experiment_cache = args.clear_experiment_cache
    if any(p.endswith(".yaml") for p in args.envs):
        # yaml file passed positional
        # assert default value:
        paths_action = next(a for a in parser._actions if a.dest == "paths")
        assert args.paths == paths_action.default, "When passing YAML files positionally, do not set --paths."
        args.paths = args.envs
        args.envs = []

    envs = args.envs or [
        "HumanoidStandup-v5",
        "Humanoid-v5",
        "Ant-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Reacher-v5",
        "Swimmer-v5",
        "Pusher-v5",
        "InvertedDoublePendulum-v5",
        "InvertedPendulum-v5",
        "CartPole-v1",
        "Acrobot-v1",
        "LunarLander-v3",
    ]

    if any(p.endswith(".yaml") for p in args.paths):
        # ------------ Submission File(s) -------------------

        assert all(p.endswith(".yaml") for p in args.paths), "Either all or none of the paths must be YAML files."
        # Load experiment groups from YAML files
        yaml_paths = args.paths
        paths = []
        all_run_paths: dict[str, Path] = {}
        all_run_infos: dict[str, SubmissionRun] = {}
        submissions_map = {}
        args.single = True
        for yaml_path in yaml_paths:
            run_paths = get_run_directories_from_submission(yaml_path)
            run_infos: dict[str, SubmissionRun] = get_runs_from_submission(yaml_path)
            env_mapping = {info["run_id"]: info["run_key"] for info in run_infos.values()}
            # slight chanc this fail if a run just finished
            assert env_mapping.keys() == run_paths.keys(), (
                f"Mismatch in runs for {yaml_path}. {set(env_mapping.keys()).symmetric_difference(run_paths.keys())}"
            )
            if args.envs:
                keep = set()
                # filter:
                for info in run_infos.values():
                    if info["run_id"] in args.envs:  # TODO: does not work for multi HP tuning
                        keep.add(info["run_id"])
                run_paths = {k: v for k, v in run_paths.items() if k in keep}
            all_run_paths.update(run_paths)
            all_run_infos.update(run_infos)
        # group by env and yaml_path to load less studies
        clear_studies = args.clear_study
        all_studies: dict[str, optuna.Study] = {}
        for yaml_path in yaml_paths:
            database_path = (
                "outputs/shared/optuna_hp_study.db"
                if "sympol" not in yaml_path.lower()
                else "outputs/shared/optuna_hp_study_sympol.db"
            )
            submission_name = (
                Path(yaml_path)
                .name.removesuffix(".yaml")
                .removeprefix("submissions")
                .removeprefix("submission")
                .strip("_")
            )
            for env in envs:
                paths = [
                    p
                    for rid, p in all_run_paths.items()
                    if all_run_infos[rid]["run_key"] == env and all_run_infos[rid]["submission_name"] == submission_name
                ]
                if not paths:
                    logger.warning("No runs found for env %s in provided YAML files.", env)
                    continue
                # If clear studies is False we do not clear, if true turn into a string after the first interation to clear all iterations
                # strings stay as strings.
                clear_this_study = (
                    clear_studies
                    if clear_studies is not None
                    else make_study_name("pbt_study", env, suffix=f"_{submission_name}")
                )
                if args.plot_only:
                    data_file = (
                        Path(f"outputs/shared/experiments/Default-mlp-{env}")
                        / "hyperparameter_importance_combined.parquet"
                    )
                    if not data_file.exists():
                        logger.error("Plot only specified but data file %s does not exist.", data_file)
                        continue
                    importance_results = pd.read_parquet(data_file)
                    plot_importance_studies(
                        importance_results,
                        output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                        params=list(distributions.keys()),
                    )
                    # TODO
                    # no studies to update all_results
                else:
                    studies = optuna_create_studies(
                        *paths,
                        database_path=database_path,
                        env=env,
                        dir_depth=0,
                        distributions=distributions,
                        load_if_exists=True,
                        study_each_epoch=None,
                        submission_file_path=None,  # used to filter RUNNING, but we already did
                        clear_experiment_cache=clear_experiment_cache,
                        clear_study=clear_this_study,  # TODO: Step studies are NOT cleared this way!
                        excludes=load_excludes(),
                        metric="episode_reward_mean",
                        submission_study=submission_name,
                    )
                    clear_experiment_cache = False  # needs to be done only once.
                    # Plots for the global files are wrong until all yaml files are processed
                    # Sufficent to do this only for the submission file studies, as the others are incomplete
                    reduced_studies = {k: v for k, v in studies.items() if submission_name in k}
                    study_results = optuna_analyze_studies(
                        reduced_studies,
                        None,
                        params=list(distributions.keys()),
                    )
                    if study_results is None or study_results.empty:
                        logger.warning("No study results for env %s submission %s", env, submission_name)
                        continue
                    study_results.to_parquet(
                        f"outputs/shared/experiments/Default-mlp-{env}/hyperparameter_importance_{submission_name}.parquet"
                    )
                    plot_importance_studies(
                        study_results,
                        output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                        params=list(distributions.keys()),
                    )
                all_studies.update(studies)
            if clear_studies is not False and not isinstance(clear_studies, str):
                clear_studies = None  # if false do not switch to clearning
        logger.info("All Trials processed, plotting combined importance.")
        for env in envs:
            remote_breakpoint()
            env_studies = {k: v for k, v in all_studies.items() if env in k}
            plot_importance_studies(
                env_studies,
                output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                params=list(distributions.keys()),
            )
        sys.exit()
    # ------------ ALL -------------------
    database_path = "outputs/shared/optuna_hp_study.db"
    if any("sympol" in p.lower() for p in args.paths):
        if not all("sympol" in p.lower() for p in args.paths):
            raise ValueError("Either all or none of the paths must be sympol experiments.")
        database_path = "outputs/shared/optuna_hp_study_sympol.db"
    for env in envs:
        paths = [p % env for p in args.paths]
        studies = optuna_create_studies(
            *paths,
            database_path=database_path,
            env=env,
            dir_depth=1,
            distributions=distributions,
            load_if_exists=True,
            study_each_epoch=None,
            submission_file_path=Path("experiments"),
            clear_experiment_cache=clear_experiment_cache,
            clear_study=args.clear_study,
            excludes=load_excludes(),
            metric="episode_reward_mean",
        )
        clear_experiment_cache = False  # needs to be done only once.
        plot_importance_studies(studies, output_path=paths[0], params=list(distributions.keys()))
