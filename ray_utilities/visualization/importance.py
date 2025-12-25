from __future__ import annotations

import argparse
import atexit
import logging
import math
import os
import re
import signal
import sys
import warnings
from collections.abc import Sized
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain, product
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, NamedTuple, Sequence, cast, overload
from zipfile import ZipFile

import matplotlib as mpl  # fmt: skip

mpl.use("Agg")  # Non-interactive backend safe for multiprocessing
import numpy as np
import optuna
import optuna.importance
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing_extensions import Literal

from experiments.create_tune_parameters import default_distributions, write_distributions_to_json
from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.misc import cast_numpy_numbers, round_floats
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.testing_utils import remote_breakpoint
from ray_utilities.visualization._common import Placeholder
from ray_utilities.visualization.data import (
    LOG_SETTINGS,
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
    try_literal_eval,
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
LargeMode = Literal["all", "large", "non_large"]

StudyName = str
DatabasePath = Path


class ImportanceTask(NamedTuple):
    filter_kl: bool | None
    filter_vf_share: bool | None
    centered_flag: bool
    large_mode: LargeMode
    submission_label: str | None
    evaluator_name: str
    evaluator_template: optuna.importance.BaseImportanceEvaluator


logger = logging.getLogger(__name__)


class EnvAnalysisProcessTask(NamedTuple):
    env: str
    submission_name: str
    database_path: str
    study_names: list[str]
    parquet_file: str
    param_names: list[str]
    excludes: frozenset[Literal["kl_loss", "no_kl_loss", "vf_share_layers", "no_vf_share_layers"]]
    output_path: str
    has_cached_results: bool


class EnvAnalysisProcessResult(NamedTuple):
    env: str
    outfiles: list[Path] | None
    tqdm_idx: int | None = None


def _run_env_analysis_process(task: EnvAnalysisProcessTask, tqdm_idx: int | None) -> EnvAnalysisProcessResult:
    # Set matplotlib backend for multiprocessing - must be done before importing pyplot
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend safe for multiprocessing

    logger.info(
        "Starting analysis for env %s submission %s with studies %s", task.env, task.submission_name, task.study_names
    )

    # Create progress bar for this process
    pbar = None
    if tqdm_idx is not None:
        pbar = tqdm(total=len(task.study_names), desc=f"Analyzing {task.env}", position=tqdm_idx, leave=False)

    try:
        parquet_path = Path(task.parquet_file)
        study_results = None
        if task.has_cached_results and parquet_path.exists():
            study_results = load_study_results(parquet_path)

        if study_results is None:
            logger.info(
                "No cached results found for env %s submission %s - analyzing studies", task.env, task.submission_name
            )
            # Pass tuples to enable parallel loading in worker threads
            study_generator = ((name, Path(task.database_path)) for name in task.study_names)
            if len(task.study_names) == 0:
                logger.warning(
                    "No studies to check for env %s submission %s. Studies: %s",
                    task.env,
                    task.submission_name,
                    task.study_names,
                )
                return EnvAnalysisProcessResult(task.env, None, tqdm_idx=tqdm_idx)
            study_results = optuna_analyze_studies(
                study_generator,
                None,
                params=task.param_names,
                excludes=cast(
                    "Collection[Literal['kl_loss', 'no_kl_loss', 'vf_share_layers', 'no_vf_share_layers']]",
                    set(task.excludes),
                ),
                max_workers=max(1, min(len(task.study_names), os.cpu_count() or 6, 28)),
                submission_filter=task.submission_name,
                progress_bar=pbar,
            )
            if study_results is None or study_results.empty:
                logger.warning("No study results for env %s submission %s", task.env, task.submission_name)
                return EnvAnalysisProcessResult(task.env, None, tqdm_idx=tqdm_idx)
            save_analysis_to_parquet(study_results, parquet_path)

        # Return results for plotting in main process
        # Don't plot in worker process to avoid matplotlib/multiprocessing issues
        logger.info(
            "Analysis complete for env %s submission %s, skipping plots in worker", task.env, task.submission_name
        )
        return EnvAnalysisProcessResult(task.env, None, tqdm_idx)
    finally:
        if pbar is not None:
            pbar.close()


_REPORT_INTERVAL = 32

DEBUG = False

# remote_breakpoint = partial(remote_breakpoint, port=5681)
# remote_breakpoint = lambda port=None: None


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
        if not isinstance(run_frames, pd.DataFrame):
            if not run_frames:
                logger.warning("No runs found for experiment at %s", path)
                continue
            combined_df = combine_df(run_frames)
            save_run_data(path, combined_df)
        elif run_frames.empty:
            logger.warning("No runs found for experiment at %s", path)
            continue
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


STUDY_COLUMN_NAMES = ("evaluator_name", "key", "centered", "large_mode", "step")


def __clean_wrong_key(col: tuple[Any, ...]):
    return (col[0], col[1].replace("_|", "|").rstrip("_"), *col[2:])


def load_study_results(parquet_file: str | Path, **kwargs) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(parquet_file, **kwargs)
    except FileNotFoundError:
        return None
    except ValueError:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_file)
        df = table.to_pandas(ignore_metadata=True)
        index = df.pop("param")
        clean_columns = [
            tuple(try_literal_eval(b) for b in multi) if isinstance(multi := try_literal_eval(c), tuple) else multi
            for c in df.columns
        ]
        try:
            mc_columns = pd.MultiIndex.from_tuples(clean_columns, names=STUDY_COLUMN_NAMES)
            df.columns = mc_columns
            df.columns.names = STUDY_COLUMN_NAMES
        except ValueError as ve:
            logger.error("Could not load parquet file %s with columns %s: %r", parquet_file, clean_columns[:5], ve)
            return None
            # old format with out large
        df.index = index
        df = df.sort_index(axis=1, key=__sort_columns)

    if "_|" in df.columns.get_level_values("key")[0]:
        df.columns = df.columns.map(__clean_wrong_key)
        save_analysis_to_parquet(df, parquet_file)
    return df


def save_analysis_to_parquet(df, parquet_file: str | Path):
    """Ensure the 'step' column level has uniform types and write to parquet.

    This function will:
    - detect mixed types in the 'step' level and raise ValueError("The DataFrame has column names of mixed type")
    - convert non-string step values to strings to avoid parquet warnings/errors
    - attempt to write the DataFrame to parquet while catching user warnings and raising if
      the warning contains "The DataFrame has column names of mixed type"
    """
    # Only act if there is a 'step' level in the MultiIndex
    if "step" in df.columns.names:
        step_idx = df.columns.names.index("step")
        steps = df.columns.get_level_values(step_idx)
        has_str = any(isinstance(s, str) for s in steps)
        has_non_str = any(not isinstance(s, str) for s in steps)

        # if we have mixed type cast all to str
        if has_str and has_non_str:
            new_columns: list[tuple[Any, ...]] = []
            for col in df.columns:
                col_list = list(col)
                col_list[step_idx] = str(col_list[step_idx])
                new_columns.append(tuple(col_list))
            df.columns = pd.MultiIndex.from_tuples(new_columns, names=df.columns.names)

    # Try to write to parquet and convert pandas UserWarnings into exceptions when relevant
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.to_parquet(parquet_file)
        for w in caught:
            msg = str(w.message)
            if "The DataFrame has column names of mixed type" in msg:
                remote_breakpoint()
                # raise ValueError(msg)


def _get_storage(experiment_path: str | Path, direction: str | None = "maximize") -> optuna.storages.BaseStorage:
    experiment_path = Path(experiment_path)
    if experiment_path.is_dir():
        db_path = f"{experiment_path}/optuna_study.db"
    else:
        db_path = str(experiment_path)
    storage_str = f"sqlite:///{db_path}"
    try:
        # Querying all trials might take some time
        storage = optuna.storages.RDBStorage(
            url=storage_str,
            # We have 33 epochs + global multiplied up to times 4.
            engine_kwargs={
                "connect_args": {"timeout": 30},
                "pool_size": 34,
                "max_overflow": 34,
                "pool_recycle": 60,
                "pool_timeout": 62,
            },
        )
    except Exception:
        logger.error("Could not oben RDBStorage %s", db_path)
        raise
    if direction is not None:
        # Save query at study setup
        storage.get_study_directions = lambda study_id: [
            optuna.study.StudyDirection.MAXIMIZE if direction == "maximize" else optuna.study.StudyDirection.MINIMIZE
        ]  # noqa: ARG005
    logger.info("Opened db %s", db_path, stacklevel=2)
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


@overload
def get_optuna_study(
    experiment_path: str | Path | optuna.storages.BaseStorage,
    env: str | None,
    step: int | None = None,
    suffix: str = "",
    *,
    load_if_exists=True,
    clear_study: bool | Literal["all"] | Sequence[str] = False,
    name_only: Literal[False] = False,
) -> optuna.Study: ...


@overload
def get_optuna_study(
    experiment_path: str | Path | optuna.storages.BaseStorage,
    env: str | None,
    step: int | None = None,
    suffix: str = "",
    *,
    load_if_exists=True,
    clear_study: bool | Literal["all"] | Sequence[str] = False,
    name_only: Literal[True],
) -> str: ...


def get_optuna_study(
    experiment_path: str | Path | optuna.storages.BaseStorage,
    env: str | None,
    step: int | None = None,
    suffix: str = "",
    *,
    load_if_exists=True,
    clear_study: bool | Literal["all"] | Sequence[str] = False,
    name_only: bool = False,
) -> optuna.Study | str:
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
    if name_only:
        return study_name
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
    *,
    centered_metric: float | None = None,
    analysis_step: int | str = "global",
    is_large: bool = False,
    submission_label: str | None = None,
    per_pbt_epoch: bool = False,
    extra_user_attrs: dict[str, Any] | None = None,
    hparam: str | None = None,
):
    params = params.copy()
    user_attrs = {
        "identifier": identifier,
        "vf_share_layers": params.pop("vf_share_layers"),
        "use_kl_loss": params.pop("use_kl_loss"),
        "analysis_step": analysis_step,
        "per_pbt_epoch": per_pbt_epoch,
        "submission_study": submission_label,
        "raw_metric": metric_result,
        "centered_metric": centered_metric,
        "_centered": centered_metric is not None,
        "hparam": hparam,
        "_large": is_large,
    }
    if extra_user_attrs:
        user_attrs.update(extra_user_attrs)
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
    if not trial:
        return
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


@overload
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
    load_studies_only: bool = False,
) -> dict[Any, optuna.Study]: ...


@overload
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
    load_studies_only: Literal["names_only"],
) -> dict[Any, tuple[str, Path]]: ...


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
    load_studies_only: bool | Literal["names_only"] = False,
) -> dict[Any, optuna.Study] | dict[Any, tuple[str, Path]]:
    """
    Create or load Optuna studies for one or many PBT experiments.

    Args:
        disable_checks: If True, disable Optuna's distribution checks. This is useful when
            loading studies where the hyperparameter distributions have changed over time.
    """
    assert load_if_exists, "Not loading existing is unstable"
    metric_to_check = metric
    if disable_checks:
        _disable_distribution_check()
    storage_path = Path(database_path or experiment_paths[0])
    storage = _get_storage(storage_path)
    if clear_experiment_cache:
        clear_tracking_study(storage)
        STORED_EXPERIMENTS.clear()
    tracking_study = get_experiment_track_study(storage)
    if clear_experiment_cache:
        assert not STORED_EXPERIMENTS
    study_names_only = load_studies_only == "names_only"
    studies: dict[Literal["global"] | int, optuna.Study] | dict[Literal["global"] | int, tuple[str, Path]] = {}
    if not study_each_epoch:
        global_study = get_optuna_study(
            storage, env, load_if_exists=load_if_exists, clear_study=clear_study, name_only=study_names_only
        )
        if isinstance(global_study, optuna.Study):
            global_study.set_user_attr("analysis_step", "global")
            studies[GLOBAL_STUDY] = global_study
        else:
            studies[GLOBAL_STUDY] = (global_study, storage_path)
    if study_each_epoch is not False:
        for epoch in sorted(set(range(1, 33)) | set(range(1, 9))):
            step = epoch * 8192 * 4
            step_study = get_optuna_study(storage, env, step, clear_study=clear_study, name_only=study_names_only)
            if isinstance(step_study, optuna.Study):
                step_study.set_user_attr("analysis_step", step)
                studies[step] = step_study
            else:
                studies[step] = (step_study, storage_path)

    step_keys = sorted(step for step in studies if isinstance(step, int))

    def _format_analysis_key(raw_key: Literal["global"] | int) -> str:
        if raw_key == GLOBAL_STUDY:
            key_label = GLOBAL_STUDY
        else:
            key_label = f"step={raw_key}"
        if env and env not in key_label:
            key_label = f"{env}_{key_label}"
        return key_label

    def _coerce_to_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return _coerce_to_int(value.iloc[0])
        if hasattr(value, "item"):
            try:
                return int(value.item())
            except Exception:  # noqa: BLE001
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_to_float(value: Any) -> float:
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        if isinstance(value, pd.Series):
            if value.empty:
                raise ValueError("Empty series cannot be coerced to float.")
            return float(value.iloc[0])
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _resolve_step_key(current_step: int, total_steps: int | None) -> int:
        if current_step in studies:
            return current_step
        if total_steps is not None and total_steps in studies:
            return total_steps
        if not step_keys:
            raise KeyError("No per-step studies configured.")
        return min(step_keys, key=lambda step: (abs(step - current_step), step))

    if load_studies_only:
        return {_format_analysis_key(key): study for key, study in studies.items()}

    for experiment_path in experiment_paths:
        logger.info("Checking path for experiments: %s", experiment_path)
        running_experiments = set()
        if dir_depth == 0:
            experiment_path = Path(experiment_path)  # noqa: PLW2901
            experiment_id = experiment_path.name.split("-")[-1]
            if experiment_id in STORED_EXPERIMENTS:
                logger.info("Skipping run %s as the id is already stored", experiment_id)
                continue
            if any(excl in str(experiment_path) for excl in excludes) or "cometml" in str(experiment_path):
                if "cometml" not in str(experiment_path):
                    logger.info(f"Excluding experiment path {experiment_path} due to exclude patterns.")  # noqa: G004
                continue
            experiment_subdirs = [experiment_path]
        else:
            experiment_subdirs = Path(experiment_path).glob(f"*/{'/'.join(['*'] * dir_depth)}")
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

            trials_to_add = {study: [] for study in studies.values()}
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
                if group_stat is None:
                    group_stat = _group_stat
                is_large_flag = False
                if group_stat not in ("batch_size", "train_batch_size_per_learner"):
                    batch_size_raw = df.iloc[0].config.get(
                        "train_batch_size_per_learner",
                        df.iloc[0].config.get("batch_size", df.iloc[0].get("batch_size")),
                    )
                    batch_size_value = _coerce_to_int(batch_size_raw)
                    is_large_flag = bool(batch_size_value is not None and batch_size_value >= 8192)
                else:
                    batch_size_raw = df.iloc[0].config.get(
                        "minibatch_size",
                        df.iloc[0].config.get("minibatch_size", df.iloc[0].get("minibatch_size")),
                    )
                    batch_size_value = _coerce_to_int(batch_size_raw)
                    is_large_flag = bool(batch_size_value is not None and batch_size_value >= 8192)

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
                # Consider using groupkey to take mean over duplicates.
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
                    if "grad_clip" in params and (pd.isna(params["grad_clip"]) or params["grad_clip"] == float("inf")):
                        params["grad_clip"] = None
                    assert params["grad_clip"] is None or not math.isnan(params["grad_clip"])
                    # Clean placeholders
                    if metric not in final_row:
                        raise KeyError(f"Metric column '{metric}' not found in combined dataframe")  # noqa: TRY301
                    try:
                        metric_result = float(final_row[metric].iloc[0])
                    except Exception:
                        metric_result = float(final_row[metric])
                    metric_centered: float | None = None
                    if per_pbt_epoch:
                        try:
                            shift_values = value_shifter_with_first.loc[final_row.config.pbt_epoch.item()]
                            metric_key_name = metric if isinstance(metric, str) else "-".join(metric)
                            reference_value = shift_values.get(metric_key_name)
                            if reference_value is None:
                                raise KeyError(metric_key_name, "not in shift values")
                            metric_centered = metric_result - _coerce_to_float(reference_value)
                        except KeyError as ke:
                            logger.error(
                                "Could not compute centered metric for %s at epoch %r: %r",
                                trial_identifier,
                                final_row.config.pbt_epoch,
                                ke,
                            )
                            metric_centered = None
                        except Exception:  # noqa: BLE001
                            # Most common KeyError as some epoch and main data is missing.
                            logger.exception("Could not compute centered metric for %s", trial_identifier)
                            remote_breakpoint()
                            metric_centered = None

                    if not per_pbt_epoch:
                        analysis_step_key: Literal["global"] | int = GLOBAL_STUDY
                    else:
                        current_step_value = _coerce_to_int(final_row.current_step)
                        if current_step_value is None:
                            logger.warning(
                                "Skipping trial %s as current_step could not be determined.", trial_identifier
                            )
                            no_errors = False
                            continue
                        total_steps_raw = final_row.config.get("cli_args", {}).get("total_steps")
                        total_steps_value = _coerce_to_int(total_steps_raw)
                        try:
                            analysis_step_key = _resolve_step_key(current_step_value, total_steps_value)
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "Could not map step %s for trial %s to configured studies.",
                                current_step_value,
                                trial_identifier,
                            )
                            no_errors = False
                            continue

                    study_to_add = studies.get(analysis_step_key)
                    if study_to_add is None:
                        logger.warning("No study configured for key %s", analysis_step_key)
                        no_errors = False
                        continue

                    filtered_params = (
                        {
                            k: v
                            for k, v in params.items()
                            if k in distributions or k in ("vf_share_layers", "use_kl_loss")
                        }
                        if distributions
                        else params
                    )

                    trial = create_finished_trial(
                        metric_result=metric_result,
                        identifier=trial_identifier,
                        params=filtered_params,
                        distributions=distributions,
                        centered_metric=metric_centered,
                        analysis_step="global" if analysis_step_key == GLOBAL_STUDY else analysis_step_key,
                        is_large=is_large_flag,
                        submission_label=submission_study,
                        per_pbt_epoch=per_pbt_epoch,
                        hparam=group_stat,
                    )
                    trials_to_add[study_to_add].append(trial)
                    if i % _REPORT_INTERVAL == 0:
                        if per_pbt_epoch:
                            logger.debug(
                                "Adding trial for run %s at step %s with metric %s - processed %d trials",
                                run_id,
                                analysis_step_key,
                                metric_result,
                                _REPORT_INTERVAL,
                            )
                        else:
                            logger.debug(
                                "Adding trial for run %s with metric %s - processed %d trials",
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

    return {_format_analysis_key(key): study for key, study in studies.items()}


PARAMS_TO_CHECK = {
    "batch_size",
    "lr",
    "gamma",
}


def _compose_submission_key(base_key: str, submission_label: str | None) -> str:
    # Study names should share the same key base
    base_key = re.sub(r"step=\d+_?", "", base_key)
    base_key = re.sub(r"_global", "", base_key)
    if not submission_label:
        return base_key
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", submission_label)
    return f"{base_key}|submission={safe_label}"


def _filter_trials_for_view(
    trials: Sequence[optuna.trial.FrozenTrial],
    *,
    analysis_step: int | str,
    centered: bool,
    large_mode: LargeMode,
    submission_label: str | None,
    use_kl_loss: bool | None,
    vf_share_layers: bool | None,
) -> list[optuna.trial.FrozenTrial]:
    filtered: list[optuna.trial.FrozenTrial] = []
    for trial in trials:
        attrs = trial.user_attrs
        if attrs.get("analysis_step", "global") != analysis_step:
            continue
        if submission_label is not None and attrs.get("submission_study") != submission_label:
            continue
        is_large_trial = bool(attrs.get("_large", False))
        if large_mode == "large" and not is_large_trial:
            continue
        if large_mode == "non_large" and is_large_trial:
            continue
        if vf_share_layers is not None and attrs.get("vf_share_layers") != vf_share_layers:
            continue
        if use_kl_loss is not None and attrs.get("use_kl_loss") != use_kl_loss:
            continue
        trial_copy = deepcopy(trial)
        raw_metric = attrs.get("raw_metric")
        if raw_metric is not None:
            trial_copy.value = raw_metric
        trial_copy.user_attrs = dict(trial_copy.user_attrs)
        trial_copy.user_attrs["view_centered"] = centered
        trial_copy.user_attrs["view_large_mode"] = large_mode
        trial_copy.user_attrs["view_submission"] = submission_label
        if centered:
            centered_metric = attrs.get("centered_metric")
            if centered_metric is None:
                continue
            trial_copy.value = centered_metric
        filtered.append(trial_copy)
    return filtered


_STEP_SETTINGS = {
    "vf_loss_coeff": 0.05,
    "kl_coeff": 0.005,
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
        logger.debug("Parameter %s has multiple distributions: %s", name, dists)
        is_log = LOG_SETTINGS.get(name, False)
        step: float | None = None
        if not is_log:
            step = _STEP_SETTINGS.get(name)
        # cannot have step and log
        if not step:
            step = None
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
        if all(isinstance(c, (int, float, np.floating, np.number)) for c in choices):
            if all(isinstance(c, int) for c in choices):
                new_dist = optuna.distributions.IntDistribution(
                    low=round_floats(min(choices)),
                    high=round_floats(max(choices)),
                    log=bool(is_log),
                )
            else:
                if min(choices) == 0 and is_log:
                    # log scale cannot include zero - this is the entropy coefficient.
                    choices.discard(0)
                new_dist = optuna.distributions.FloatDistribution(
                    low=round_floats(min(choices)), high=round_floats(max(choices)), log=bool(is_log), step=step
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


def __sort_columns(col: pd.Index):
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


def _get_importances(
    study: optuna.Study,
    evaluator: optuna.importance.BaseImportanceEvaluator,
    params: list[str],
    *,
    analysis_step: int | str,
    centered: bool,
    large_mode: LargeMode,
    submission_label: str | None,
    use_kl_loss: bool | None = None,
    vf_share_layers: bool | None = None,
    base_trials: Sequence[optuna.trial.FrozenTrial] | None = None,
):
    base_trials = (
        list(base_trials)
        if base_trials is not None
        else study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    )
    filtered_trials = _filter_trials_for_view(
        base_trials,
        analysis_step=analysis_step,
        centered=centered,
        large_mode=large_mode,
        submission_label=submission_label,
        use_kl_loss=use_kl_loss,
        vf_share_layers=vf_share_layers,
    )
    if not filtered_trials or len(filtered_trials) <= 1:
        return {}, []

    working_study = deepcopy(study)

    def _patched_get_trials(*args, **kwargs):
        if kwargs.get("deepcopy", True):
            return deepcopy(filtered_trials)
        return filtered_trials

    working_study.get_trials = _patched_get_trials  # type: ignore[assignment]
    try:
        completed_trials = _fix_distributions(working_study, params or PARAMS_TO_CHECK)
        importances = optuna.importance.get_param_importances(working_study, evaluator=evaluator, params=params)
    except ValueError as ve:
        if "dynamic search" not in str(ve):
            raise
        retry_study = deepcopy(working_study)
        retry_study.get_trials = _patched_get_trials  # type: ignore[assignment]
        completed_trials = _fix_distributions(retry_study, params or PARAMS_TO_CHECK)
        importances = optuna.importance.get_param_importances(retry_study, evaluator=evaluator, params=params)
        working_study = retry_study
    return importances, completed_trials


def _analyze_single_study(
    key: Any,
    study: optuna.Study | tuple[StudyName, DatabasePath],
    evaluators: dict[str, optuna.importance.BaseImportanceEvaluator],
    params: list[str] | None,
    excludes: Collection[Literal["kl_loss", "no_kl_loss", "vf_share_layers", "no_vf_share_layers"]] = (),
    base_trials: Sequence[optuna.trial.FrozenTrial] | None = None,
    workers_per_study: int | None = 4,
    *,
    submission_filter: str | Collection[str] | bool = True,
) -> list[dict[str, Any]]:
    """Analyze one study and return result rows."""
    rows: list[dict[str, Any]] = []
    logger.info("Analyzing study for key: %s", key)
    if isinstance(study, tuple):
        # load study
        study_name, storage_path = study
        storage = _get_storage(storage_path)
        if isinstance(study_name, tuple):
            study_name = study_name[0]
        study = optuna.load_study(study_name=study_name, storage=storage)
    analysis_step_value: int | str = study.user_attrs.get("analysis_step", "global")
    base_trials = (
        list(base_trials)
        if base_trials is not None
        else study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    )
    submission_labels = sorted(
        {
            cast("str", label)
            for label in (trial.user_attrs.get("submission_study") for trial in base_trials)
            if isinstance(label, str) and label  # and (not submission_filter or submission_filter in label)
        }
    )
    has_centered = any(trial.user_attrs.get("centered_metric") is not None for trial in base_trials)
    has_large = any(trial.user_attrs.get("_large", False) for trial in base_trials)

    view_specs: list[tuple[bool, LargeMode, str | None]] = []

    def _append_spec(spec: tuple[bool, LargeMode, str | None]) -> None:
        if spec not in view_specs:
            view_specs.append(spec)

    large_modes: list[LargeMode] = ["all"]
    if has_large:
        large_modes.extend(["large", "non_large"])

    for large_mode in large_modes:
        if not submission_filter:
            _append_spec((False, large_mode, None))
            if has_centered:
                _append_spec((True, large_mode, None))
        if submission_filter is True:
            continue  # filter out all submissionsl
        for submission in submission_labels:
            # submission_filter == False includes all submissions
            if submission_filter:
                if isinstance(submission_filter, str):
                    if submission_filter not in submission:
                        logger.info("Skipping submission label %s due to filter %s", submission, submission_filter)
                        continue
                else:
                    # REVERSE: if present they are added!
                    for subfilter in submission_filter:
                        if subfilter in submission:
                            break
                    else:
                        logger.info(
                            "Skipping submission label %s not found in collection %s", submission, submission_filter
                        )
                        continue
            _append_spec((False, large_mode, submission))
            if has_centered:
                _append_spec((True, large_mode, submission))

    tasks: list[ImportanceTask] = []
    for filter_kl, filter_vf_share in product([None, True, False], [None, True, False]):
        if filter_kl is False and "no_kl_loss" in excludes:
            continue
        if filter_kl is True and "kl_loss" in excludes:
            continue
        if filter_vf_share is False and "no_vf_share_layers" in excludes:
            continue
        if filter_vf_share is True and "vf_share_layers" in excludes:
            continue
        for centered_flag, large_mode, submission_label in view_specs:
            for evaluator_name, evaluator_template in evaluators.items():
                tasks.append(
                    ImportanceTask(
                        filter_kl,
                        filter_vf_share,
                        centered_flag,
                        large_mode,
                        submission_label,
                        evaluator_name,
                        evaluator_template,
                    )
                )

    def _run_task(task: ImportanceTask):
        (
            filter_kl,
            filter_vf_share,
            centered_flag,
            large_mode,
            submission_label,
            evaluator_name,
            evaluator_template,
        ) = task
        evaluator_instance = deepcopy(evaluator_template)
        importances, completed_trials = _get_importances(
            study,
            evaluator_instance,
            params,
            analysis_step=analysis_step_value,
            centered=centered_flag,
            large_mode=large_mode,
            submission_label=submission_label,
            use_kl_loss=filter_kl,
            vf_share_layers=filter_vf_share,
            base_trials=base_trials,
        )
        if not completed_trials:
            return [], None

        submission_value = submission_label or "default"
        step_value = analysis_step_value
        rows_local: list[dict[str, Any]] = []
        for param, importance in importances.items():
            variant_key = _compose_submission_key(str(key), submission_label)
            rows_local.append(
                {
                    "param": param,
                    "importance": importance,
                    "evaluator_name": evaluator_name,
                    "key": variant_key,
                    "centered": centered_flag,
                    "large_mode": large_mode,
                    "step": step_value,
                    "number_of_trials": len(completed_trials),
                    "study_name": study.study_name,
                }
            )
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        report_lines = [
            f"Hyperparameter importances for study {study.study_name} "
            f"(kl_loss: {filter_kl}, vf_share_layers: {filter_vf_share}, "
            f"centered: {centered_flag}, large_mode: {large_mode}, submission: {submission_value}) "
            f"({evaluator_name}) with {len(completed_trials)} trials:",
        ]
        for param, importance in sorted_importances:
            report_lines.append(f"  {param}: {importance:.5f}")
        return rows_local, "\n".join(report_lines)

    try:
        if tasks:
            if False:
                cpu_count = os.cpu_count() or 4
                inner_cap = max(1, workers_per_study or cpu_count)
                worker_count = min(len(tasks), inner_cap)
                with ThreadPoolExecutor(max_workers=worker_count) as combo_executor:
                    for combo_rows, report in combo_executor.map(_run_task, tasks):
                        if combo_rows:
                            rows.extend(combo_rows)
                        if report:
                            logger.info("%s", report)
            else:
                # do sequential as CPU bound
                for task in tasks:
                    combo_rows, report = _run_task(task)
                    if combo_rows:
                        rows.extend(combo_rows)
                    if report:
                        logger.info("%s", report)
        else:
            logger.warning("No tasks generated for study with key %s and submission filter %s", key, submission_filter)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to analyze study for key %s", key)
        try:
            if len(study.get_trials()) > 1:
                remote_breakpoint()
        except Exception:  # noqa: BLE001
            remote_breakpoint()
    return rows


def optuna_analyze_studies(
    studies: Mapping[Any, optuna.Study | tuple[StudyName, DatabasePath]] | Iterable[tuple[StudyName, DatabasePath]],
    output_path: str | Path | None,
    params: list[str] | None = None,
    max_workers: int | None = None,
    excludes: Collection[Literal["kl_loss", "no_kl_loss", "vf_share_layers", "no_vf_share_layers"]] = (),
    progress_bar: tqdm | None = None,
    *,
    submission_filter: str | Collection[str] | bool = True,
) -> pd.DataFrame | None:
    logger.info(
        "Starting optuna study analysis for %s studies",
        len(studies) if isinstance(studies, (Mapping, Sized)) else "an unknown number of",
    )
    from optuna.importance import (  # noqa: PLC0415
        FanovaImportanceEvaluator,
        MeanDecreaseImpurityImportanceEvaluator,
        PedAnovaImportanceEvaluator,
    )

    evaluators = {
        "MeanDecreaseImpurity": MeanDecreaseImpurityImportanceEvaluator(),
        "PedAnovaLocal": PedAnovaImportanceEvaluator(evaluate_on_local=True),  # default
        "Fanova(default)": FanovaImportanceEvaluator(),  # default
        # "MeanDecreaseImpurity8": MeanDecreaseImpurityImportanceEvaluator(n_trees=12, max_depth=12),
        "PedAnovaGlobal": PedAnovaImportanceEvaluator(evaluate_on_local=False),
        # "Fanova8": FanovaImportanceEvaluator(n_trees=12, max_depth=12),
    }

    all_results: list[dict[str, Any]] = []

    # Parallelize per-study processing
    inner_workers = None
    try:
        number_studies = len(studies)  # pyright: ignore[reportArgumentType]
    except TypeError:
        number_studies = 33
    if not max_workers and not isinstance(studies, (Mapping, Sized)):
        logger.warning("max_workers not set and studies is not sized; defaulting to 4 workers.")
        max_workers = 4
        inner_workers = 4
    workers = max_workers or max(1, min((os.cpu_count() or 6), number_studies, 20))

    # Convert generator to list to avoid blocking during dict comprehension
    # Studies passed as tuples will be loaded in parallel by worker threads
    if not isinstance(studies, Mapping):
        logger.info("Converting study generator to list...")
        study_items = list(studies)
        logger.info("Prepared %s study references for parallel loading", len(study_items))
    else:
        study_items = list(studies.items())
        logger.info("Using %s studies from mapping", len(study_items))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        logger.info(
            "Submitting %s study analysis tasks to ThreadPoolExecutor with %s workers", len(study_items), workers
        )
        futures = {
            executor.submit(
                _analyze_single_study,
                key,
                study if not isinstance(study, (Path, str)) else (key, study),
                evaluators,
                params,
                excludes,
                None,
                workers_per_study=inner_workers or max(4, (os.cpu_count() or 6)),
                submission_filter=submission_filter,
            ): key
            for key, study in study_items
        }
        logger.info("Submitted %s futures, waiting for completion...", len(futures))
        try:
            for future in as_completed(futures):
                if progress_bar is not None:
                    progress_bar.update(1)
                logger.info("Completed analysis for study with key: %s", futures[future])
                rows = future.result()
                if rows:
                    all_results.extend(rows)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received: cancelling all running futures.")
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    # Create combined DataFrame with multilevel columns
    if not all_results:
        return None
    logger.info("Combining results from %d analyses", len(all_results))
    results_df = pd.DataFrame(all_results)

    pivot_df = results_df.pivot_table(
        index="param", columns=list(STUDY_COLUMN_NAMES), values="importance", fill_value=0.0
    )
    pivot_df.columns = pivot_df.columns.map(__try_step_cast)
    pivot_df = pivot_df.sort_index(axis=1, key=__sort_columns)

    if output_path is not None:
        # should not save to a unique file when we write only for submissions
        try:
            save_path = Path(output_path)
            if save_path.is_dir():
                save_path = save_path / "hyperparameter_importance_combined.csv"
            else:
                raise ValueError("output_path must be a directory to save importances")
            pivot_df.to_csv(save_path, index=True)
            # Cast all integers to strings again if "global" is in columns
            # When called via plot optuna will save the combined file
            save_analysis_to_parquet(pivot_df, save_path.with_suffix(".parquet"))
            logger.info("Saved combined importances to %s", save_path)
        except Exception:
            logger.exception("Could not export combined importances to %s", output_path)
    return pivot_df


_check_distribution_compatibility = optuna.distributions.check_distribution_compatibility


def _disable_distribution_check():
    # Alternative: move all trials to a new study with the changed distribution
    optuna.distributions.check_distribution_compatibility = lambda a, b: None


def _clean_key_col(cols):
    return (
        cols[0],
        re.sub(r"-v(\d)_\|", r"-v\1|", re.sub(r"_?step=\d+", "", re.sub("_global", "", cols[1]))).rstrip("_"),
        *cols[2:],
    )


def __sort_index(idx: str | Any):
    if not isinstance(idx, str):
        return str
    if idx in ("batch_size", "minibatch_size", "accumulate_gradients_every", "minibatch_scale"):
        return 0
    if "num_env" in idx:
        return 1
    if idx in ("lr", "learning_rate"):
        return 2
    if "coeff" in idx:
        return 3
    if "clip" in idx:
        if "vf" not in idx:
            return 4
        return 5
    if "vf" in idx:
        return 5
    return 5


def _sort_index(idx: pd.Index):
    return idx.map(__sort_index)


def plot_importance_studies(
    studies: Mapping[Any, optuna.Study | tuple[str, Path]] | pd.DataFrame,
    output_path: str | Path,
    params: list[str] | None = None,
    *,
    env: str,
    submission_filter: str | bool | Collection[str] = True,
    title: bool = False,
) -> list[Path] | None:
    # note there is also import optuna.visualization as optuna_vis
    if not isinstance(studies, pd.DataFrame):
        # remote_breakpoint()
        study_results = optuna_analyze_studies(studies, output_path, params=params, submission_filter=submission_filter)
    else:
        study_results = studies
    if study_results is None:
        return None
    study_results.columns = study_results.columns.map(_clean_key_col)
    if study_results.columns.nlevels > 1:
        importances = {
            evaluator_name: study_results[evaluator_name] for evaluator_name in study_results.columns.levels[0]
        }
    else:
        importances = {"": study_results}
    output_dir = Path(output_path)
    written_paths: list[Path] = []

    def _iter_level(frame: pd.DataFrame, level_name: str) -> list[tuple[Any, pd.DataFrame]]:
        if level_name not in frame.columns.names:
            return [(None, frame)]
        unique_values = list(dict.fromkeys(frame.columns.get_level_values(level_name)))
        level_frames: list[tuple[Any, pd.DataFrame]] = []
        for value in unique_values:
            try:
                sub_frame = frame.xs(value, level=level_name, axis=1)
            except KeyError:
                continue
            if isinstance(sub_frame, pd.Series):
                sub_frame = sub_frame.to_frame()
            level_frames.append((value, sub_frame))
        return level_frames

    for evaluator_name, importance_df in importances.items():
        if "key" not in importance_df.columns.names and all(name is None for name in importance_df.columns.names):
            importance_df.columns.names = STUDY_COLUMN_NAMES[1:]
        for key in importance_df.columns.get_level_values("key").unique():
            key_df = importance_df[key]
            if env in key:
                key_with_env = key
            else:
                key_with_env = f"{env}_{key}"
            key_with_env = key_with_env.replace("|", " ").rstrip("_")
            submission_tag = "default"
            if "|submission=" in str(key):
                submission_tag = str(key).split("|submission=", 1)[1]

            for centered_value, centered_df in _iter_level(key_df, "centered"):
                centered_bool = bool(centered_value) if centered_value is not None else False
                for large_value, large_df in _iter_level(centered_df, "large_mode"):
                    large_mode = str(large_value) if large_value is not None else "all"
                    fig = None
                    try:
                        view_df = large_df.sort_index(axis=1, key=__sort_columns)
                        if view_df.empty:
                            continue
                        if centered_bool is False and "global" in view_df.columns:
                            assert view_df.columns[-1] == "global", (
                                f"Expected last column to be 'global' not {view_df.columns[-1]}"
                            )
                        # Group index to have batch_size, vf and other parameters nearby.
                        view_df = view_df.sort_index(axis=0, key=_sort_index)

                        add_global = "global" in view_df.columns
                        fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(view_df))))
                        cmap = sns.color_palette("magma", as_cmap=True)
                        max_mask = view_df.eq(view_df.max())
                        sns.heatmap(
                            view_df,
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
                        xlabels = list(view_df.columns)
                        new_labels: list[str] = []
                        for idx, column in enumerate(xlabels):
                            if column == "global":
                                continue
                            if idx % 8 == 0 or idx == len(xlabels) - (2 if add_global else 1):
                                try:
                                    step_value = int(column)
                                    label = f"{step_value / 1e6:.1f}"
                                except (TypeError, ValueError):
                                    label = str(column)
                                new_labels.append(label)
                            else:
                                new_labels.append("")
                        if add_global:
                            new_labels.append("G")
                        ax.set_xticklabels(new_labels, rotation=0, ha="center")
                        ytick_labels = ax.get_yticklabels()
                        ytick_labels = [lbl.get_text() for lbl in ytick_labels]
                        ytick_labels = [
                            " ".join(
                                # capitalize
                                ylabel.replace("train_batch_size_per_per_learner", "batch size")
                                .replace("accumulate_gradients_every", "grad accumu.")
                                .split("_"),
                            )
                            for ylabel in ytick_labels
                        ]

                        ax.set_yticklabels(ytick_labels, rotation=0, va="top")
                        ax.set(ylabel=None)

                        for col_idx, column in enumerate(view_df.columns):
                            column_mask = max_mask[column]
                            if column_mask.all() or int(column_mask.sum()) <= 1:
                                continue
                            max_rows = view_df.index[column_mask]
                            for row in max_rows:
                                row_idx = view_df.index.get_loc(row)
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
                        if title:
                            plt.title(f"Hyperparameter Importance {key_with_env} ({evaluator_name})")
                        plt.xlabel("Step")
                        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key_with_env)).replace("pbt_study_env_", "")
                        safe_submission_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", submission_tag)
                        if submission_tag in safe_key:
                            safe_submission_tag = ""
                        else:
                            safe_submission_tag = f"submission={safe_submission_tag}"
                        safe_evaluator = re.sub(r"[^A-Za-z0-9_.-]", "_", evaluator_name).rstrip("_.-")
                        subdirs = [submission_tag] if submission_tag != "default" else []
                        subdirs.extend(
                            [
                                "centered=True" if centered_bool else "centered=False",
                                f"{large_mode}",
                            ]
                        )
                        if "kl_loss" in key:
                            subdirs.append("no_kl_loss" if "no_kl_loss" in key else "kl_loss")
                        if "vf_share_layers" in key:
                            subdirs.append("no_vf_share_layers" if "no_vf_share_layers" in key else "vf_share_layers")

                        if "default" in safe_submission_tag:
                            safe_submission_tag = "all"
                        file_name = f"hp_importance_{safe_key}-{safe_submission_tag}-{safe_evaluator}.pdf"
                        if large_mode != "all":
                            file_name += f"_{large_mode}"
                        file_name = file_name.replace("(default)", "").replace("__", "_").replace("--", "-")
                        file_path = Path(output_dir, *subdirs, file_name)
                        assert file_path not in written_paths, "File path already written: %s" % file_path
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(
                            file_path,
                            format="pdf",
                            bbox_inches="tight",
                        )
                        written_paths.append(file_path)
                        logger.info(
                            "Saved heatmap for key '%s' centered %s large_mode %s submission=%s at\n'%s'",
                            key,
                            centered_bool,
                            large_mode,
                            submission_tag,
                            file_path,
                            stacklevel=2,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "Could not plot heatmap for key=%s centered=%s large_mode=%s submission=%s",
                            key,
                            centered_bool,
                            large_mode,
                            submission_tag,
                        )
                        remote_breakpoint()
                    finally:
                        if fig:
                            plt.close(fig)
    return written_paths


def __create_zipfile(suffix: str = "") -> ZipFile:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        suffix += "_"
    zippath = f"outputs/shared/optuna_hp_study_plots_{suffix}{timestamp}.zip"
    zipfile = ZipFile(zippath, "w")

    def __close_zipfile(path=zippath):
        if zipfile:
            zipfile.close()
            logger.info("Closed zip file. %s created.", path)

    atexit.register(__close_zipfile)
    return zipfile


if __name__ == "__main__":
    # Example invocation:
    #   python ray_utilities/visualization/importance.py \
    #       experiments/submissions_pbt_fine_base.yaml \
    #       vf_share_layers kl_loss no_kl_loss no_vf_share_layers \
    #       experiments/submissions_pbt_grouped.yaml kl_loss no_kl_loss \
    #       experiments/submissions_ppo_with_kl.yaml no_kl_loss no_vf_share_layers
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
        nargs="*",  # cannot use const
        default=None,
        help=(
            "Clear the experiment tracking study. Without values, clears once on the first submission. "
            "With one or more env names, clears once on the first submission but only for matching envs."
        ),
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
    parser.add_argument("--try-plot-only", action="store_true", help="Try to plot only from existing analysis files.")
    parser.add_argument("--analyze-only", "-a", action="store_true", help="Only analyze studies, do not plot.")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing analysis files.")
    parser.add_argument(
        "--update-db-only", action="store_true", help="Only update the Optuna database, do not analyze or plot."
    )
    parser.add_argument("--zip", "-z", action="store_true", help="Whether to zip the output analysis files.")
    parser.add_argument("--add-title", "-t", action="store_true", help="Whether to add titles to the plots.")

    args = parser.parse_args()
    file_excludes = load_excludes()
    assert not args.update_db_only or (not args.plot_only or not args.analyze_only), (
        "--update-db-only cannot be combined with --analyze-only or --plot-only; "
        "--analyze-only cannot be combined with --plot-only."
    )
    assert not args.update_db_only or not args.zip, "--update-db-only cannot be combined with --zip."

    # possibly for repair sqlite3 "outputs/shared/optuna_hp_study.db" ".dump" | sqlite3 new.db

    logger = nice_logger(logger, logging.DEBUG)
    # Interpret --clear-experiment-cache values: None = not provided; [] = provided without envs (global);
    # [envs...] = provided env names to clear (only on first submission in submissions mode)
    _clear_cache_arg = args.clear_experiment_cache  # None | list[str]
    clear_cache_present: bool = _clear_cache_arg is not None  # if present and clear_cache_envs is None clear all
    clear_cache_envs: set[str] | None = None
    if clear_cache_present and len(_clear_cache_arg) > 0:
        clear_cache_envs = set(_clear_cache_arg)
    distributions = load_distributions_from_json(write_distributions_to_json(default_distributions), as_optuna=True)
    distributions.pop("test", None)
    distributions.pop("minibatch_scale", None)
    if "vf_clip_param" in distributions and 0.0 not in distributions["vf_clip_param"].choices:
        # We changed this distribution
        distributions["vf_clip_param"].choices = (0.0, *distributions["vf_clip_param"].choices)
    # Help type-checkers: explicit cast to Optuna distribution mapping

    # TODO: overlload load_distributions_from_json
    optuna_dists = cast("dict[str, optuna.distributions.BaseDistribution]", distributions)
    # Make studies for each submission file and combined studies
    if "DEBUG" in os.environ or "test" in sys.argv:
        DEBUG = True
        paths = ("outputs/shared/experiments/Default-mlp-Acrobot-v1",)
        _env = "Acrobot-v1"
        paths = [
            "outputs/shared/experiments/Default-mlp-HumanoidStandup-v5",
            "outputs/shared_backup/needs_sync/Default-mlp-HumanoidStandup-v5",
            "outputs/shared_backup/Default-mlp-HumanoidStandup-v5",
        ]
        _env = "HumanoidStandup-v5"
        # Determine whether to clear experiment cache for this env (DEBUG path)
        _clear_now = clear_cache_present and (clear_cache_envs is None or _env in clear_cache_envs)
        ot_studies = optuna_create_studies(
            *paths,
            # TESTING
            database_path=f"outputs/shared/optuna_hp_study_{_env}.db",
            env=_env,
            dir_depth=1,
            distributions=distributions,
            load_if_exists=True,
            study_each_epoch=None,
            submission_file_path=Path("experiments"),
            clear_experiment_cache=_clear_now,
            excludes=file_excludes,
            # TESTING
        )
        studies = optuna_analyze_studies(ot_studies, output_path=paths[0], params=list(distributions.keys()))
        # Want to plot for each env.
        if studies is None:
            sys.exit(1)
        plot_importance_studies(
            studies, output_path=paths[0], params=list(distributions.keys()), env=_env, title=args.add_title
        )
        sys.exit(0)
    # clear_study = args.clear_study
    if any(p.endswith(".yaml") for p in args.envs):
        # yaml file passed positional
        # assert default value:
        paths_action = next(a for a in parser._actions if a.dest == "paths")
        assert args.paths == paths_action.default, "When passing YAML files positionally, do not set --paths."
        args.paths = args.envs
        args.envs = []

    envs = args.envs or [
        "Acrobot-v1",
        "CartPole-v1",
        "LunarLander-v3",
        "Hopper-v5",
        "HumanoidStandup-v5",
        "Humanoid-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
        "Swimmer-v5",
        "InvertedDoublePendulum-v5",
        "InvertedPendulum-v5",
        "Reacher-v5",
        "Pusher-v5",
        "Ant-v5",
    ]

    zipfile = None
    saved_files = []
    zip_suffix = ""

    if any(p.endswith(".yaml") for p in args.paths):
        # ------------ Submission File(s) -------------------

        allowed_filters = ("no_kl_loss", "kl_loss", "vf_share_layers", "no_vf_share_layers")
        assert all(
            p.endswith(".yaml") or p in ("no_kl_loss", "kl_loss", "vf_share_layers", "no_vf_share_layers")
            for p in args.paths
        ), "Either all or none of the paths must be YAML files."
        yaml_paths: list[Path] = []
        submission_excludes: dict[
            str, set[Literal["kl_loss", "no_kl_loss", "vf_share_layers", "no_vf_share_layers"]]
        ] = {}
        last_yaml_path = None
        for path_str in args.paths:
            if path_str.endswith(".yaml"):
                yaml_p = Path(path_str)
                assert yaml_p.exists() or yaml_p == "submissions.yaml"
                yaml_paths.append(yaml_p)
                last_yaml_path = yaml_p
                submission_excludes[last_yaml_path.stem] = set()
                continue
            if last_yaml_path is None:
                raise ValueError("The first path must be a yaml file")
            submission_excludes[last_yaml_path.stem].add(path_str)
        # Load experiment groups from YAML files
        paths = []
        all_run_paths: dict[str, Path] = {}
        all_run_infos: dict[str, SubmissionRun] = {}
        submissions_map = {}
        args.single = True
        for yaml_path in yaml_paths:
            run_paths = dict(get_run_directories_from_submission(yaml_path))
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
                    if info["run_key"] in args.envs:  # Limitation: this branch ignores multi HP tuning
                        keep.add(info["run_id"])
                run_paths = {k: v for k, v in run_paths.items() if k in keep}
            all_run_paths.update(run_paths)
            all_run_infos.update(run_infos)
            zip_suffix += re.sub(r"submissions?_?", "", yaml_path.stem).replace("pbt_", "")
        if args.zip:
            zipfile = __create_zipfile(zip_suffix)

        # group by env and yaml_path to load less studies
        clear_studies = args.clear_study
        all_studies: dict[str, optuna.Study | tuple[str, Path]] = {}
        submission_names = set()
        first_submission = True

        for yaml_path in yaml_paths:
            submission_name = (
                Path(yaml_path)
                .name.removesuffix(".yaml")
                .removeprefix("submissions")
                .removeprefix("submission")
                .strip("_")
            )
            if submission_name:
                # can use submissions.yaml --plot-only
                submission_names.add(submission_name)
            # Process each environment in parallel using per-env databases
            if args.plot_only:
                continue

            def _process_env(
                env: str,
                *,
                submission_name_param: str,
                clear_studies_val: str | bool | None,
                clear_exp_cache_now: bool,
                yaml_path_param: Path,
            ) -> tuple[
                str,
                dict[Any, optuna.Study] | dict[Any, tuple[str, Path]],
                EnvAnalysisProcessTask | None,
                list[Path] | None,
            ]:
                parquet_file = Path(
                    f"outputs/shared/experiments/Default-mlp-{env}/hyperparameter_importance_{submission_name_param}.parquet"
                )
                env_paths = [
                    p
                    for rid, p in all_run_paths.items()
                    if all_run_infos[rid]["run_key"] == env
                    and all_run_infos[rid]["submission_name"] == submission_name_param
                ]
                if not env_paths:
                    logger.warning("No runs found for env %s in provided YAML files.", env)
                    return env, {}, None, None
                clear_this_study = (
                    clear_studies_val
                    if clear_studies_val is not None
                    else make_study_name("pbt_study", env, suffix=f"_{submission_name_param}")
                )
                # Per-env database name, with optional _sympol suffix when submission_name contains 'sympol'
                db_suffix = "_sympol" if "sympol" in submission_name_param.lower() else ""
                database_path = f"outputs/shared/optuna_hp_study_{env}{db_suffix}.db"

                if args.plot_only:
                    # DEPRECTED: Unreachable!
                    data_file = parquet_file
                    if not data_file.exists():
                        data_file = (
                            Path(f"outputs/shared/experiments/Default-mlp-{env}")
                            / "hyperparameter_importance_combined.parquet"
                        )
                    if not data_file.exists():
                        logger.error("Plot only specified but data file %s does not exist.", data_file)
                        return env, {}, None, None
                    importance_results = load_study_results(data_file)
                    if importance_results is None:
                        logger.error("Plot only specified but could not load results from %s.", data_file)
                        outfiles = None
                    else:
                        # PLOT ONLY
                        param_choices = list(distributions.keys())
                        if env not in ("CartPole-v1", "Acrobot-v1", "LunarLander-v3", "Hopper-v5"):
                            param_choices.remove("vf_loss_coeff")
                            param_choices.remove("entropy_coeff")
                        # PLOT ONLY
                        outfiles = plot_importance_studies(
                            importance_results,
                            output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                            params=param_choices,
                            env=env,
                            title=args.add_title,
                        )
                    studies = optuna_create_studies(
                        *env_paths,
                        database_path=database_path,
                        env=env,
                        dir_depth=0,
                        distributions=optuna_dists,
                        load_if_exists=True,
                        study_each_epoch=None,
                        submission_file_path=None,
                        clear_experiment_cache=clear_exp_cache_now,
                        clear_study=cast("Sequence[str] | Literal[False]", clear_this_study),
                        excludes=file_excludes,
                        metric="episode_reward_mean",
                        submission_study=submission_name_param,
                        load_studies_only=True,
                    )
                    return env, studies, None, outfiles
                # Non-plot-only path

                has_cached_results = bool(args.try_plot_only and parquet_file.exists())
                if has_cached_results and load_study_results(parquet_file) is None:
                    # Outdated version; does not mean we need to query all trials again,
                    # just need to build analysis and parquet file again
                    has_cached_results = False
                load_studies_only: Literal["names_only"] | bool = (
                    "names_only" if has_cached_results or args.analyze_only else False
                )
                _clear_this_study: Sequence[str] | Literal[False] = clear_this_study
                studies: dict[str | int, optuna.study.Study] | dict[str | int, tuple[str, Path]] = (
                    optuna_create_studies(
                        *env_paths,
                        database_path=database_path,
                        env=env,
                        dir_depth=0,
                        distributions=optuna_dists,
                        load_if_exists=True,
                        study_each_epoch=None,
                        submission_file_path=None,
                        clear_experiment_cache=clear_exp_cache_now,
                        clear_study=cast("Any", _clear_this_study),
                        excludes=file_excludes,
                        metric="episode_reward_mean",
                        submission_study=submission_name_param,
                        load_studies_only=load_studies_only,
                    )
                )
                study_names = [
                    study.study_name if isinstance(study, optuna.Study) else study[0] for study in studies.values()
                ]
                if args.update_db_only:
                    return env, studies, None, None
                analysis_task = EnvAnalysisProcessTask(
                    env=env,
                    submission_name=submission_name_param,
                    database_path=database_path,
                    study_names=study_names,
                    parquet_file=str(parquet_file),
                    param_names=list(distributions.keys()),
                    excludes=frozenset(submission_excludes[yaml_path_param.stem]),
                    output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                    has_cached_results=has_cached_results,
                )
                return env, studies, analysis_task, None

            process_workers = max(1, min(len(envs), int((os.cpu_count() or 2) * 0.75)))
            max_process_workers = min(len(envs), 2 * (os.cpu_count() or 2), 28)
            with (
                ProcessPoolExecutor(max_workers=process_workers) as process_executor,
                ThreadPoolExecutor(max_workers=max_process_workers) as executor,
            ):
                tqdm_indices = set(range(max_process_workers))
                tqdm_indices_used = set()
                process_futures: set[Future[EnvAnalysisProcessResult]] = set()
                futures = {
                    # For the first submission, pass clear_studies as-is. For later submissions,
                    # if clearing was requested (True or "all"), clear only the submission-specific studies.
                    executor.submit(
                        _process_env,
                        env,
                        submission_name_param=submission_name,
                        clear_studies_val=(
                            clear_studies
                            if first_submission
                            else (
                                make_study_name("pbt_study", env, suffix=f"_{submission_name}")
                                if (clear_studies is True or clear_studies == "all")
                                else clear_studies
                            )
                        ),
                        clear_exp_cache_now=(
                            first_submission
                            and clear_cache_present
                            and (clear_cache_envs is None or env in clear_cache_envs)
                        ),
                        yaml_path_param=yaml_path,
                    ): env
                    for idx, env in enumerate(envs)
                }
                try:
                    not_done_envs = set(envs)
                    not_done_tasks = {}
                    env_studies: dict[Any, optuna.Study] | dict[Any, tuple[str, Path]]
                    # _process_env execution
                    for future in as_completed(futures):
                        _env, env_studies, analysis_task, outfiles = future.result()
                        study_update = {
                            k: (
                                study if isinstance(study, optuna.Study) else (study, Path(analysis_task.database_path))
                            )
                            for k, study in env_studies.items()
                        }
                        all_studies.update(study_update)
                        if outfiles and zipfile:  # have no outfiles here currently
                            for outfile in outfiles:
                                if outfile.exists():
                                    zipfile.write(outfile, arcname=outfile.name)
                                    saved_files.append(outfile)
                        if analysis_task:
                            # Submit and immediately wait to avoid queue deadlock
                            logger.info("Submitting analysis process for env %s submission %s", _env, submission_name)
                            try:
                                tqdm_idx = min(tqdm_indices - tqdm_indices_used)
                            except ValueError:
                                tqdm_idx = 0
                            tqdm_indices_used.add(tqdm_idx)

                            # Submit the future
                            not_done_tasks[analysis_task.env] = analysis_task
                            process_future = process_executor.submit(
                                _run_env_analysis_process, analysis_task, tqdm_idx=None
                            )
                            process_futures.add(process_future)

                            # Immediately drain completed futures to prevent queue buildup
                            timeouts = 0
                            while len(process_futures) >= process_workers:
                                logger.info(
                                    "Draining completed processes... tasks in flight: %s/%s",
                                    len(process_futures),
                                    process_workers,
                                )
                                try:
                                    done_futures, process_futures = wait(
                                        process_futures, return_when=FIRST_COMPLETED, timeout=900
                                    )
                                except TimeoutError:
                                    logger.error("No analysis processes completed within the last 15 min")
                                    timeouts += 1
                                    if timeouts >= 3:
                                        logger.error("Multiple timeouts reached while waiting for analysis processes.")
                                        if process_executor._processes:
                                            try:
                                                for p in process_executor._processes.values():
                                                    os.kill(p.pid, signal.SIGKILL)  # or SIGINT / SIGTERM
                                            except Exception as e:
                                                logger.error(
                                                    "Error killing process: %s. Not completed envs %s, Tasks %s ",
                                                    e,
                                                    not_done_envs,
                                                    not_done_tasks,
                                                )
                                            raise
                                else:
                                    for done_future in done_futures:
                                        result: EnvAnalysisProcessResult = done_future.result()
                                        not_done_envs.discard(result.env)
                                        not_done_tasks.pop(result.env, None)
                                        logger.info(
                                            "Analysis process for env %s completed. Remaining: %d",
                                            result.env,
                                            len(not_done_envs),
                                        )
                                        tqdm_indices_used.discard(result.tqdm_idx)
                                        if result.outfiles and zipfile:
                                            for outfile in result.outfiles:
                                                if outfile.exists():
                                                    zipfile.write(outfile, arcname=outfile.name)
                                                    saved_files.append(outfile)
                    # complete all processes
                    if process_futures:
                        logger.info("Waiting for remaining %s analysis processes to complete...", len(process_futures))
                        try:
                            for process_future in as_completed(process_futures, timeout=1800):
                                result = process_future.result()
                                not_done_envs.discard(result.env)
                                not_done_tasks.pop(result.env, None)
                                logger.info(
                                    "Analysis process for env %s completed. Remaining: %d",
                                    result.env,
                                    len(not_done_envs),
                                )
                                if result.outfiles and zipfile:
                                    for outfile in tqdm(result.outfiles, desc="Adding to zip", unit="file"):
                                        if outfile.exists():
                                            zipfile.write(outfile, arcname=outfile.name)
                                            saved_files.append(outfile)
                        except TimeoutError:
                            logger.error("Not all futures completed within the timeout period.")
                        else:
                            logger.info(
                                "All %s analysis processes completed for submission %s",
                                len(process_futures),
                                submission_name,
                            )

                        # Now generate plots in main process (not in workers) to avoid matplotlib issues
                        logger.info("Generating plots in main process for submission %s...", submission_name)
                        for env in envs:
                            parquet_file = Path(
                                f"outputs/shared/experiments/Default-mlp-{env}/hyperparameter_importance_{submission_name}.parquet"
                            )
                            param_choices = list(distributions.keys())
                            if env not in ("CartPole-v1", "Acrobot-v1", "LunarLander-v3", "Hopper-v5"):
                                param_choices = param_choices.copy()
                                param_choices.remove("vf_loss_coeff")
                            if parquet_file.exists():
                                logger.info("Generating plots for env %s", env)
                                study_results = load_study_results(parquet_file)
                                if study_results is not None and not study_results.empty:
                                    outfiles = plot_importance_studies(
                                        study_results,
                                        output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                                        params=param_choices,
                                        env=env,
                                        title=args.title,
                                    )
                                    if outfiles and zipfile:
                                        for outfile in outfiles:
                                            if outfile.exists():
                                                zipfile.write(outfile, arcname=outfile.name)
                                                saved_files.append(outfile)
                        logger.info("Plotting complete for submission %s", submission_name)
                except KeyboardInterrupt:
                    logger.warning(
                        "KeyboardInterrupt received; canceling all tasks and aborting. Waiting 5s to finish writes"
                    )
                    for future in futures:
                        future.cancel()
                    for process_future in process_futures:
                        process_future.cancel()
                    import time

                    time.sleep(5)
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                        process_executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=False)
                        process_executor.shutdown(wait=False)
                    try:
                        if input("Exit now? (y/n): ").lower().startswith("y"):
                            sys.exit(1)
                    except KeyboardInterrupt:
                        sys.exit("1")
            logger.info("Exiting executor context (will shutdown executors) for submission %s...", submission_name)

        if args.update_db_only:
            logger.info("Update DB only specified; exiting after DB update.")
            sys.exit(0)

        if args.plot_only:
            is_sympol_paths = False
            if any("sympol" in p.lower() for p in args.paths):
                if not all("sympol" in p.lower() for p in args.paths):
                    raise ValueError("Either all or none of the paths must be sympol experiments.")
                is_sympol_paths = True
                zip_suffix += "sympol"
            all_studies = {}
            for env in envs:
                db_suffix = "_sympol" if is_sympol_paths else ""
                database_path = f"outputs/shared/optuna_hp_study_{env}{db_suffix}.db"

                studies = optuna_create_studies(
                    None,
                    database_path=database_path,
                    env=env,
                    dir_depth=0,
                    distributions=optuna_dists,
                    load_if_exists=True,
                    study_each_epoch=None,
                    submission_file_path=None,
                    clear_experiment_cache=args.clear_experiment_cache,
                    clear_study=args.clear_study,
                    excludes=file_excludes,
                    metric="episode_reward_mean",
                    load_studies_only="names_only",
                )
                all_studies.update(studies)

        # After first submission, do not clear experiment cache again
        if first_submission:
            first_submission = False
        logger.info("All Trials processed, plotting combined importance.")

        # Parallelize combined importance plotting across environments
        def _plot_combined_importance(
            env: str,
            all_studies_data: dict,
            param_list: list[str],
            submission_filter_val: bool | list[str],
            add_title: bool = False,
        ) -> tuple[str, list[Path] | None]:
            """Worker function to plot combined importance for one environment."""
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")  # Non-interactive backend for multiprocessing

            parquet_file = Path(
                f"outputs/shared/experiments/Default-mlp-{env}/hyperparameter_importance_combined.parquet"
            )
            importance_results = load_study_results(parquet_file)
            if importance_results is None:
                env_studies2 = {k: v for k, v in all_studies_data.items() if env in k}
                if not env_studies2:
                    logger.warning(
                        "No combined importance results found for env %s; skipping plotting combined importance.",
                        env,
                    )
                    return env, None
            else:
                env_studies2 = importance_results
            params_choices = param_list
            if env not in ("CartPole-v1", "Acrobot-v1", "LunarLander-v3", "Hopper-v5"):
                param_choices = param_list.copy()
                param_choices.remove("vf_loss_coeff")
            outfiles = plot_importance_studies(
                env_studies2,
                output_path=f"outputs/shared/experiments/Default-mlp-{env}",
                params=params_choices,
                env=env,
                submission_filter=submission_filter_val,
                title=add_title,
            )
            logger.info("Completed plotting combined importance for env %s", env)
            return env, outfiles

        # Determine submission filter value
        submission_filter_val: bool | list[str] = False

        # Run plotting in parallel using ProcessPoolExecutor
        plot_workers = max(1, min(len(envs), int((os.cpu_count() or 2) * 0.75)))
        plot_workers = len(envs)
        plot_workers = 1
        with ProcessPoolExecutor(max_workers=plot_workers) as plot_executor:
            plot_futures = {}
            for env in envs:
                # Skip if no data available and not in try_plot_only mode
                parquet_file = Path(
                    f"outputs/shared/experiments/Default-mlp-{env}/hyperparameter_importance_combined.parquet"
                )
                if not parquet_file.exists() and not args.try_plot_only:
                    logger.warning(
                        "No combined importance results found for env %s; skipping plotting combined importance. "
                        "Use try_plot_only to create analysis.",
                        env,
                    )
                    continue

                future = plot_executor.submit(
                    _plot_combined_importance,
                    env,
                    all_studies,
                    list(distributions.keys()),
                    submission_filter_val,
                    add_title=args.add_title,
                )
                plot_futures[future] = env

            # Collect results and add to zipfile
            for future in as_completed(plot_futures):
                env, outfiles = future.result()
                logger.info("Received plotting results for env %s", env)
                if outfiles and zipfile:
                    for outfile in outfiles:
                        if outfile.exists():
                            zipfile.write(outfile, arcname=outfile.name)
                            saved_files.append(outfile)

        logger.info("Completed all combined importance plotting")
        sys.exit()
    # ------------ ALL -------------------
    # Per-env databases; append _sympol if all paths indicate sympol experiments
    is_sympol_paths = False
    if any("sympol" in p.lower() for p in args.paths):
        if not all("sympol" in p.lower() for p in args.paths):
            raise ValueError("Either all or none of the paths must be sympol experiments.")
        is_sympol_paths = True
        zip_suffix += "sympol"
    cleared_cache_once = False
    if args.zip:
        zipfile = __create_zipfile(zip_suffix)
    for env in envs:
        paths = [p % env for p in args.paths]
        db_suffix = "_sympol" if is_sympol_paths else ""
        database_path = f"outputs/shared/optuna_hp_study_{env}{db_suffix}.db"
        # Decide if we clear cache for this env
        clear_now_for_env = False
        if not cleared_cache_once and clear_cache_present and (clear_cache_envs is None or env in clear_cache_envs):
            clear_now_for_env = True
        studies = optuna_create_studies(
            *paths,
            database_path=database_path,
            env=env,
            dir_depth=1,
            distributions=optuna_dists,
            load_if_exists=True,
            study_each_epoch=None,
            submission_file_path=Path("experiments"),
            clear_experiment_cache=clear_now_for_env,
            clear_study=args.clear_study,
            excludes=file_excludes,
            metric="episode_reward_mean",
        )
        if clear_now_for_env:
            cleared_cache_once = True
        param_choices = list(distributions.keys())
        if env not in ("CartPole-v1", "Acrobot-v1", "LunarLander-v3", "Hopper-v5"):
            param_choices = param_choices.copy()
            param_choices.remove("vf_loss_coeff")
        outfiles = plot_importance_studies(studies, output_path=paths[0], params=param_choices, env=env)
        if outfiles and zipfile:
            for outfile in outfiles:
                if outfile.exists():
                    zipfile.write(outfile, arcname=outfile.name)
                    saved_files.append(outfile)
        logger.info("All Trials processed for env %s. Saved files: %s", env, outfiles)
