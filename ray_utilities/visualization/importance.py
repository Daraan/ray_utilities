from __future__ import annotations

from copy import deepcopy
import logging
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.testing_utils import remote_breakpoint
from ray_utilities.misc import round_floats

try:  # pragma: no cover - compatibility shim for older NumPy
    from numpy import trapezoid as _np_trapezoid
except ImportError:  # pragma: no cover - fallback when numpy.trapezoid is unavailable
    from numpy import trapz as _np_trapezoid  # type: ignore[assignment]

import optuna.importance

from experiments.create_tune_parameters import write_distributions_to_json
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.visualization._common import Placeholder
from ray_utilities.visualization.data import (
    clean_placeholder_keys,
    combine_df,
    get_and_check_group_stat,
    get_running_experiments,
    ifill,
    load_run_data,
    save_run_data,
)
from ray_utilities.visualization.data import ifill as data_ifill

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
except ImportError:  # pragma: no cover - optional dependency
    RandomForestRegressor = None  # type: ignore[assignment]
    permutation_importance = None  # type: ignore[assignment]

ImportanceMethod = str
RegressorFactory = Callable[[], Any]
MetricKey = str | tuple[Any, ...]


logger = logging.getLogger(__name__)

_REPORT_INTERVAL = 32


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


def get_optuna_study(
    experiment_path: str | Path | optuna.storages.BaseStorage,
    env: str | None,
    step: int | None = None,
    *,
    load_if_exists=True,
):
    if isinstance(experiment_path, (Path, str)):
        storage = _get_storage(experiment_path)
    else:
        storage = experiment_path
    study_name = "pbt_study"
    if env is not None:
        study_name += f"_env={env}"
    if step is not None:
        study_name += f"_step={step}"
    else:
        study_name += f"_global"
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


def create_fake_trial(
    metric_result: float,
    identifier: str,
    params: dict[str, Any],
    intermediate_values: dict[int, float] | None = None,
    distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
):
    try:
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=metric_result,
            params=params,
            intermediate_values=intermediate_values,
            distributions=distributions,
            user_attrs={"identifier": identifier},
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
            user_attrs={"identifier": identifier},
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
) -> dict[Any, optuna.Study]:
    if disable_checks:
        disable_distribution_check()
    studies: dict[Any, optuna.Study] = {}
    storage = _get_storage(database_path or experiment_paths[0])
    tracking_study = get_experiment_track_study(storage)
    if not study_each_epoch:
        study = get_optuna_study(storage, env, load_if_exists=load_if_exists)
        studies[GLOBAL_STUDY] = study
    if study_each_epoch is not False:
        # Create a study for each epoch at step 8192 * 4 until max_step
        for epoch in range(1, 32 + 1):
            step = epoch * 8192 * 4
            studies[step] = get_optuna_study(storage, env, step)
        # This is not sufficient for the 1/8 perturbation intervals
        for epoch in range(1, 8 + 1):
            step = epoch * 147456
            if step not in studies:
                studies[step] = get_optuna_study(storage, env, step)
    for experiment_path in experiment_paths:
        logger.info("Checking path for experiments: %s", experiment_path)
        if dir_depth == 0:
            experiment_subdirs = [Path(experiment_path)]
        else:
            experiment_subdirs = Path(experiment_path).glob(f"*/{'/'.join(['*'] * dir_depth)}")
        running_experiments = set()
        if submission_file_path:
            running_experiments = get_running_experiments(submission_file_path)
        for experiment_dir in list(experiment_subdirs):
            no_errors = True
            experiment_id = experiment_dir.name.split("-")[-1]
            if experiment_id in running_experiments:
                logger.info("skipping running experiment: %s", experiment_dir.name)
                continue
            if experiment_id in STORED_EXPERIMENTS:
                logger.info("Skipping run %s as the id is already stored", experiment_id)
                continue

            logger.info("Checking experiment path: %s", experiment_dir)
            # Check if this directory is an experiment directory
            marker_files = {".validate_storage_marker", "pbt_global.txt", "tuner.pkl"}
            dir_contents = [p.name for p in experiment_dir.glob("*")]
            if not any(marker_file in dir_contents for marker_file in marker_files):
                logger.warning("Directory %s does not appear to be a valid experiment directory.", experiment_dir)
                continue
            if env and env not in experiment_dir.name:
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
                        group_numeric = group.select_dtypes(include=["number"])[
                            (group.current_step == group.current_step.max()).values  # noqa: PD011
                        ]
                        final_row = group_numeric.mean()
                    else:
                        run_id = group_key
                        trial_identifier = run_id
                        final_row = group.iloc[-1]
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
                    params = round_floats(params)
                    # Clean placeholders
                    if metric not in final_row and metric == "episode_reward_mean":
                        from ray_utilities.constants import (  # noqa: PLC0415
                            EPISODE_RETURN_MEAN,
                            EPISODE_RETURN_MEAN_EMA,
                        )

                        # introduced it later
                        if (
                            EPISODE_RETURN_MEAN_EMA in df.evaluation
                            and not pd.isna(final_row[("evaluation", EPISODE_RETURN_MEAN_EMA)]).all()
                        ):
                            backport_metric = ("evaluation", EPISODE_RETURN_MEAN_EMA)
                        else:
                            backport_metric = ("evaluation", EPISODE_RETURN_MEAN)
                        logger.info(
                            "Metric 'episode_reward_mean' was not found in the data, changing metric to: %s",
                            backport_metric,
                        )
                        metric = backport_metric
                    elif metric not in final_row:
                        raise KeyError(f"Metric column '{metric}' not found in combined dataframe")  # noqa: TRY301
                    try:
                        metric_result = float(final_row[metric].iloc[0])
                    except Exception:
                        metric_result = float(final_row[metric])

                    # Create intermediate values from the training history
                    if False:
                        # should not be relevant
                        intermediate_values = {}
                        # if len(df) > 1 and metric_columns:
                        #    for idx, row in df.iterrows():
                        #        step = int(row.get("training_iteration", idx[1] if isinstance(idx, tuple) else idx))
                        #        intermediate_values[step] = float(row[metric_columns[0]])

                    # Create and add trial to study
                    trial = create_fake_trial(
                        metric_result=metric_result,
                        identifier=trial_identifier,
                        params={k: v for k, v in params.items() if k in distributions} if distributions else params,
                        distributions=distributions,
                        # intermediate_values=intermediate_values
                    )
                    if not per_pbt_epoch:
                        trials_to_add[studies[GLOBAL_STUDY]].append(trial)
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
                                upper_bound = closest_key + final_row.config.get("train_batch_size_per_learner", 2048)
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
                                    logger.warning("Could not determine epoch of %s at step %s", run_id, current_step)
                                    # is in the lower half between two keys - ceil up
                                    try:
                                        study_to_add = studies[closest_key + 8192 * 4]
                                    except KeyError:
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
                                        logger.warning(
                                            "Could not determine pbt epoch / current_step of trial %s with at step %s. "
                                            "Added it to epoch of step %s.",
                                            run_id,
                                            current_step,
                                            closest_key + 8192 * 4,
                                        )
                        if study_to_add is not None:
                            trials_to_add[study_to_add].append(trial)

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
                remote_breakpoint()
                continue
            logger.info("Adding trials from experiment %s to studies", experiment_dir)
            for study, trials in trials_to_add.items():
                maybe_add_trials_to_study(trials, study)
            if no_errors and study_each_epoch is None:  # add it as finished when we checked both types.
                add_finished_experiment(experiment_id, tracking_study)

    return studies


PARAMS_TO_CHECK = {
    "batch_size",
    "lr",
    "gamma",
}


def _fix_distributions(study: optuna.Study, params: Collection[str]):
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
                choices.update(dist.choices)
            elif isinstance(dist, optuna.distributions.FloatDistribution):
                choices.update([dist.low, dist.high])
            elif isinstance(dist, optuna.distributions.IntDistribution):
                choices.update(range(dist.low, dist.high + 1))
            else:
                logger.warning("Cannot merge distribution of type %s for parameter %s", type(dist), name)
        if all(isinstance(c, (int, float)) for c in choices):
            if all(isinstance(c, int) for c in choices):
                new_dist = optuna.distributions.IntDistribution(low=min(choices), high=max(choices))
            else:
                new_dist = optuna.distributions.FloatDistribution(low=min(choices), high=max(choices))
        else:
            if None in choices:
                choices.discard(None)
                choices = (*sorted(choices), None)
                new_dist = optuna.distributions.CategoricalDistribution(choices=choices)
            else:
                new_dist = optuna.distributions.CategoricalDistribution(choices=tuple(sorted(choices)))
        logger.debug("Setting merged distribution for parameter %s: %s", name, new_dist)
        # Now set it for all trials
        for trial in completed_trials:
            trial.distributions[name] = new_dist
    return completed_trials


def optuna_analyze_studies(
    studies: dict[Any, optuna.Study],
    output_path: str | Path,
    params: list[str] | None = None,
) -> None:
    from optuna.importance import (
        MeanDecreaseImpurityImportanceEvaluator,
        PedAnovaImportanceEvaluator,
        FanovaImportanceEvaluator,
    )

    evaluators = {
        "MeanDecreaseImpurity": MeanDecreaseImpurityImportanceEvaluator(),
        "PedAnovaLocal": PedAnovaImportanceEvaluator(evaluate_on_local=True),
        "Fanova(default)": FanovaImportanceEvaluator(),  # default
        "MeanDecreaseImpurity32": MeanDecreaseImpurityImportanceEvaluator(n_trees=32, max_depth=32),
        "PedAnovaGlobal": PedAnovaImportanceEvaluator(evaluate_on_local=False),
        "Fanova32": FanovaImportanceEvaluator(n_trees=32, max_depth=32),
    }

    for key, study in studies.items():
        logger.info("Analyzing study for key: %s", key)
        completed_trials = None
        try:
            for evaluator_name, evaluator in evaluators.items():
                try:
                    importances = optuna.importance.get_param_importances(study, evaluator=evaluator, params=params)
                except ValueError as ve:
                    if "dynamic search" in str(ve):
                        study = deepcopy(study)  # noqa: PLW2901
                        completed_trials = _fix_distributions(study, params or PARAMS_TO_CHECK)
                        study.get_trials = lambda *args, **kwargs: completed_trials  # type: ignore[assignment]
                        importances = optuna.importance.get_param_importances(study, evaluator=evaluator, params=params)
                    else:
                        raise
                else:
                    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
                # logger.info("Top %d important hyperparameters for study %s:", top_n, key)
                try:
                    save_path = Path(output_path)
                    if save_path.is_dir():
                        save_path = save_path / f"hyperparameter_importance-{study.study_name}_{evaluator_name}.csv"
                    else:
                        raise ValueError("output_path must be a directory to save importances")
                    importances["_number_of_trials"] = len(completed_trials)
                    importances["_evaluator"] = evaluator_name  # pyright: ignore[reportArgumentType]
                    pd.DataFrame.from_dict(importances, orient="index").to_csv(save_path, index=True)
                    logger.info("Saved importances to %s", save_path)
                except Exception:
                    logger.exception("Could likely not export importances to %s", output_path)
                importances.pop("_evaluator", None)
                sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
                report_str = f"Hyperparameter importances for study {study.study_name} ({evaluator_name}) with {len(completed_trials)} trials:\n"
                for param, importance in sorted_importances:
                    report_str += f"  {param}: {importance:.5f}\n"
                logger.info("%s", report_str)
        except Exception as e:
            logger.exception("Failed to analyze study for key %s: %r", key, e)


_check_distribution_compatibility = optuna.distributions.check_distribution_compatibility


def disable_distribution_check():
    # TODO: Alternatively move all trials to a new study with the changed distribution
    optuna.distributions.check_distribution_compatibility = lambda a, b: None


if __name__ == "__main__":
    from experiments.create_tune_parameters import default_distributions
    from ray_utilities import nice_logger

    os.chdir(Path(__file__).parent.parent.parent)
    # possibly for repair sqlite3 "outputs/shared/optuna_hp_study.db" ".dump" | sqlite3 new.db

    logger = nice_logger(logger, logging.DEBUG)
    distributions = load_distributions_from_json(write_distributions_to_json(default_distributions), as_optuna=True)
    distributions.pop("test", None)
    distributions.pop("minibatch_scale", None)
    if "vf_clip_param" in distributions and 0.0 not in distributions["vf_clip_param"].choices:
        # We changed this distribution
        distributions["vf_clip_param"].choices = (0.0, *distributions["vf_clip_param"].choices)
    if "DEBUG" in os.environ:
        paths = ("outputs/shared/experiments/Default-mlp-Acrobot-v1",)
        env = "Acrobot-v1"
    else:
        paths = [
            "outputs/shared/experiments/Default-mlp-Ant-v5",
            "outputs/shared_backup/needs_sync/Default-mlp-Ant-v5",
            "outputs/shared_backup/Default-mlp-Ant-v5",
        ]
        env = "Ant-v5"
    studies = optuna_create_studies(
        *paths,
        database_path="outputs/shared/optuna_hp_study.db",
        env=env,
        dir_depth=1,
        distributions=distributions,
        load_if_exists=True,
        study_each_epoch=None,
        submission_file_path=Path("experiments"),
    )
    optuna_analyze_studies(studies, output_path=paths[0], params=list(distributions.keys()))
    if "DEBUG" in os.environ:
        import sys

    sys.exit(0)
    for env in ["CartPole-v1", "Walker2d-v5", "LunarLander-v3"]:
        paths = [
            f"outputs/shared/experiments/Default-mlp-{env}",
            f"outputs/shared_backup/needs_sync/Default-mlp-{env}",
            f"outputs/shared_backup/Default-mlp-{env}",
        ]
        studies = optuna_create_studies(
            *paths,
            database_path="outputs/shared/optuna_hp_study.db",
            env=env,
            dir_depth=1,
            distributions=distributions,
            load_if_exists=True,
            study_each_epoch=None,
            submission_file_path=Path("experiments"),
        )
