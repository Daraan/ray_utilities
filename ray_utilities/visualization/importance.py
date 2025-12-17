from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser

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
        run_frames = load_run_data(path)
        if not run_frames:
            logger.warning("No runs found for experiment at %s", path)
            continue
        combined_df = combine_df(run_frames)
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


def get_optuna_study(experiment_path, env: str | None, step: int | None = None, *, load_if_exists=True):
    db_path = f"{experiment_path}/optuna_study.db"
    storage_str = f"sqlite:///{db_path}"
    study_name = "pbt_study"
    if env is not None:
        study_name += f"_env={env}"
    if step is not None:
        study_name += f"_step={step}"
    storage = optuna.storages.RDBStorage(
        url=storage_str,
        engine_kwargs={"connect_args": {"timeout": 10}},
    )
    if Path(db_path).exists() and not load_if_exists:
        os.remove(db_path)
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception as e:
            logger.warning("Failed to delete existing study: %s", e)
    return optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=load_if_exists, direction="maximize"
    )


def create_fake_trial(
    metric_result,
    params: dict[str, Any],
    intermediate_values: dict[int, float] | None = None,
    distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
):
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=metric_result,
        params=params,
        intermediate_values=intermediate_values,
        distributions=distributions,
    )
    return trial


GLOBAL_STUDY = "global"


def optuna_create_studies(
    *experiment_paths,
    env: str | None,
    study_each_epoch: bool = False,
    dir_depth=1,
    metric="episode_reward_mean",
    group_stat: str | None = None,
    distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
    load_if_exists: bool = True,
) -> dict[Any, optuna.Study]:
    studies: dict[Any, optuna.Study] = {}
    if not study_each_epoch:
        study = get_optuna_study(experiment_paths[0], env, load_if_exists=load_if_exists)
        studies[GLOBAL_STUDY] = study
    else:
        # Create a study for each epoch at step 8192 * 4 until max_step
        for epoch in range(32):
            step = epoch * 8192 * 4
            studies[step] = get_optuna_study(experiment_paths[0], env, step)

    for experiment_path in experiment_paths:
        logger.info("Checking experiment path: %s", experiment_path)
        if dir_depth == 0:
            experiment_subdirs = [Path(experiment_path)]
        else:
            experiment_subdirs = Path(experiment_path).glob(f"*/{'/'.join(['*'] * dir_depth)}")
        for experiment_dir in experiment_subdirs:
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
                group_stat, df = get_and_check_group_stat(df, group_stat, ("pbt_group_key",))

                # Create trials for each run_id
                # TODO: possibly groupkey -> take mean
                for group_key, group in df.groupby("run_id" if not study_each_epoch else (("pbt_epoch", "run_id"))):
                    if group.empty:
                        continue
                    if isinstance(group_key, tuple):
                        run_id = group_key[1]
                    else:
                        run_id = group_key
                    # Extract final metric value (last row)
                    final_row = group.iloc[-1]

                    # Extract hyperparameters from config
                    params = clean_placeholder_keys(final_row.to_dict(), flatten=True)
                    if distributions:
                        for key in distributions.keys():
                            if key not in params:
                                params[key] = final_row.config.get(
                                    key, final_row.config.get("cli_args", {}).get(key, "NOT_FOUND")
                                )
                                if isinstance(params[key], (pd.DataFrame, pd.Series)) and params[key].size == 1:
                                    params[key] = params[key].item()
                                if params[key] == "NOT_FOUND":
                                    params[key] = getattr(DefaultArgumentParser, key)
                        params = {k: v for k, v in params.items() if not isinstance(v, str) or v != "NOT_FOUND"}
                    # Clean placeholders
                    if metric not in final_row and metric == "episode_reward_mean":
                        from ray_utilities.constants import (  # noqa: PLC0415
                            EPISODE_RETURN_MEAN,
                            EPISODE_RETURN_MEAN_EMA,
                        )

                        # introduced it later
                        if EPISODE_RETURN_MEAN_EMA in df.evaluation:
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
                        params={k: v for k, v in params.items() if k in distributions} if distributions else params,
                        distributions=distributions,
                        # intermediate_values=intermediate_values
                    )
                    if GLOBAL_STUDY in studies:
                        studies[GLOBAL_STUDY].add_trial(trial)
                    else:
                        # TODO: will not work if trial ended too early
                        studies[int(final_row.current_step)].add_trial(trial)
                    logger.debug("Added trial for run %s with metric %s", run_id, metric_result)

            except Exception as e:
                logger.exception("Failed to process experiment at %s: %r", experiment_dir, e)
                continue
    return studies


PARAMS_TO_CHECK = {
    "batch_size",
    "lr",
    "gamma",
}


def optuna_analyze_studies(
    studies: dict[Any, optuna.Study],
    top_n: int | None = None,
    params: list[str] | None = None,
) -> None:
    for key, study in studies.items():
        logger.info("Analyzing study for key: %s", key)
        try:
            importances = optuna.importance.get_param_importances(study, params=params)
            sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
            # logger.info("Top %d important hyperparameters for study %s:", top_n, key)
            for param, importance in sorted_importances[:top_n]:
                logger.info("  %s: %.4f", param, importance)
        except Exception as e:
            logger.exception("Failed to analyze study for key %s: %r", key, e)


if __name__ == "__main__":
    logging.debug("Starting optuna analysis")
    from experiments.create_tune_parameters import default_distributions
    from ray_utilities import nice_logger

    logger = nice_logger(logger, logging.DEBUG)
    distributions = load_distributions_from_json(write_distributions_to_json(default_distributions), as_optuna=True)
    distributions.pop("test", None)
    studies = optuna_create_studies(
        "outputs/shared/experiments/Default-mlp-Acrobot-v1",
        env="Acrobot-v1",
        dir_depth=2,
        distributions=distributions,
        load_if_exists=True,
    )
    optuna_analyze_studies(studies, params=list(distributions.keys()))
