from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

PATHS = [Path("outputs/experiments/shared"), Path("outputs/experiments/shared_backup")]

DROP_COLUMNS = [
    "hostname",
    "done",
    ("config", "cli_args", "evaluate_every_n_steps_before_step"),
    ("config", "cli_args", "offline_loggers"),
    ("config", "cli_args", "fcnet_hiddens"),
    ("config", "cli_args", "perturbation_factors"),
    ("config", "cli_args", "comment"),
    ("config", "cli_args", "head_fcnet_hiddens"),
    ("config", "_config_files"),
    # Maybe want to keep but not hashable
    ("config", "fork_from", "parent_time"),
]

# Experiment output directory structure is - if sorted in groups else one level higher

# Project/group/Name*run_id/

logger = logging.getLogger(__name__)


def load_run_data(offline_run: str | Path, experiment_dir=None):
    if isinstance(offline_run, str):
        if "/" in offline_run:
            # Assume path
            offline_run = Path(offline_run)
        else:
            # find it
            base_paths = [Path(experiment_dir)] if experiment_dir else PATHS
            for base_path in base_paths:
                # search in project/group/*run_id and project/*run_id then go to next base path
                search_with_group = list(base_path.glob(f"*/*/*{offline_run}"))
                if search_with_group:
                    run_path = search_with_group
                    break
                search_without_group = list(base_path.glob(f"*/*{offline_run}"))
                if search_without_group:
                    run_path = search_without_group
                    break
            else:
                raise FileNotFoundError(f"Run ID {offline_run} not found in any of the base paths.")
            if len(run_path) > 1:
                raise ValueError(f"Multiple runs found for ID {offline_run}: {run_path}")
            offline_run = run_path[0]
    if not offline_run.exists():
        raise FileNotFoundError(f"Run path {offline_run} does not exist.")
    # Load data
    result_files = offline_run.glob("*/result*.json")
    run_data = {}
    for result_file in tqdm(result_files):
        df = pd.read_json(result_file, lines=True)
        # Drop unwanted columns if they exist
        df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
        # Flatten nested dict columns and convert to MultiIndex
        # ptimes = df[("config", "fork_from", "parent_time")]
        # mask = ~ptimes.isna()

        # df.loc[mask.values, ("config", "fork_from", "parent_time")] = ptimes[mask.values].map(str).values
        df = pd.json_normalize(df.to_dict(orient="records"), sep="/")
        df.columns = pd.MultiIndex.from_tuples([tuple(col.split("/")) for col in df.columns])
        # df.columns = pd.MultiIndex.from_tuples(cast("list[tuple[str, ...]]", df.columns))
        experiment_key = df.config.experiment_key.iloc[-1].item()
        assert result_file.name == "result.json" or experiment_key in result_file.name, (
            f"Experiment key {experiment_key} does not match result file name {result_file.name}"
        )
        run_data[experiment_key] = df
    logger.info(f"Loaded data for run {offline_run} with {len(run_data.keys())} experiments")  # noqa: G004
    return run_data


def combine_df(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple DataFrames into a single DataFrame by concatenation.

    Args:
        dataframes: Dictionary of DataFrames to combine.

    Returns:
        A single DataFrame resulting from concatenation of the input DataFrames.
    """

    # Remove 'run_id' columns at any level if they exist to avoid conflicts
    def drop_run_id_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            mask = [not any(lvl == "run_id" for lvl in col) for col in df.columns]
            return df.loc[:, mask]
        else:
            return df.drop(columns=["run_id"], errors="ignore")

    dfs: list[pd.DataFrame] = []
    for df in dataframes.values():
        df_clean = drop_run_id_columns(df)
        # Find the correct column name for 'training_iteration' in MultiIndex or single index
        if isinstance(df_clean.columns, pd.MultiIndex):
            idx_name = next((col for col in df_clean.columns if col[0] == "training_iteration"), None)
            if idx_name is None:
                raise KeyError("No 'training_iteration' column found in DataFrame columns.")
        else:
            idx_name = "training_iteration"
        df_clean = df_clean.set_index(idx_name, drop=True)
        dfs.append(df_clean)

    combined_df = pd.concat(dfs, keys=dataframes.keys(), names=["run_id", "training_iteration"])
    try:
        combined_df = combined_df.drop_duplicates(keep="first")
    except TypeError:
        configs = combined_df.pop("config")  # not hashable
        duplicates = combined_df.index.duplicated(keep="first")
        combined_df = combined_df[~duplicates]
        configs = configs[~duplicates]
        # Merge again
        # Add a MultiIndex level "config" to all columns in configs
        if not isinstance(configs.columns, pd.MultiIndex):
            configs.columns = pd.MultiIndex.from_product([["config"], configs.columns])
        else:
            configs.columns = pd.MultiIndex.from_tuples(
                [("config", *col) if isinstance(col, tuple) else ("config", col) for col in configs.columns]
            )
        combined_df = pd.concat([combined_df, configs], axis=1)

    # Reset only the outermost 'run_id' index, not all levels
    # combined_df = combined_df.reset_index(level=0)
    # Ensure 'run_id' is the first index level, and avoid duplicate index names
    # index_names = [n for n in combined_df.index.names if n != "run_id"]
    # combined_df = combined_df.set_index(["run_id", "training_iteration"], drop=True)
    # combined_df = combined_df.reorder_levels(["run_id", *index_names])
    return combined_df


TEST = True


def plot_run_data(
    df: pd.DataFrame,
    metrics: list[str],
    experiment_keys: list[str] | None = None,
    smoothing: int = 1,
    figsize: tuple[int, int] = (12, 8),
    group_by=("pbt_epoch", "pbt_group_key"),
    *,
    plot_std: bool = False,
) -> None:
    """Plot specified metrics from the run data.

    Args:
        df: Combined DataFrame with MultiIndex columns and index.
        metrics: List of metric names to plot.
        experiment_keys: Optional list of experiment keys to include in the plot.
            If None, all experiments in df are used.
        smoothing: Smoothing window size for rolling mean. Default is 1 (no smoothing).
        figsize: Size of the figure to create.
        group_by: Tuple of column names (as strings, not tuples) to group by under 'config'.
    """
    if experiment_keys is None:
        experiment_keys = df.index.get_level_values(0).unique().to_list()

    # Select the group-by columns from MultiIndex columns
    group_cols = [("config", k) for k in group_by]
    # Extract the group-by columns as a DataFrame (each column is 1D)
    group_values: list[pd.Series] = [
        df.current_step.droplevel(0, axis=1),
        *(df[col] for col in group_cols),
        df.config.seed,
    ]
    # Combine into a DataFrame for groupby
    group_df = pd.concat(group_values, axis=1, keys=["current_step", *group_by, "seed"], copy=False)
    # Problem we get duplicated values as the dfs contain their parents data - need to drop these when we aggregate
    group_df = group_df[~group_df.duplicated(keep="first")]
    group_df.columns = ["current_step", *group_by, "seed"]  # flatten for groupby
    group_df = group_df.drop("seed", axis=1)

    # Example usage: grouped = df.groupby([group_df[k] for k in group_keys])
    grouped = df.groupby([group_df[k] for k in group_df.columns])
    # If you want to use group_df for further processing, do so here.

    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_group = grouped[metric]
        stats = metric_group.describe()
        # Drop all levels from the columns except the last
        # stats.columns = [col[-1] if isinstance(col, tuple) else col for col in stats.columns]
        for exp_key in experiment_keys if not TEST else experiment_keys[:15]:
            # df_exp = df.loc[exp_key]
            # if metric not in df_exp.columns.get_level_values(0):
            #    logger.warning("Metric %s not found in experiment %s. Skipping.", metric, exp_key)  # noqa: G004
            #    continue
            # metric_series = df_exp[metric].droplevel(0, axis=1)
            # if smoothing > 1:
            #    metric_series = metric_series.rolling(window=smoothing, min_periods=1).mean()
            # Plot mean, std, min, max from stats
            # stats is a DataFrame with multi-index (group keys, then stat rows)
            # We need to select the rows for this exp_key and plot mean, std, min, max

            # stats.loc[...] shape: (stat, group_keys, metric)
            # For each group, plot mean, std, min, max as lines
            # Here, stats is grouped by group_keys, so we need to select the correct group
            # We'll plot mean, std, min, max for each group in this experiment
            # stats.index: MultiIndex (group_keys, stat)
            # stats.columns: metric
            # We'll plot mean, std, min, max as separate lines
            stat_names = ["mean", "std"]
            for stat in stat_names:
                stat_df: pd.DataFrame = stats.xs(stat, level=-1, axis=1).reset_index()
                stat_df.columns = [*(n[0] for n in stats.index.names), stat]
                if stat == "mean":
                    # Plot mean line
                    sns.lineplot(data=stat_df, x="current_step", y=stat, ax=ax, label=f"{exp_key} mean", linestyle="-")
                    # Fill between mean ± std if std is available
                    if "std" in stats.columns.get_level_values(-1):
                        std_df: pd.DataFrame = stats.xs("std", level=-1, axis=1).reset_index()
                        std_df.columns = [*(n[0] for n in stats.index.names), "std"]
                        ax.fill_between(
                            stat_df["current_step"],
                            stat_df["mean"] - std_df["std"],
                            stat_df["mean"] + std_df["std"],
                            alpha=0.2,
                            label=f"{exp_key} mean±std",
                        )
                else:
                    sns.lineplot(
                        data=stat_df, x="current_step", y=stat, ax=ax, label=f"{exp_key} {stat}", linestyle="--"
                    )
        ax.set_title(f"Metric: {metric}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric)
        # ax.legend()
    plt.tight_layout()
    plt.show()


# Example for grouping by MultiIndex columns:
# group_cols = [("config", "pbt_epoch"), ("config", "pbt_group_key")]
# group_values = [df[col] for col in group_cols]
# group_df = pd.concat(group_values, axis=1)
# group_df.columns = ["pbt_epoch", "pbt_group_key"]
# grouped = df.groupby([group_df["pbt_epoch"], group_df["pbt_group_key"]])
