# PLOTS TODOS
# - Some logscale have not enough xticks
# - Fontsize
# Could make huge minibatch size runs with batch_size = 16384
# or just do gradient accumulation

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Hashable, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colormaps, patheffects
from typing_extensions import Final, Literal

from ray_utilities.testing_utils import remote_breakpoint
from ray_utilities.visualization._common import Placeholder, PlotOption
from ray_utilities.visualization.data import (
    DEFAULT_GROUP_BY,
    LOG_SETTINGS,
    MAX_ENV_LOWER_BOUND,
    ENV_BOUNDS,
    _connect_groups,
    _drop_duplicate_steps,
    _try_cast,
    get_and_check_group_stat,
    logger,
    which_continued,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# mpl.rcParams["savefig.facecolor"] = "#f7f7f7"

logger = logging.getLogger(__name__)

nan: Final = np.nan


def make_cmap(
    values,
    name: str = "viridis",
    *,
    log: bool = False,
):
    # Create a continuous colormap for the group_stat (logarithmic scale)
    if log:
        norm = mcolors.LogNorm(
            vmin=values.replace(0, nan).replace(float("-inf"), nan).min(),
            vmax=values.max(),
        )
    else:
        norm = mcolors.Normalize(
            vmin=values.min(),
            vmax=values.max(),
        )
    cmap = colormaps.get_cmap(name)

    # Map group_stat values to colors
    unique_stats = values.unique()
    color_map = {val: mcolors.to_hex(cmap(norm(val))) for val in unique_stats if not log or val > 0}
    # For zero or negative values, fallback to a default color
    for val in unique_stats:
        if (log and val <= 0) or pd.isna(val) or val in (float("inf"), float("-inf"), None):
            color_map[val] = "#cccccc"
    # add masked values color, included nan, -inf
    return color_map, cmap, norm


def plot_error_distribution(
    plot_df: pd.DataFrame,
    metric_key: str,
    group_stat: str,
    *,
    plot_errors_type: Literal["box", "violin"],
    palette: Mapping[Hashable, str] | None = None,
    epoch_col: str = "pbt_epoch",
    plot_option: PlotOption,
) -> Figure | None:
    """Create a separate figure showing the distribution of *metric_key* per *group_stat*."""
    if epoch_col not in plot_df:
        logger.warning("Epoch column %s missing from plot_df, skipping error plot.", epoch_col)
        return None
    data = plot_df[[metric_key, group_stat, epoch_col]].copy()
    data["_training_iteration"] = plot_df.index.get_level_values("training_iteration")
    if data.empty:
        logger.debug("No data available for error plot (%s)", metric_key)
        return None
    gp = data.groupby([group_stat, epoch_col])
    start = gp["_training_iteration"].transform("min")
    end = gp["_training_iteration"].transform("max")
    threshold = start + (end - start) / 2
    data = data[data["_training_iteration"] >= threshold]
    if data.empty:
        logger.debug("No data in second half of epochs for (%s)", metric_key)
        return None
    data = data.drop(columns="_training_iteration")

    # Detect extreme outliers using IQR method
    metric_values = data[metric_key].dropna()
    if len(metric_values) > 0:
        q1 = metric_values.quantile(0.25)
        q3 = metric_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Check if we have extreme outliers (values beyond 3*IQR)
        extreme_outliers = (metric_values < lower_bound) | (metric_values > upper_bound)
        has_extreme_outliers = extreme_outliers.any()
    else:
        has_extreme_outliers = False

    fig_err, ax_err = plt.subplots(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))

    # Use brokenaxes if we have extreme outliers
    if has_extreme_outliers:
        try:
            from brokenaxes import brokenaxes

            # Determine the break points
            normal_max = q3 + 1.5 * iqr
            extreme_min = (
                metric_values[metric_values > upper_bound].min() if (metric_values > upper_bound).any() else None
            )
            extreme_max = metric_values.max()

            # Create brokenaxes with a gap
            fig_err = plt.figure(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))
            if extreme_min is not None and extreme_max > normal_max * 1.2:
                # Gap between normal range and outliers
                gap_size = 0.05
                ax_err = brokenaxes(
                    ylims=((metric_values.min() * 0.95, normal_max * 1.05), (extreme_min * 0.95, extreme_max * 1.05)),
                    hspace=0.3,
                    fig=fig_err,
                )
            else:
                # Fall back to normal plot if break is too small
                fig_err, ax_err = plt.subplots(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))
                has_extreme_outliers = False
        except ImportError:
            logger.debug("brokenaxes not installed, using standard plot for outlier data")
            fig_err, ax_err = plt.subplots(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))
            has_extreme_outliers = False
        except Exception as e:
            logger.warning("Failed to create broken axes: %r, falling back to standard plot", e)
            fig_err, ax_err = plt.subplots(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))
            has_extreme_outliers = False

    fig_err, ax_err = plt.subplots(figsize=(max(len(data[group_stat].unique()) * 0.4 + 5, 8), 6))
    plot_fn = sns.boxplot if plot_errors_type == "box" else sns.violinplot
    plot_kwargs = {
        "data": data,
        "x": group_stat,
        "hue": group_stat,
        "legend": False,
        "y": metric_key,
        "ax": ax_err,
        "linewidth": 1,
    }
    if palette:
        palette_colors = {stat: palette.get(stat, "#4c72b0") for stat in pd.unique(data[group_stat])}
        plot_kwargs["palette"] = palette_colors
    else:
        plot_kwargs["color"] = "#4c72b0"
    plot_fn(**plot_kwargs)
    if plot_option.title:
        ax_err.set_title(f"{metric_key} distribution by {group_stat}")
    ax_err.set_xlabel(" ".join(group_stat.split("_")))
    ax_err.set_ylabel(" ".join(metric_key.split("_")))
    ax_err.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig_err


def _plot_metrics_of_group(
    group: pd.DataFrame, metric_key: str, ax, color, background_colors, *, label, group_by_keys_only
):
    if not group_by_keys_only:
        # default case
        sns.lineplot(
            data=group,
            x="current_step",
            y=metric_key,
            ax=ax,
            # linestyle="--",
            color=color,
            label=label,
            linewidth=3,
        )
    else:
        # There will be jumps at the end of each epoch, split the data and plot separately
        # to avoid lines going across the plot
        last_epoch_group = None
        epoch_grouper = group.groupby("pbt_epoch", sort=True)
        bg_color_idx = 0
        for _, epoch_group in epoch_grouper:
            # Shade background:
            if last_epoch_group is not None:
                _shade_background(
                    ax,
                    background_colors[bg_color_idx % 2],
                    last_epoch_group["current_step"].max(),
                    epoch_group["current_step"].max(),
                )
            last_epoch_group = epoch_group
            bg_color_idx += 1
            # Plot data
            sns.lineplot(
                data=epoch_group,
                x="current_step",
                y=metric_key,
                ax=ax,
                # linestyle="--",
                color=color,
                label=label,
                linewidth=3,
            )


# Decorator to temporarily set the seaborn plotting context for a function
from typing import Callable
import seaborn as sns
from functools import wraps


def with_seaborn_context(
    context: Literal["paper", "notebook", "talk", "poster"] = "talk", font_scale: float = 1.2
) -> Callable:
    """
    Decorator to temporarily set the seaborn plotting context for a function.

    Args:
        context: The seaborn context to use (e.g., "paper", "notebook", "talk", "poster").
        font_scale: Scaling factor for fonts.

    Returns:
        The decorated function, which will run with the specified seaborn context.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with sns.plotting_context(context, font_scale=font_scale):
                return func(*args, **kwargs, sns_context=context, font_scale=font_scale)

        return wrapper

    return decorator


def apply_context_fonts(fig, ctx):
    for ax in fig.axes:
        # --- Axes title ---
        ax.title.set_fontsize(ctx["axes.titlesize"])

        # --- Axis labels ---
        ax.xaxis.label.set_fontsize(ctx["axes.labelsize"])
        ax.yaxis.label.set_fontsize(ctx["axes.labelsize"])

        # --- Tick labels ---
        for t in ax.get_xticklabels():
            t.set_fontsize(ctx["xtick.labelsize"])
        for t in ax.get_yticklabels():
            t.set_fontsize(ctx["ytick.labelsize"])

        # --- Legend ---
        legend = ax.get_legend()
        if legend:
            for t in legend.get_texts():
                t.set_fontsize(ctx["legend.fontsize"])
            if legend.get_title():
                legend.get_title().set_fontsize(ctx.get("legend.title_fontsize", ctx["legend.fontsize"]))


@with_seaborn_context(font_scale=1.6)
def plot_run_data(
    df: pd.DataFrame,
    metrics: Sequence[str | tuple[str, ...]],
    experiment_keys: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    group_stat: str | None = None,
    group_by: str | Sequence[str] = DEFAULT_GROUP_BY,
    *,
    pbt_metric: str | tuple[str, ...] | None = None,
    log: bool | None = None,
    plot_option: PlotOption = PlotOption(),  # noqa: B008
    show: bool = False,
    pbt_plot_interval: int | bool = True,
    plot_errors: bool | Literal["only"] | str | Sequence[str] = True,
    plot_errors_type: Literal["box", "violin"] = "box",
    fig: Figure | None = None,
    # auto parameter by decorator, only use for unpickling
    sns_context=None,
    font_scale=None,
) -> tuple[Figure, dict[str, Figure]]:
    """Plot specified metrics from the run data.

    Args:
        df: Combined DataFrame with MultiIndex columns and index.
        metrics: List of metric names to plot.
        pbt_metric: The selection metric that was used during PBT. This metric
            will be used to determine the best runs for highlighting and ordering.
        experiment_keys: Optional list of experiment keys to include in the plot.
            If None, all experiments in df are used.
        figsize: Size of the figure to create.
        group_stat: Refers to the column that relates to the grouping statistic, e.g. pbt_group_key.
            By default the last key of group_by is used.
        group_by: Tuple of column names (as strings, not tuples) to group by under 'config'.
        log: Whether to use logarithmic scale for the group_stat color mapping. If None, checks LOG_SETTINGS.
        show: Whether to display the plot immediately.
        plot_option:
            main_only: Whether to plot only the main branch runs.
            plot_reduced: When True, to reduce clutter will plot curves of the best,
                second best, and worst of each group_stat,
                the old value +/- 1 level and the new value +/- 1 level and the second best.
        plot_errors: Plot a second graphic with a boxplot or violin plot of the error bars for each group.
    """  # noqa: W291
    if not isinstance(plot_errors, bool) and plot_errors != "only":
        plot_errors = plot_errors == group_by and plot_errors
        if not plot_errors:
            logger.info("Will not plot error bars as plot_errors=%r does not match group_by=%r", plot_errors, group_by)

    depth = len(df.columns[0])

    def ifill(*cols, n=depth):
        return (*cols, *(Placeholder,) * (n - len(cols)))

    if experiment_keys is None:
        experiment_keys = df.index.get_level_values(0).unique().to_list()
        if "DEBUG" in os.environ:
            experiment_keys = experiment_keys[:5]
    group_stat, df = get_and_check_group_stat(df, group_stat, group_by)
    final_metric_was_none = pbt_metric is None
    if log is True:
        log_base = LOG_SETTINGS[group_stat]
    elif log is False:
        log_base = None
    else:
        log = bool(LOG_SETTINGS.get(group_stat, False))
        log_base = LOG_SETTINGS[group_stat] if log else None

    # Secondary x-axis mappers
    first_change: tuple[str, int] = df[ifill("config", "pbt_epoch")].diff().iloc[1:].ne(0).idxmax()  # pyright: ignore[reportAssignmentType]
    perturbation_interval = df.attrs.get("perturbation_interval", None)
    if not perturbation_interval:
        try:
            perturbation_interval = df.current_step.loc[(first_change[0], first_change[1] - 1)].item()
        except AttributeError as ae:
            # might be a DataFrame
            if "item" not in str(ae):
                raise
            try:
                perturbation_interval = int(df.current_step.loc[(first_change[0], first_change[1] - 1)].iloc[0, 0])
            except Exception as e:
                logger.error("Failed to get perturbation interval at %s %r", (first_change[0], first_change[1] - 1), e)
                remote_breakpoint()
                pass
                raise
        except KeyError:
            df = _drop_duplicate_steps(df)
            perturbation_interval = df.attrs["perturbation_interval"]
    secondary_to_main = lambda xs: (xs + 0.5) * perturbation_interval  # noqa: E731
    main_to_secondary = lambda xs: (xs - perturbation_interval / 2) / perturbation_interval  # noqa: E731
    num_pbt_epochs = df[ifill("config", "pbt_epoch")].max().item() + 1

    # Select the group-by columns from MultiIndex columns
    group_cols = [("config", k) for k in group_by]
    # Extract the group-by columns as a DataFrame (each column is 1D)
    if ("config", "pbt_group_key") in group_cols and ("config", "pbt_group_key") not in df:
        _, df = which_continued(df)  # this sets pbt_group_key
    try:
        try:
            stat_columns = df[group_stat].droplevel(0, axis=1)
        except ValueError as ve:
            if "No axis named 1" not in str(ve):
                raise
            # If group_stat is a series cannot drop levels
            logger.debug("Failed to drop level from group_stat %s", group_stat)
            target_level = df.current_step.columns.nlevels - 1
            stat_columns = df[group_stat].to_frame()
            # Need to turn into frame with same depth
            if stat_columns.columns.nlevels != target_level:
                stat_columns.columns = pd.MultiIndex.from_arrays([[v] for v in ifill(group_stat, n=target_level)])
        group_values: list[pd.Series] = [
            df.current_step.droplevel(0, axis=1),
            *(df[col] for col in group_cols),
            df.config.seed,
            stat_columns,
        ]
    except AttributeError as ae:
        if "seed" not in str(ae):
            raise
        df.loc[:, ifill("config", "seed")] = df.config.cli_args.seed.to_numpy()
        df.sort_index(axis=1, inplace=True)
        group_values: list[pd.Series] = [
            df.current_step.droplevel(0, axis=1),
            *(df[col] for col in group_cols),
            df.config.seed,
            stat_columns,  # pyright: ignore[reportPossiblyUnboundVariable]
        ]
    except KeyError as ke:
        logger.exception("Failed to get group columns %s from df columns %s", group_cols, df.columns)
        remote_breakpoint()

    # Combine into a DataFrame for groupby
    try:
        group_df = pd.concat(group_values, axis=1, keys=["current_step", *group_by, "seed", group_stat], copy=False)
    except AssertionError:
        remote_breakpoint()
    # Problem we get duplicated values as the dfs contain their parents data - need to drop these when we aggregate
    group_df = group_df[~group_df.duplicated(keep="first")]
    group_df.columns = ["current_step", *group_by, "seed", group_stat]  # flatten for groupby
    group_df = group_df.drop("seed", axis=1)

    # Example usage: grouped = df.groupby([group_df[k] for k in group_keys])
    num_metrics = len(metrics)
    ax2 = secax = None
    restored_figure = False
    if fig is None:
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
    elif len(metrics) > 1:
        logger.warning("Cannot use a figure for more than one metric, creating a new figure.")
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
    else:
        from matplotlib.axes._secondary_axes import SecondaryAxis

        main_axes = []
        twin_axes = []
        secondary_axes = []

        for a in fig.axes:
            if isinstance(a, SecondaryAxis):
                secondary_axes.append(a)
            elif a.get_shared_x_axes().joined(a, fig.axes[0]) and a is not fig.axes[0]:
                twin_axes.append(a)
            else:
                main_axes.append(a)

        assert len(main_axes) == 1, "Expected exactly one main axis in the provided figure."
        assert len(twin_axes) <= 1, "Expected at most one twin axis in the provided figure."
        assert len(secondary_axes) <= 1, "Expected at most one secondary axis in the provided figure."
        ax = main_axes[0]
        ax2 = twin_axes[0] if twin_axes else None
        secax = secondary_axes[0] if secondary_axes else None
        restored_figure = True

    if num_metrics == 1:
        axes = [axes]

    error_figures: dict[str, Figure] = {}
    # continued_runs = which_continued(df)
    # cont_mask = df.index.get_level_values("run_id").isin(continued_runs)
    # df.loc[cont_mask, ("config", "__pbt_main_branch__")] = True

    for i, metric in enumerate(metrics):
        if plot_option.exclude(metric, group_by):
            logger.warning("Skipping metric %s as it is excluded by plot options.", metric)
            continue
        if final_metric_was_none:
            pbt_metric = metric
        if (metric == "episode_reward_mean" or pbt_metric == "episode_reward_mean") and "episode_reward_mean" not in df:
            from ray_utilities.constants import EPISODE_RETURN_MEAN, EPISODE_RETURN_MEAN_EMA  # noqa: PLC0415

            # introduced it later
            if EPISODE_RETURN_MEAN_EMA in df.evaluation:
                backport_metric = ("evaluation", EPISODE_RETURN_MEAN_EMA)
            else:
                backport_metric = ("evaluation", EPISODE_RETURN_MEAN)
            logger.info(
                "Metric 'episode_reward_mean' was not found in the data, changing metric to: %s", backport_metric
            )
            if pbt_metric == "episode_reward_mean":
                pbt_metric = backport_metric
            if metric == "episode_reward_mean":
                metric = backport_metric  # noqa: PLW2901
        assert pbt_metric is not None
        ax: Axes = axes[i]
        if ax2 is None:
            ax2 = ax.twinx()
        ax_2handles = ax2_labels = None
        ax2.set_ylabel(" ".join(group_stat.split("_")))
        # we did not save perturbation interval, check where pbt_epoch first changes
        # Use transform to add it relative to the data

        metric_key = metric if isinstance(metric, str) else metric[-1]

        plot_df: pd.DataFrame = pd.concat(
            [
                (
                    df[["current_step", metric]]
                    if isinstance(metric, str)
                    else df[[ifill("current_step"), ifill(*metric)]]
                ),
                df[[ifill("config", group_by[0]), ifill("config", "__pbt_main_branch__")]],
                df[("config", group_stat)].round(10),
                *(
                    ()
                    if metric == pbt_metric
                    else [df[pbt_metric if isinstance(pbt_metric, str) else ifill(*pbt_metric)]]
                ),
                *(() if "pbt_epoch" == group_by[0] else [df[[ifill("config", "pbt_epoch")]]]),
            ],
            axis=1,
        )
        pbt_metric_key = pbt_metric if isinstance(pbt_metric, str) else pbt_metric[-1]
        if metric != pbt_metric and pbt_metric_key == metric_key:
            pbt_metric_key = f"pbt_{pbt_metric_key}"
        plot_df.columns = (
            ["current_step", metric_key, group_by[0], "__pbt_main_branch__", group_stat]
            + ([pbt_metric_key] if metric != pbt_metric else [])
            + (["pbt_epoch"] if "pbt_epoch" != group_by[0] else [])
        )
        plot_df["__pbt_main_branch__"] = plot_df["__pbt_main_branch__"].infer_objects(copy=False).fillna(False)  # noqa: FBT003
        # plot_df.sort_values(["training_iteration", "__pbt_main_branch__", metric], inplace=True)
        assert group_stat in plot_df

        # TODO: for minibatch size have two different max sets
        if group_stat == "minibatch_size":
            # for consistency could scale upt to 8192 for all cases.
            pass
        if group_stat == "gradient_clip":
            plot_df[group_stat] = plot_df[group_stat].fillna(float("inf"))
        color_map, cmap, norm = make_cmap(plot_df[group_stat], log=log)
        # Sort each subgroup by their std from highest to lowest
        # Sort plot_df within each (group_by[0], group_stat) group by the std of the metric (descending)
        if sort_by_std := True:
            std_by_group = (
                plot_df.groupby([group_by[0], group_stat])[metric_key]  # no multiindex here
                .std()
                .sort_values(ascending=False)
            )
            # Reorder plot_df so that rows belonging to groups with higher std appear first
            plot_df["__group_sort__"] = list(zip(plot_df[group_by[0]], plot_df[group_stat], strict=True))
            plot_df["__group_sort__"] = pd.Categorical(
                plot_df["__group_sort__"], categories=list(std_by_group.index), ordered=True
            )
        else:
            plot_df["__group_sort__"] = 0

        # Assign ranks for each epoch based on their last iteration metric value
        # TODO: do this by final_metric_key?
        max_group_steps = plot_df.groupby([group_by[0], group_stat, "current_step"])[pbt_metric_key].mean()
        max_group_steps_frame = max_group_steps.to_frame().reset_index()
        # Select only the value from highest current step
        max_group_current_step = max_group_steps_frame.groupby([group_by[0], group_stat])["current_step"].max()
        max_step_indexer = (
            max_group_current_step.to_frame().reset_index().set_index([group_by[0], group_stat, "current_step"])
        )
        group_ranks = max_group_steps.loc[max_step_indexer.index].groupby([group_by[0]]).rank(method="max")
        max_rank = group_ranks.max()
        second_max_rank = group_ranks[group_ranks != max_rank].max()
        min_rank = group_ranks.min()
        ranks_to_keep = {max_rank, second_max_rank, min_rank}

        # For each (group_by[0], group_stat) assign their rank based on last iteration metric value
        # Assign group_ranks to the respective rows with matching (group_by[0], group_stat)

        try:
            plot_df["__group_rank__"] = plot_df[[group_by[0], group_stat]].apply(
                lambda x: group_ranks.loc[(x[group_by[0]], x[group_stat])].item(), axis=1
            )
        except KeyError:  # noqa: TRY203
            # group stat is nan. Possibly None value in choices
            remote_breakpoint()
            raise

        # ATTENTION # XXX - for older runs the pbt_epoch might go into the next epoch if a run was continued!

        # As there is not last perturbation check which run is at the end highest and sort it to the top

        # TODO: Does likely not work for uneven lengths of runs, i.e. when we tune batch_size
        # then need to check for current_step max
        if plot_option.plot_reduced and not plot_option.main_only:
            max_stat = plot_df[group_stat].max()
            min_stat = plot_df[group_stat].min()
            stats_to_keep = {max_stat, min_stat}
            continued_runs, df = which_continued(df)
            # Need to know which the main groups are
            # Remove those that do not have the highest rank, second highest rank, lowest rank,
            # And highest lowest group_stat, however keep last group_stat always

            def filter_groups(group: pd.DataFrame) -> pd.DataFrame:
                main_group = group.index.get_level_values("run_id").isin(continued_runs.index).any().item()  # noqa: B023
                example = group.iloc[0]
                logger.debug(
                    "Filtering group: epoch %s stat %s value - Rank %s Main group: %s Keep: %s",
                    example[group_by[0]],
                    example[group_stat],
                    example["__group_rank__"],
                    main_group,
                    main_group or example["__group_rank__"] in ranks_to_keep or example[group_stat] in stats_to_keep,  # noqa: B023
                )

                def keep_row(row: pd.Series) -> bool:
                    r = row["__group_rank__"]
                    s = row[group_stat]

                    return main_group or r in ranks_to_keep or s in stats_to_keep  # noqa: B023

                return group[group.apply(keep_row, axis=1)]

            logger.debug("Applying group filtering to reduce clutter.... Shape before %s", plot_df.shape)
            plot_df = plot_df.groupby([group_by[0], group_stat], group_keys=False)[plot_df.columns].apply(filter_groups)
            logger.debug("Shape after %s", plot_df.shape)
        plot_df = plot_df.sort_values(
            ["training_iteration", "__pbt_main_branch__", "__group_sort__", "__group_rank__", metric_key]
        )
        # Iterators:
        # Last data should not be needed if group_by != pbt_epoch
        try:
            max_epoch = plot_df["pbt_epoch"].max()
            last_data = plot_df[plot_df["pbt_epoch"] == max_epoch]
        except KeyError:
            # will happen if group_by[0] != "pbt_epoch", but should now always be present
            max_epoch = 0
            # remote_breakpoint()
        else:
            # TODO: What about those runs that were trained too long
            last_data = last_data.sort_values(["training_iteration", "__group_sort__", "__group_rank__", metric_key])
            last_group_iter = iter(last_data.groupby([group_by[0], group_stat], sort=False))
            # Remove overstepped trials
        if plot_df.current_step.max() >= 1_175_000 or 1 < max_epoch <= 10:
            plot_df = plot_df[plot_df["current_step"] <= 1_179_648]
        elif max_epoch == 0 or max_epoch >= 24:
            plot_df = plot_df[plot_df["current_step"] <= 1_048_576]

        if pbt_plot_interval:
            if pbt_plot_interval is True:
                pbt_plot_interval = max(2, (4 if max_epoch >= 20 else max_epoch // 5))
            # do not draw for only one epoch
            if secax is None:
                secax = ax.secondary_xaxis("top", functions=(main_to_secondary, secondary_to_main), transform=None)
            secax.set_xlabel("PBT Epoch", labelpad=4)
            if max_epoch == 0:
                secax.xaxis.label.set_visible(False)
            else:
                secax.set_xticks([e for e in range(num_pbt_epochs) if e % pbt_plot_interval == 1])
            # Show tick labels for secondary xaxis, inside and closer to the plot
            secax.xaxis.set_tick_params(which="both", bottom=False, top=False, labelbottom=True, labeltop=False, pad=-5)
            secax.xaxis.set_label_position("top")
            # Also move the label closer to the plot
            secax.set_xlabel("PBT Epoch", labelpad=4)
        if max_epoch == 0:
            # Assume baseline run
            # Shade the background in steps of 147456
            step_size = 147456
            x_min, x_max = ax.get_xlim()
            current = x_min
            color_idx = 0
            background_colors = ["#f0f0f0", "#909090"]
            while current < x_max:
                left = current
                right = min(current + step_size, x_max)
                _shade_background(ax, background_colors[color_idx % 2], left, right)
                current = right
                color_idx += 1
        # Move main x-axis tick labels closer to the plot
        ax.xaxis.set_tick_params(pad=2)
        ax.set_xlabel("Environment Steps", labelpad=3)
        if log:
            ax2.set_yscale("log", base=log_base or 10)

        grouper = plot_df.groupby([group_by[0], group_stat], sort=False)

        # Plot Non continues
        # Plot each group separately to control color
        last_main_group = None
        last_epoch = -1
        # Track background shading for group changes
        background_colors = ["#f0f0f0", "#909090"]
        last_bg_epoch = None
        bg_color_idx = 0
        prev_group: pd.DataFrame = None  # type: ignore[assignment]
        seen_labels = set()
        # TODO: pbt_epoch is first group, if not grouping over epoch ,,,,
        plotted_rest_this_epoch = False
        GROUP_STAT_COLOR = "blue"
        # When we plot all epochs at the same time and only check any() for the main group we will have multiple main groups
        group_by_keys_only = len(group_by) == 1 and group_by[0] == "pbt_group_key"
        plot_main_group = "pbt_epoch" in group_by or not group_by_keys_only
        for i, ((group0, stat_val), group) in enumerate(grouper):
            if group_by[0] == "pbt_epoch":
                pbt_epoch = group0
            elif "pbt_epoch" in group:
                pbt_epoch = group["pbt_epoch"].min()
            else:
                pbt_epoch = 0
            # Add background shade when group changes
            if pbt_epoch == max_epoch:
                (pbt_epoch, stat_val), group = next(last_group_iter)  # noqa: PLW2901

            if pbt_epoch != last_epoch:
                last_epoch = pbt_epoch
                plotted_rest_this_epoch = False
            if "DEBUG" in os.environ and pbt_epoch > 1:
                break
            if plot_main_group and (
                group["__pbt_main_branch__"].any()
                or (pbt_epoch == max_epoch and group["__group_rank__"].max() == max_rank)
            ):
                # If we do not group by epoch but by pbt_group_key we should not plot a main group.
                if last_main_group is not None and pbt_epoch > 0:
                    # Add connection to last points
                    group = _connect_groups(last_main_group, group, group_stat)  # noqa: PLW2901
                logger.debug("Main group: %s %s", pbt_epoch, stat_val)
                last_main_group = group.copy()
                # Print the group_stat value over time on the secondary y-axis
                if not restored_figure:
                    sns.lineplot(
                        data=group[["current_step", group_stat]].drop_duplicates().sort_values("current_step"),
                        x="current_step",
                        y=group_stat,
                        ax=ax2,
                        color=GROUP_STAT_COLOR,  # color_map[stat_val],
                        linestyle="-",
                        linewidth=2,
                        legend=True,
                        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                    )
                # Add one of the legend handles to the legend of ax 1

                # On a secondary axis we want to plot the group_stat value over time
                # Plot group_stat value over time on a secondary y-axis
            elif plot_option.main_vs_second_best:
                if group["__group_rank__"].max() != second_max_rank:
                    continue
                logger.debug("Plotting second best group: %s %s", pbt_epoch, stat_val)
            elif plot_option.main_vs_rest:
                if plotted_rest_this_epoch:
                    continue
                plotted_rest_this_epoch = True
                # Group all other groups together for the original frame for thich pbt_epoch
                try:
                    if pbt_epoch != max_epoch:
                        group = plot_df[(~plot_df.__pbt_main_branch__) & (plot_df[group_by[0]] == group0)]
                    else:
                        plot_df_last = plot_df[plot_df["pbt_epoch"] == max_epoch]
                        group = plot_df_last[plot_df_last["__group_rank__"] < max_rank]
                    group = group.copy()
                    group[group_stat] = "other"  # <-- sets on a copy
                except IndexError:
                    logger.error("Failed to group rest for epoch %s", pbt_epoch)
                    remote_breakpoint()
                    raise
                except Exception as e:
                    logger.error("Failed to group rest for epoch %s: %r", pbt_epoch, e)
                    remote_breakpoint()
                    raise
            elif plot_option.main_only and not group_by_keys_only:
                logger.debug("Skipping non-main group: %s %s", pbt_epoch, stat_val)
                continue
            elif last_main_group is not None and pbt_epoch > 0 and "pbt_epoch" in group_by:
                logger.debug("Plotting non-main group: %s %s", pbt_epoch, stat_val)
                group = _connect_groups(last_main_group, group, group_stat)  # noqa: PLW2901

            # Shading: - if we plot groups only shade below
            if not group_by_keys_only and (last_bg_epoch is None or pbt_epoch != last_bg_epoch):
                logger.debug("plotting shade %s", pbt_epoch)
                # Shade the region for this epoch
                try:
                    if last_bg_epoch is not None:
                        # Shade from previous group's min step to its max step (the region of the previous group)
                        prev_min: float = prev_group["current_step"].min()
                        prev_max: float = prev_group["current_step"].max()
                        _shade_background(ax, background_colors[bg_color_idx % 2], prev_min, prev_max)
                        bg_color_idx += 1
                    else:
                        # Shade from left edge to first group's min step
                        curr_min = group["current_step"].min()
                        _shade_background(ax, background_colors[bg_color_idx % 2], ax.get_xlim()[0], curr_min)
                        bg_color_idx += 1
                except Exception as e:
                    logger.error("Failed to shade background for epoch %s: %r", pbt_epoch, e)
                    remote_breakpoint()
            if isinstance(pbt_epoch, str):
                remote_breakpoint()

            last_bg_epoch = pbt_epoch
            prev_group = group
            try:
                color_map[stat_val]
            except KeyError:
                remote_breakpoint()
            # Plot
            # if len(group.index) != group.index.nunique():
            #    logger.warning(
            #        "Group has duplicated indices: epoch %s stat %s len %s unique %s",
            #        pbt_epoch,
            #        stat_val,
            #        len(group.index),group.index.unique(),
            #    )
            # From the group join we have duplicated indices for the main branch but that does not cause errors
            # try:
            #    group = group.drop_duplicates(keep="last")
            # except Exception as e:
            #    logger.error("Failed to drop duplicates for epoch %s stat %s: %r", pbt_epoch, stat_val, e)
            #    remote_breakpoint()
            # else:
            #    if len(group.index) != group.index.nunique():
            #        remote_breakpoint()

            if not restored_figure:
                try:
                    _plot_metrics_of_group(
                        group,
                        metric_key,
                        ax,
                        color_map[stat_val],
                        background_colors=background_colors,
                        label=str(stat_val) if stat_val not in seen_labels else None,
                        group_by_keys_only=group_by_keys_only,
                    )
                except Exception as e:
                    logger.error("Failed to plot group for epoch %s stat %s: %r", pbt_epoch, stat_val, e)
                    group = group.set_index("current_step", append=True)
                    # duplicated = group.index[group.index.duplicated(keep=False)].unique()
                    # Keep last as these are the one we continue with;
                    # however there are examples where the group_stat was not consistent
                    if (
                        group_stat == "batch_size"
                        and (df.config.experiment_id.iloc[0] != "tws47f2512191818752f4").all()
                    ):
                        if (df.batch_size.iloc[0] == df.batch_size).all().item():
                            remote_breakpoint(5679)
                    group = group[~group.index.duplicated(keep="last")]
                    # duplicated = group.index.duplicated(keep=False)
                    # dupl_cols = group.loc[duplicated]
                    # wired thing both duplicated had wrong batch_size values
                    group = group.reset_index(level="current_step")
                    _plot_metrics_of_group(
                        group,
                        metric_key,
                        ax,
                        color_map[stat_val],
                        background_colors=background_colors,
                        label=str(stat_val) if stat_val not in seen_labels else None,
                        group_by_keys_only=group_by_keys_only,
                    )

            seen_labels.add(stat_val)
        # Shade from last group's max step to right edge
        if "prev_group" in locals() and prev_group is not None:
            prev_max = prev_group["current_step"].max()
            _shade_background(ax, background_colors[bg_color_idx % 2], prev_max, ax.get_xlim()[1])

        # Add a colorbar for the continuous group_stat
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        sm.set_clim(norm.vmin * 0.95, norm.vmax * 1.10)  # pyright: ignore[reportOptionalOperand]
        ax.set_xlim(-1000, plot_df.current_step.max() + 1000)
        ax2.set_ylim(norm.vmin * 0.95, norm.vmax * 1.10)  # pyright: ignore[reportOptionalOperand]
        # Make the colorbar 20% smaller by adjusting its fraction
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=True)
        if plot_option.colorbar:
            remote_breakpoint()
            cbar = plt.colorbar(sm, ax=ax2, pad=0.01, fraction=0.075)  # default fraction is ~0.1
            # Remove all ticks from both sides of the colorbar
            cbar.ax.tick_params(axis="both", which="both", left=False, right=True, labelleft=False, labelright=False)
            cbar.set_ticks([])
            # Right tick labels are bleeding into the colormap if we use fancytoolbox legend
            # Move right y-axis tick labels further right
            # labels overlap with ticks from colorbar
            for label in ax2.get_yticklabels():
                label.set_x(label.get_position()[0] + (0.04 if log else 0.025))
        else:
            # remove any existing colorbar
            if ax2.images and ax2.images[0].colorbar is not None:
                cbar = ax2.images[0].colorbar

        # Align the limits of ax2 and the colorbar to be identical

        # ax.set_title(f"Metric: {metric_key}")
        # ax.set_xlabel("Training Steps")
        y_label = " ".join(
            metric_key.replace("episode_reward_mean", "episode reward (ema)").split("_"),
        )
        ax.set_ylabel(y_label)
        # Custom legend: one entry per unique pbt_group_key
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels, keep order
        seen = set()
        unique = []
        for h, l in zip(handles, labels, strict=True):
            if l not in seen:
                unique.append((h, l))
                seen.add(l)
        # Only keep entries where label is a pbt_group_key (not e.g. 'mean', etc.)
        group_keys = [lbl for _, lbl in unique if lbl is not None]
        # Remove duplicates while preserving order of appearance
        seen_keys = set()
        group_keys_ordered = []
        for k in group_keys:
            if k not in seen_keys:
                group_keys_ordered.append(k)
                seen_keys.add(k)

        # Sort group_keys_ordered by their group_stat value (convert to float if possible)
        def _try_float(x):
            try:
                return float(x)
            except Exception:
                return x

        group_keys_sorted = sorted(group_keys_ordered, key=_try_float, reverse=True)
        # Filter and order legend entries according to group_keys_sorted
        filtered_sorted = []
        for key in group_keys_sorted:
            for h, lbl in unique:
                if lbl == key:
                    filtered_sorted.append((h, lbl))
                    break
        if filtered_sorted:
            handles, labels = zip(*filtered_sorted, strict=True)
        ax2.legend()
        if (legend2 := ax2.get_legend()) is not None:
            ax_2handles, ax2_labels = ax2.get_legend_handles_labels()
            legend2.remove()
            # if ax_2handles:
            #    handles = (*handles, ax_2handles[0])
            #    labels = (*labels, ax2_labels[0])
            # else:
            h2 = plt.Line2D([], [], color=GROUP_STAT_COLOR, label="best")
            handles = (h2, *handles)
            labels = ("best", *labels)
        # Place the legend below the plot in a fancybox
        group_stat = str(group_stat)
        if group_stat == "lr":
            legend_title = "learning rate"
            ax2.set_ylabel("learning rate")
            # Apply scientific notation to small float labels
            if isinstance(labels, tuple):
                labels = list(labels)
            formatted_labels = []
            for label in labels:
                try:
                    val = float(label)
                    # Check if it's a small float and not already in scientific notation
                    if val < 0.001 and len(str(label)) > 7 and "e" not in str(label).lower():
                        formatted_labels.append(f"{val:.2e}")
                    else:
                        formatted_labels.append(label)
                except (ValueError, TypeError):
                    formatted_labels.append(label)
            labels = tuple(formatted_labels)
        else:
            legend_title = " ".join(
                map(
                    str.lower,
                    group_stat.replace("train_batch_size_per_per_learner", "batch size")
                    .replace("accumulate_gradients_every", "grad accumu.")
                    .split("_"),
                )
            )
        try:
            legend = ax.legend(
                handles,
                labels,
                title=legend_title,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(len(labels), 5),
                fancybox=True,
                framealpha=0.8,
            )
        except ValueError as ve:
            if len(handles) == len(labels) == 0 and plot_option.main_only and not plot_df["__pbt_main_branch__"].any():
                return None, None
            logger.exception("Failed to create legend with handles %s and labels %s: %r", handles, labels, ve)
            remote_breakpoint()
        # Reduce legend font size by 2
        # fontsize = max(legend.get_texts()[0].get_fontsize() - 2, 1) if legend.get_texts() else 10
        # for text in legend.get_texts():
        #    text.set_fontsize(fontsize)
        # if legend.get_title() is not None:  # pyright: ignore[reportUnnecessaryComparison]
        #    legend.get_title().set_fontsize(fontsize)

        env_type = df.iloc[0].config.cli_args.env_type
        if not isinstance(env_type, str):
            env_type = env_type.item()
        agent_type = df.iloc[0].config.cli_args.agent_type
        if not isinstance(agent_type, str):
            agent_type = agent_type.item()
        if plot_option.title:
            ax.set_title(f"{env_type} - {agent_type.upper()} - {metrics[0]}")

        if plot_errors and not plot_option.main_only and not plot_option.plot_reduced:
            err_fig = plot_error_distribution(
                plot_df,
                metric_key,
                group_stat,
                plot_errors_type=plot_errors_type,
                palette=color_map,
                epoch_col=group_by[0],
                plot_option=plot_option,
            )
            if err_fig is not None:
                error_figures[metric_key] = err_fig
        if env_type in ENV_BOUNDS or env_type in MAX_ENV_LOWER_BOUND:
            y_min, y_max = ENV_BOUNDS.get(env_type, (None, None))
            if env_type in MAX_ENV_LOWER_BOUND:
                curr_y, _ = ax.get_ylim()
                if curr_y < MAX_ENV_LOWER_BOUND[env_type]:
                    y_min = MAX_ENV_LOWER_BOUND[env_type]
            ax.set_ylim(bottom=y_min, top=y_max)
    if restored_figure:
        plotting_context = sns.plotting_context(sns_context, font_scale=font_scale or 1.6)
        apply_context_fonts(fig, plotting_context)

    plt.tight_layout()
    if show:
        plt.show()
    return (fig, error_figures)


def _shade_background(ax: Axes, color: str, left: float, right: float, **kwargs):
    kwargs.setdefault("alpha", 0.2)
    ax.axvspan(
        left,
        right,
        color=color,
        zorder=0,
        **kwargs,
    )


def plot_intra_group_variances(
    group_variance_global: pd.DataFrame,
    group_variance_per_epoch: pd.DataFrame,
    out_path: Path,
    path_base: str,
    metric_str: str,
    per_epoch_str: str,
    format: str,
    plot_option: PlotOption,
    *,
    large: bool,
):
    # Plot group_variance_global var column as horizontal bar chart
    large_str = "_large" if large else ""

    group_variance_global = group_variance_global.copy()
    group_variance_global.columns = group_variance_global.columns.get_level_values(-1)

    # Global
    split_idx = group_variance_global.index.str.split("=")
    values = pd.Series([_try_cast(v[-1]) for v in split_idx])
    group_variance_global.index = values
    group_variance_global.sort_index(inplace=True)

    group_stat = split_idx[0][0]
    group_variance_global.index.names = [group_stat]
    group_stat_str = " ".join(
        group_stat.replace("train_batch_size_per_per_learner", "batch size")
        .replace("Train Batch Size Per Learner", "Batch Size")
        .split("_"),
    )
    color_map, cmap, norm = make_cmap(values, log=group_stat in LOG_SETTINGS)

    fig_global, ax_global = plt.subplots(figsize=(6, max(4, len(group_variance_global) * 0.33)))

    sns.barplot(
        group_variance_global,
        x="var",
        y=group_stat,
        ax=ax_global,
        hue=group_stat,
        palette=color_map,
        orient="h",
        legend=False,
        hue_norm=norm,
    )
    # group_variance_global["var"].plot.barh(ax=ax_global, colormap=cmap)#color="skyblue")
    ax_global.set_xlabel("Variance")
    ax_global.set_ylabel(group_stat_str)
    if plot_option.title:
        ax_global.set_title("Group Variance Global")
    fig_global.tight_layout()

    # Save global variance plot
    bar_path_global = out_path.with_name(
        f"{path_base}{metric_str}{large_str}-group_variance_global{per_epoch_str}_bar.{format}"
    )
    fig_global.savefig(bar_path_global, format=format, bbox_inches="tight")
    plt.close(fig_global)
    logger.info("Saved group variance global bar chart to '%s'", bar_path_global)

    del values, split_idx, group_variance_global

    # Per Epoch
    group_variance_per_epoch = group_variance_per_epoch.copy()
    group_variance_per_epoch.columns = group_variance_per_epoch.columns.get_level_values(-1)
    level2 = group_variance_per_epoch.index.get_level_values("pbt_group_key")
    split_idx = level2.str.split("=")
    values = [_try_cast(v[-1]) for v in split_idx]
    group_variance_per_epoch.index = pd.MultiIndex.from_arrays(
        [group_variance_per_epoch.index.get_level_values("pbt_epoch"), pd.Series(values)],
        names=["pbt_epoch", group_stat],
    )
    group_variance_per_epoch.sort_index(inplace=True)
    group_variance_per_epoch.index.names = ["pbt_epoch", group_stat]

    # Plot group_variance_per_epoch var column as bar chart
    fig_epoch, ax_epoch = plt.subplots(figsize=(max(6, min(len(group_variance_per_epoch) * 0.15, 16)), 6))
    # group_variance_per_epoch["var"].plot.bar(ax=ax_epoch, color="lightcoral")
    sns.barplot(
        group_variance_per_epoch,
        x="pbt_epoch",
        y="var",
        ax=ax_epoch,
        hue=group_stat,
        palette=color_map,
        orient="v",
    )
    ax_epoch.set_ylabel("Variance")
    ax_epoch.set_xlabel("Epoch")
    if plot_option.title:
        ax_epoch.set_title("Group Variance Per Epoch")
    ax_epoch.tick_params(axis="x", rotation=45)
    fig_epoch.tight_layout()

    # Save per epoch variance plot
    bar_path_per_epoch = out_path.with_name(
        f"{path_base}{metric_str}{large_str}-group_variance_per_epoch{per_epoch_str}_bar.{format}"
    )
    fig_epoch.savefig(bar_path_per_epoch, format=format, bbox_inches="tight")
    plt.close(fig_epoch)
    logger.info("Saved group variance per epoch bar chart to '%s'", bar_path_per_epoch)
    return bar_path_global, bar_path_per_epoch
