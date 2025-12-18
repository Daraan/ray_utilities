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
    _connect_groups,
    _drop_duplicate_steps,
    _get_group_stat,
    get_and_check_group_stat,
    which_continued,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
            vmin=values.replace(0, nan).min(),
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
        if log and val <= 0:
            color_map[val] = "#cccccc"
    return color_map, cmap, norm


def plot_error_distribution(
    plot_df: pd.DataFrame,
    metric_key: str,
    group_stat: str,
    *,
    plot_errors_type: Literal["box", "violin"],
    palette: Mapping[Hashable, str] | None = None,
    epoch_col: str = "pbt_epoch",
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
    ax_err.set_title(f"{metric_key} distribution by {group_stat}")
    ax_err.set_xlabel(group_stat)
    ax_err.set_ylabel(metric_key)
    ax_err.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig_err


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
    show: bool = True,
    pbt_plot_interval: int = 4,
    plot_errors: bool | Literal["only"] | str | Sequence[str] = True,
    plot_errors_type: Literal["box", "violin"] = "box",
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
    log = log if log is not None else group_stat in LOG_SETTINGS

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
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)

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
        ax2 = ax.twinx()
        ax_2handles = ax2_labels = None
        ax2.set_ylabel(group_stat)
        # we did not save perturbation interval, check where pbt_epoch first changes
        # Use transform to add it relative to the data
        if pbt_plot_interval:
            secax = ax.secondary_xaxis("top", functions=(main_to_secondary, secondary_to_main), transform=None)
            secax.set_xlabel("PBT Epoch")
            secax.set_xticks([e for e in range(num_pbt_epochs) if e % pbt_plot_interval == 1])
            # Show tick labels for secondary xaxis, inside and closer to the plot
            secax.xaxis.set_tick_params(which="both", bottom=False, top=False, labelbottom=True, labeltop=False, pad=-2)
            secax.xaxis.set_label_position("top")
            # Also move the label closer to the plot
            secax.set_xlabel("PBT Epoch", labelpad=3)
        # Move main x-axis tick labels closer to the plot
        ax.xaxis.set_tick_params(pad=2)
        ax.set_xlabel("Environment Steps", labelpad=3)
        if log:
            ax2.set_yscale("log")

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
            # remote_breakpoint()
            raise

        # ATTENTION # XXX - for older runs the pbt_epoch might go into the next epoch if a run was continued!

        # As there is not last perturbation check which run is at the end highest and sort it to the top

        # TODO: Does likely not work for uneven lengths of runs, i.e. when we tune batch_size
        # then need to check for current_step max
        if plot_option.plot_reduced and not plot_option.main_only:
            max_stat = plot_df[group_stat].max()
            min_stat = plot_df[group_stat].min()
            stats_to_keep = {max_stat, min_stat}
            continued_runs = which_continued(df)
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
            max_epoch = 1
            remote_breakpoint()
        else:
            # TODO: What about those runs that were trained too long
            last_data = last_data.sort_values(["training_iteration", "__group_sort__", "__group_rank__", metric_key])
            last_group_iter = iter(last_data.groupby([group_by[0], group_stat], sort=False))
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
                        group = plot_df_last[plot_df["__group_rank__"] < max_rank]
                    group = group.copy()
                    group[group_stat] = "other"  # <-- sets on a copy
                except IndexError:
                    logger.error("Failed to group rest for epoch %s", pbt_epoch)
                    remote_breakpoint()
                except Exception as e:
                    logger.error("Failed to group rest for epoch %s: %r", pbt_epoch, e)
                    remote_breakpoint()
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

            try:
                if not group_by_keys_only:
                    # default case
                    sns.lineplot(
                        data=group,
                        x="current_step",
                        y=metric_key,
                        ax=ax,
                        # linestyle="--",
                        color=color_map[stat_val],
                        label=str(stat_val) if stat_val not in seen_labels else None,
                        linewidth=3,
                    )
                else:
                    # There will be jumps at the end of each epoch, split the data and plot separately
                    # to avoid lines going across the plot
                    last_epoch_group = None
                    epoch_grouper = group.groupby("pbt_epoch", sort=True)
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
                            color=color_map[stat_val],
                            label=str(stat_val) if stat_val not in seen_labels else None,
                            linewidth=3,
                        )
            except Exception as e:
                logger.error("Failed to plot group for epoch %s stat %s: %r", pbt_epoch, stat_val, e)
                group = group.set_index("current_step", append=True)
                duplicated = group.index[group.index.duplicated(keep=False)].unique()
                dupl_cols = group.loc[duplicated]
                remote_breakpoint()
                # wired thing both duplicated had wrong batch_size values
                group = group.reset_index(level="current_step")
            seen_labels.add(stat_val)
        # Shade from last group's max step to right edge
        if "prev_group" in locals() and prev_group is not None:
            prev_max = prev_group["current_step"].max()
            _shade_background(ax, background_colors[bg_color_idx % 2], prev_max, ax.get_xlim()[1])

        # Add a colorbar for the continuous group_stat
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        sm.set_clim(norm.vmin * 0.95, norm.vmax * 1.05)  # pyright: ignore[reportOptionalOperand]
        ax.set_xlim(-1000, plot_df.current_step.max() + 1000)
        ax2.set_ylim(norm.vmin * 0.95, norm.vmax * 1.05)  # pyright: ignore[reportOptionalOperand]
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
        ax.set_ylabel(metric_key)
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
        if (legend2 := ax2.get_legend()) is not None:
            ax_2handles, ax2_labels = ax2.get_legend_handles_labels()
            legend2.remove()
            if ax_2handles:
                handles = (*handles, ax_2handles[0])
                labels = (*labels, ax2_labels[0])
            else:
                h2 = plt.Line2D([], [], color=GROUP_STAT_COLOR, label=group_stat)
                handles = (h2, *handles)
                labels = (group_stat, *labels)
        # Place the legend below the plot in a fancybox
        try:
            legend = ax.legend(
                handles,
                labels,
                title=group_stat,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(len(labels), 6),
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
        ax.set_title(f"{env_type} - {agent_type.upper()} - {metrics[0]}")

        if plot_errors and not plot_option.main_only and not plot_option.plot_reduced:
            err_fig = plot_error_distribution(
                plot_df,
                metric_key,
                group_stat,
                plot_errors_type=plot_errors_type,
                palette=color_map,
                epoch_col=group_by[0],
            )
            if err_fig is not None:
                error_figures[metric_key] = err_fig

        if env_type in MAX_ENV_LOWER_BOUND:
            curr_y, _ = ax.get_ylim()
            if curr_y < MAX_ENV_LOWER_BOUND[env_type]:
                ax.set_ylim(bottom=MAX_ENV_LOWER_BOUND[env_type])

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
