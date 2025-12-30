#!/usr/bin/env python3
"""Crawl experiment directories and process hyperparam stats files."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ray_utilities.visualization.data import get_runs_from_submission

logger = logging.getLogger(__name__)

# Base paths to search
PATHS = [
    Path("outputs/shared/experiments"),
    Path("outputs/shared_backup"),
    Path("outputs/shared_backup/needs_sync"),
]

# Valid experiment prefixes
VALID_PREFIXES = ["Default-mlp-", "SYMPOL-mlp-"]


def is_valid_target_dir(dir_path: Path) -> bool:
    """Check if directory is a valid target (starts with Default-mlp-ENV or SYMPOL-mlp-ENV).

    Args:
        dir_path: Directory path to check

    Returns:
        True if directory name starts with any valid prefix
    """
    return any(dir_path.name.startswith(prefix) for prefix in VALID_PREFIXES)


def find_target_dirs(base_path: Path, prefix_filter: str | None = None) -> Iterator[Path]:
    """Find all valid target directories under base_path.

    Args:
        base_path: Root path to search
        prefix_filter: Optional prefix to filter directories ("Default-mlp-" or "SYMPOL-mlp-")

    Yields:
        Valid target directory paths
    """
    if not base_path.exists():
        logger.warning("Path does not exist: %s", base_path)
        return

    for item in base_path.iterdir():
        if item.is_dir() and is_valid_target_dir(item):
            if prefix_filter is None or item.name.startswith(prefix_filter):
                yield item


def find_group_dirs(target_dir: Path) -> Iterator[Path]:
    """Find all group directories (pbt-tune_tune:GROUP) under target directory.

    Args:
        target_dir: Target experiment directory

    Yields:
        Group directory paths
    """
    if not target_dir.exists():
        return

    for item in target_dir.iterdir():
        if item.is_dir() and item.name.startswith("pbt-tune_tune:"):
            yield item


def extract_group_name(group_dir_name: str) -> str:
    """Extract GROUP name from directory name (pbt-tune_tune:GROUP).

    Args:
        group_dir_name: Directory name like 'pbt-tune_tune:batch_size'

    Returns:
        Extracted GROUP name (e.g., 'batch_size')
    """
    if ":" in group_dir_name:
        return group_dir_name.split(":", 1)[1]
    return group_dir_name


def extract_env_name(target_dir_name: str) -> str:
    """Extract ENV name from target directory.

    Args:
        target_dir_name: Directory name like 'Default-mlp-LunarLander-v3'

    Returns:
        Full directory name (used as ENV identifier)
    """
    return target_dir_name


def extract_run_id(folder_name: str) -> str:
    """Extract RUN_ID from folder name by splitting at last '-'.

    Args:
        folder_name: Folder name containing RUN_ID

    Returns:
        Extracted RUN_ID
    """
    return folder_name.rsplit("-", 1)[-1]


def find_hyperparam_files(plots_dir: Path, metric: str) -> tuple[Path | None, Path | None]:
    """Find both hyperparam stats files for a given metric.

    Args:
        plots_dir: Path to plots directory
        metric: Metric name to search for

    Returns:
        Tuple of (per_epoch_file, reduced_file) - either can be None if not found
    """
    metric_dir = plots_dir / metric
    if not metric_dir.exists():
        return (None, None)

    # Find the two files
    per_epoch_files = list(metric_dir.glob("*-all-hyperparam_stats-per_epoch_reduced.txt"))
    reduced_files = list(metric_dir.glob("*-all-hyperparam_stats_reduced.txt"))

    per_epoch_file = per_epoch_files[0] if per_epoch_files else None
    reduced_file = reduced_files[0] if reduced_files else None

    return (per_epoch_file, reduced_file)


def load_hyperparam_file(file_path: Path) -> pd.DataFrame | None:
    """Load hyperparam stats file as pandas DataFrame.

    Args:
        file_path: Path to the stats file

    Returns:
        DataFrame with multi-index or None if loading fails
    """
    try:
        # Read file line by line to handle multi-index properly
        with file_path.open("r") as f:
            lines = f.readlines()

        # Parse header
        header = lines[0].split()

        # Parse data rows
        data_rows = []
        index_tuples = []
        last_category = None

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:  # Skip malformed lines
                continue

            # Check if first element looks like a metric category
            first_col = parts[0]
            if first_col in [
                "centered_metrics",
                "centered_metrics2",
                "gini_metrics",
                "kendall_metrics",
                "normalized_metrics",
                "normed_metrics",
            ]:
                # This is a new category
                last_category = first_col
                subcategory = parts[1]
                values = [float(x) for x in parts[2:]]
            else:
                # This is a continuation row (blank first column in original)
                # first_col is actually the subcategory
                subcategory = first_col
                values = [float(x) for x in parts[1:]]

            index_tuples.append((last_category, subcategory))
            data_rows.append(values)

        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=header, index=pd.MultiIndex.from_tuples(index_tuples))

    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None
    else:
        return df


def collect_env_data(
    base_paths: list[Path],
    valid_run_ids: set[str] | None = None,
    runs: dict | None = None,
    prefix_filter: str | None = None,
) -> dict[str, dict[str, dict[str, list[pd.DataFrame]]]]:
    """Collect all hyperparam data organized by ENV, metric, and file type.

    Args:
        base_paths: List of paths to search
        valid_run_ids: Optional set of valid RUN_IDs to filter by
        runs: Optional dict of SubmissionRun data keyed by run_id
        prefix_filter: Optional prefix to filter directories ("Default-mlp-" or "SYMPOL-mlp-")

    Returns:
        Nested dict: {env_name: {metric: {"per_epoch": [dfs], "reduced": [dfs]}}}
        Each DataFrame in the list has the GROUP name stored in df.attrs["group"]
    """
    env_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]] = defaultdict(
        lambda: defaultdict(lambda: {"per_epoch": [], "reduced": []})
    )

    for base_path in base_paths:
        if not base_path.exists():
            logger.warning("Path does not exist: %s", base_path)
            continue

        logger.info("Searching in: %s", base_path.absolute())

        for target_dir in find_target_dirs(base_path, prefix_filter=prefix_filter):
            env_name = extract_env_name(target_dir.name)
            logger.info("Found target dir: %s", env_name)

            for group_dir in find_group_dirs(target_dir):
                group_name = extract_group_name(group_dir.name)
                logger.debug("Processing group: %s", group_name)

                # Find all run folders
                for run_folder in group_dir.iterdir():
                    if not run_folder.is_dir():
                        continue

                    plots_dir = run_folder / "plots"
                    if not plots_dir.exists():
                        continue

                    # Extract RUN_ID from folder name
                    run_id = extract_run_id(run_folder.name)

                    # Filter by valid_run_ids if provided
                    if valid_run_ids is not None and run_id not in valid_run_ids:
                        logger.debug("Skipping run_id %s (not in YAML files)", run_id)
                        continue

                    # Find all metrics
                    for metric_dir in plots_dir.iterdir():
                        if not metric_dir.is_dir():
                            continue

                        metric = metric_dir.name
                        per_epoch_file, reduced_file = find_hyperparam_files(plots_dir, metric)

                        # Load and store per_epoch file
                        if per_epoch_file:
                            df = load_hyperparam_file(per_epoch_file)
                            if df is not None:
                                df.attrs["group"] = group_name
                                df.attrs["run_id"] = run_id
                                if runs and run_id in runs:
                                    df.attrs["submission_name"] = runs[run_id].get("submission_name")
                                    df.attrs["submission_id"] = runs[run_id].get("submission_id")
                                env_data[env_name][metric]["per_epoch"].append(df)
                                logger.debug(
                                    "Loaded per_epoch for %s/%s/%s/%s",
                                    env_name,
                                    metric,
                                    group_name,
                                    run_id,
                                )

                        # Load and store reduced file
                        if reduced_file:
                            df = load_hyperparam_file(reduced_file)
                            if df is not None:
                                df.attrs["group"] = group_name
                                df.attrs["run_id"] = run_id
                                if runs and run_id in runs:
                                    df.attrs["submission_name"] = runs[run_id].get("submission_name")
                                    df.attrs["submission_id"] = runs[run_id].get("submission_id")
                                env_data[env_name][metric]["reduced"].append(df)
                                logger.debug(
                                    "Loaded reduced for %s/%s/%s/%s",
                                    env_name,
                                    metric,
                                    group_name,
                                    run_id,
                                )

    return env_data


def round_to_n_sig_figs(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Round all numeric values in DataFrame to n significant figures.

    Args:
        df: DataFrame to round
        n: Number of significant figures

    Returns:
        DataFrame with rounded values
    """

    def round_value(x):
        if pd.isna(x) or x == 0:
            return x
        try:
            return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
        except (ValueError, OverflowError):
            return x

    return df.map(round_value)


def concatenate_with_group_columns(
    dfs: list[pd.DataFrame], drop_duplicates: bool = False, merge_duplicates: bool = True
) -> pd.DataFrame:
    """Concatenate DataFrames with GROUP or GROUP_RUN_ID as first level of column multi-index.

    Args:
        dfs: List of DataFrames, each with attrs["group"] and attrs["run_id"] set
        drop_duplicates: If True, drop duplicate columns after concatenation
        merge_duplicates: If True, merge duplicate columns by averaging their values

    Returns:
        Concatenated DataFrame with multi-level columns (GROUP/GROUP_RUN_ID, original_columns)
    """
    if not dfs:
        return pd.DataFrame()

    # Count occurrences of each group to determine if we need RUN_ID suffix
    group_counts: dict[str, int] = {}
    for df in dfs:
        group = df.attrs.get("group", "unknown")
        group_counts[group] = group_counts.get(group, 0) + 1

    # Track which groups we've seen to assign unique names
    group_counter: dict[str, int] = defaultdict(int)

    # Create multi-level columns for each DataFrame
    dfs_with_groups: list[pd.DataFrame] = []
    for df in dfs:
        group = df.attrs.get("group", "unknown")
        run_id = df.attrs.get("run_id", "unknown")

        # If multiple runs for this group, use GROUP_RUN_ID format
        if group_counts[group] > 1:
            column_name = f"{group}_{run_id}"
        else:
            column_name = group

        group_counter[group] += 1

        # Round values to 2 significant figures
        df_rounded = round_to_n_sig_figs(df, n=2)

        # Add group as first level of column index
        df_rounded.columns = pd.MultiIndex.from_product([[column_name], df.columns], names=["GROUP", "statistic"])
        dfs_with_groups.append(df_rounded)

    # Concatenate along columns
    result = pd.concat(dfs_with_groups, axis=1)

    # Merge duplicate columns by averaging if requested
    if merge_duplicates and not drop_duplicates:
        # Find duplicate column names
        col_counts = {}
        for col in result.columns:
            col_counts[col] = col_counts.get(col, 0) + 1

        duplicates = {col for col, count in col_counts.items() if count > 1}
        if duplicates:
            logger.info("Merging %d duplicate column groups by averaging", len(duplicates))
            # Create new dataframe with averaged duplicates
            new_cols = []
            processed = set()
            for col in result.columns:
                if col in processed:
                    continue
                if col in duplicates:
                    # Average all columns with this name
                    dup_cols = [c for c in result.columns if c == col]
                    averaged = result[dup_cols].mean(axis=1)
                    new_cols.append((col, averaged))
                    processed.add(col)
                else:
                    new_cols.append((col, result[col]))
                    processed.add(col)

            # Reconstruct dataframe
            result = pd.DataFrame(dict(new_cols), index=result.index)

    # Drop duplicate columns if requested
    elif drop_duplicates:
        # Find duplicate column names
        seen = set()
        cols_to_keep = []
        for col in result.columns:
            if col not in seen:
                seen.add(col)
                cols_to_keep.append(col)
        if len(cols_to_keep) < len(result.columns):
            logger.info("Dropping %d duplicate columns", len(result.columns) - len(cols_to_keep))
            result = result[cols_to_keep]

    return result


def format_mean_std_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    """Format mean and std columns as 'mean (± std)' for LaTeX output.

    Args:
        df: DataFrame with multi-level columns containing 'mean' and 'std' statistics

    Returns:
        DataFrame with mean and std combined into single columns
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Get unique group names (level 0)
    groups = df.columns.get_level_values(0).unique()

    new_cols = []
    for group in groups:
        # Get all columns for this group
        group_cols = df[group]

        if "mean" in group_cols.columns and "std" in group_cols.columns:
            # Create combined column with format: mean (± std)
            combined = group_cols.apply(
                lambda row: f"{row['mean']:.2g} (± {row['std']:.2g})"
                if pd.notna(row["mean"]) and pd.notna(row["std"])
                else "",
                axis=1,
            )
            new_cols.append(((group, "mean (± std)"), combined))

            # Add other columns that aren't mean or std
            for col in group_cols.columns:
                if col not in ["mean", "std"]:
                    new_cols.append(((group, col), group_cols[col]))
        else:
            # No mean/std pair, keep all columns as-is
            for col in group_cols.columns:
                new_cols.append(((group, col), group_cols[col]))

    # Reconstruct with multi-index columns
    if new_cols:
        result = pd.DataFrame(dict(new_cols), index=df.index)
        return result
    return df


def save_env_results(
    env_name: str,
    env_data: dict[str, dict[str, list[pd.DataFrame]]],
    output_dir: Path,
    drop_duplicates: bool = False,
    yaml_suffix: str = "",
    filter_by_size: bool = False,
    merge_duplicates: bool = True,
) -> None:
    """Save concatenated results for an environment.

    Args:
        env_name: Name of the environment
        env_data: Dict with metric -> {"per_epoch": [dfs], "reduced": [dfs]}
        output_dir: Base output directory (PATHS[0])
        drop_duplicates: If True, drop duplicate columns
        yaml_suffix: Optional suffix to append to filenames (from YAML files)
        filter_by_size: If True, split data into large/small groups based on submission info
        merge_duplicates: If True, merge duplicate columns by averaging
    """
    # Create output directory for this environment
    env_output_dir = output_dir / env_name
    env_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each metric
    for metric, file_types in env_data.items():
        for file_type, dfs in file_types.items():
            if not dfs:
                continue

            if filter_by_size:
                # Separate into large and small groups
                large_dfs = []
                small_dfs = []
                for df in dfs:
                    submission_name = df.attrs.get("submission_name", "")
                    submission_id = df.attrs.get("submission_id", "")
                    is_large = (submission_name and ("large" in submission_name or "8192" in submission_name)) or (
                        submission_id and ("large" in submission_id or "8192" in submission_id)
                    )
                    if is_large:
                        large_dfs.append(df)
                    else:
                        small_dfs.append(df)

                # Process large group
                if large_dfs:
                    concatenated = concatenate_with_group_columns(
                        large_dfs, drop_duplicates=drop_duplicates, merge_duplicates=merge_duplicates
                    )
                    if not concatenated.empty:
                        base_name = f"{env_name}_{metric}_{file_type}_hyperparam_stats{yaml_suffix}_large"
                        txt_file = env_output_dir / f"{base_name}.txt"
                        tex_file = env_output_dir / f"{base_name}.tex"
                        concatenated.to_csv(txt_file, sep="\t")
                        logger.info("Saved: %s", txt_file)
                        # Format mean/std for LaTeX
                        latex_df = format_mean_std_for_latex(concatenated)
                        latex_str = latex_df.to_latex(multirow=True, multicolumn=True, escape=False)
                        tex_file.write_text(latex_str)
                        logger.info("Saved: %s", tex_file)

                # Process small group
                if small_dfs:
                    concatenated = concatenate_with_group_columns(
                        small_dfs, drop_duplicates=drop_duplicates, merge_duplicates=merge_duplicates
                    )
                    if not concatenated.empty:
                        base_name = f"{env_name}_{metric}_{file_type}_hyperparam_stats{yaml_suffix}_small"
                        txt_file = env_output_dir / f"{base_name}.txt"
                        tex_file = env_output_dir / f"{base_name}.tex"
                        concatenated.to_csv(txt_file, sep="\t")
                        logger.info("Saved: %s", txt_file)
                        # Format mean/std for LaTeX
                        latex_df = format_mean_std_for_latex(concatenated)
                        latex_str = latex_df.to_latex(multirow=True, multicolumn=True, escape=False)
                        tex_file.write_text(latex_str)
                        logger.info("Saved: %s", tex_file)
            else:
                # Concatenate DataFrames without filtering
                concatenated = concatenate_with_group_columns(
                    dfs, drop_duplicates=drop_duplicates, merge_duplicates=merge_duplicates
                )
                if concatenated.empty:
                    continue

                # Generate filenames
                base_name = f"{env_name}_{metric}_{file_type}_hyperparam_stats{yaml_suffix}"
                txt_file = env_output_dir / f"{base_name}.txt"
                tex_file = env_output_dir / f"{base_name}.tex"

                # Save as text
                concatenated.to_csv(txt_file, sep="\t")
                logger.info("Saved: %s", txt_file)

                # Save as LaTeX with formatted mean/std
                latex_df = format_mean_std_for_latex(concatenated)
                latex_str = latex_df.to_latex(multirow=True, multicolumn=True, escape=False)
                tex_file.write_text(latex_str)
                logger.info("Saved: %s", tex_file)


def crawl_experiments(
    yaml_files: list[Path] | None = None,
    drop_duplicates: bool = False,
    filter_by_size: bool = False,
    merge_duplicates: bool = True,
) -> None:
    """Main function to crawl through all experiment directories and process files.

    Args:
        yaml_files: Optional list of YAML submission files to filter RUN_IDs
        drop_duplicates: If True, drop duplicate columns in output
        filter_by_size: If True, split data into large/small groups based on submission info
        merge_duplicates: If True, merge duplicate columns by averaging (default: True)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    if not yaml_files:
        # No YAML files - process all Default directories
        logger.info("No YAML files provided, processing all Default-mlp-* directories")
        env_data = collect_env_data(PATHS, prefix_filter="Default-mlp-")
        output_dir = PATHS[0]
        logger.info("Saving results to: %s", output_dir.absolute())

        for env_name, metrics_data in env_data.items():
            logger.info("Processing environment: %s", env_name)
            save_env_results(
                env_name,
                metrics_data,
                output_dir,
                drop_duplicates=drop_duplicates,
                yaml_suffix="",
                filter_by_size=filter_by_size,
                merge_duplicates=merge_duplicates,
            )
    else:
        # Process each YAML file independently
        for yaml_file in yaml_files:
            logger.info("\n" + "=" * 80)
            logger.info("Processing YAML file: %s", yaml_file)
            logger.info("=" * 80)

            try:
                runs = get_runs_from_submission(yaml_file)
                valid_run_ids = set(runs.keys())
                logger.info("Loaded %d run_ids from %s", len(valid_run_ids), yaml_file)

                # Clean the YAML filename (same logic as in data.py)
                cleaned_name = (
                    yaml_file.name.removesuffix(".yaml")
                    .removeprefix("submissions")
                    .removeprefix("submission")
                    .strip("_")
                )
                yaml_suffix = f"_{cleaned_name}"

                # Determine prefix filter and output directory based on YAML name
                if "sympol" in yaml_file.name.lower():
                    prefix_filter = "SYMPOL-mlp-"
                    logger.info("SYMPOL YAML detected - filtering for SYMPOL-mlp-* directories")
                else:
                    prefix_filter = "Default-mlp-"
                    logger.info("Non-SYMPOL YAML - filtering for Default-mlp-* directories")

                # Collect data with appropriate prefix filter
                env_data = collect_env_data(
                    PATHS,
                    valid_run_ids=valid_run_ids,
                    runs=runs,
                    prefix_filter=prefix_filter,
                )

                # Determine output directory based on prefix
                output_dir = PATHS[0]
                logger.info("Saving results to: %s", output_dir.absolute())

                # Process and save results for each environment
                for env_name, metrics_data in env_data.items():
                    logger.info("Processing environment: %s", env_name)
                    save_env_results(
                        env_name,
                        metrics_data,
                        output_dir,
                        drop_duplicates=drop_duplicates,
                        yaml_suffix=yaml_suffix,
                        filter_by_size=filter_by_size,
                        merge_duplicates=merge_duplicates,
                    )

            except Exception as e:
                logger.error("Failed to process YAML file %s: %s", yaml_file, e)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl experiment directories and process hyperparam stats files.")
    parser.add_argument(
        "yaml_files",
        nargs="*",
        type=Path,
        help="Optional YAML submission files to filter RUN_IDs",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop duplicate columns in output files",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter data by size (large/small) based on submission info and save separate files",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge duplicate columns (default: merge by averaging)",
    )
    args = parser.parse_args()
    crawl_experiments(
        yaml_files=args.yaml_files,
        drop_duplicates=args.drop,
        filter_by_size=args.filter,
        merge_duplicates=not args.no_merge,
    )
