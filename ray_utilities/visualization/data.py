from __future__ import annotations

import concurrent.futures
import logging

# TODOS
# NOTE: That the current code stores and caches trials that might have failed/stopped and are restored later
# cause of this caching the trial data might not be complete!
# pyright: reportAttributeAccessIssue=warning
# ruff: noqa: G004
import pickle
import re
import time
import traceback
from ast import literal_eval
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)
from zipfile import ZIP_DEFLATED, ZipFile

import base62
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import Final, Literal, Sentinel, TypeVarTuple, Unpack

from ray_utilities.misc import ExperimentKey
from ray_utilities.testing_utils import remote_breakpoint
from ray_utilities.visualization._common import Placeholder, SubmissionRun, PlotOption, make_zip_arcname

try:
    from ray_submit import write_back
except ImportError:
    write_back = None  # type: ignore

try:
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
except ModuleNotFoundError:
    import yaml

    print(
        "ruamel.yaml not found, falling back to PyYAML which may not preserve formatting. Please install ruamel.yaml."
    )
    yaml_load = yaml.safe_load
    yaml_dump = yaml.safe_dump
else:
    yaml_load = yaml.load
    yaml_dump = yaml.dump

# from ray_utilities.constants import EPISODE_RETURN_MEAN_EMA
# from ray_utilities.testing_utils import remote_breakpoint

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PATHS = [Path("outputs/shared/experiments"), Path("outputs/shared_backup/needs_sync"), Path("outputs/shared_backup")]

DROP_COLUMNS = [
    "hostname",
    "done",
    "timers",
    ("config", "cli_args", "evaluate_every_n_steps_before_step"),
    ("config", "cli_args", "offline_loggers"),
    ("config", "cli_args", "fcnet_hiddens"),
    ("config", "cli_args", "perturbation_factors"),
    ("config", "cli_args", "comment"),
    ("config", "cli_args", "head_fcnet_hiddens"),
    ("config", "cli_args", "tune"),
    ("config", "cli_args", "log_level"),
    ("config", "cli_args", "log_stats"),
    ("config", "cli_args", "log_config"),
    ("config", "cli_args", "head_fcnet_activation"),
    ("config", "cli_args", "head_fcnet_bias_initializer"),
    ("config", "cli_args", "head_fcnet_kernel_initializer"),
    ("config", "cli_args", "fcnet_activation"),
    ("config", "cli_args", "fcnet_bias_initializer"),
    ("config", "cli_args", "fcnet_kernel_initializer"),
    ("config", "cli_args", "render_mode"),
    ("config", "cli_args", "buffer_length"),
    ("config", "cli_args", "gpu"),
    ("config", "_config_files"),
    ("config", "env_seed"),  # can be list in older versions
    # Maybe want to keep but not hashable
    ("config", "fork_from", "parent_time"),
]

DEFAULT_GROUP_BY = ("pbt_epoch", "pbt_group_key")

LOG_SETTINGS: dict[str, int] = {
    "lr": 2,
    "batch_size": 2,
    "minibatch_size": 2,
    "entropy_coeff": 2,
    "vf_clip_param": 10,
    "grad_clip": 10,
    "gamma": 10,
    "accumulate_gradients_every": 2,
}

MAX_ENV_LOWER_BOUND: dict[str, float] = {"LunarLander-v3": -750, "HalfCheetah-v5": -750}
"""Lower bound clip for plots"""

__t_margin = 1.1

ENV_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "CartPole-v1": (0, 500.0 * __t_margin),
    "Acrobot-v1": (-500.0, -80 * (2 - __t_margin)),
    "LunarLander-v3": (None, 220.0),
    "HalfCheetah-v5": (-450, 600),
}

FULL_EXPERIMENT_FILE = Path("experiment_full_data.parquet")

TEST = 0

nan: Final = np.nan


Ts = TypeVarTuple("Ts")

# Experiment output directory structure is - if sorted in groups else one level higher

# Project/group/Name*run_id/


logger = logging.getLogger(__name__)


def _cast_int_or_nan(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return float("nan")


def try_literal_eval(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _try_cast(v: Any):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    if v.is_integer():
        return int(v)
    return v


def ifill(
    *cols: Unpack[Ts], n: int
) -> tuple[Unpack[Ts]] | tuple[Unpack[Ts], Sentinel] | tuple[Unpack[Ts], Sentinel, Sentinel] | tuple[Sentinel, ...]:
    return (*cols, *(Placeholder,) * (n - len(cols)))


def get_runs_from_submission(
    submission_file: str | Path, *, status_filter: str = "SUCCEEDED"
) -> dict[str, SubmissionRun]:
    """
    Parse a submission YAML file and collect runs whose group name or status matches ``status_filter``.

    Args:
        submission_file: Path to the submission YAML file.
        status_filter: Filter string checked against group name and status (case-insensitive).

    Returns:
        A list of submission runs containing group, run key, run id, and status metadata.
    """
    path = Path(submission_file)
    if not path.exists():
        raise FileNotFoundError(f"Submission file {path} does not exist.")
    with path.open("r", encoding="utf-8") as handle:
        submission = yaml_load(handle)
    if not isinstance(submission, dict):
        raise ValueError(f"Submission file {path} must contain a mapping at the root.")
    runs: dict[str, SubmissionRun] = {}
    normalized_filter = status_filter.upper()
    submission_name = path.name.removesuffix(".yaml").removeprefix("submissions").removeprefix("submission").strip("_")

    for group_name, group_data in submission.items():
        if not isinstance(group_data, dict) or "run_ids" not in group_data:
            continue
        run_ids = group_data["run_ids"]
        if not isinstance(run_ids, dict):
            continue
        group_matches_filter = normalized_filter in group_name.upper()
        for run_key, run_entries in run_ids.items():
            if not isinstance(run_entries, dict):
                continue
            normalized_run_key = run_key.strip()
            if normalized_run_key.startswith("(") and normalized_run_key.endswith(")"):
                normalized_run_key = normalized_run_key[1:-1].strip()
            for run_id, run_info in run_entries.items():
                if not isinstance(run_info, dict):
                    continue
                if re.match(r"^.+?-v\d\)$", run_id):
                    continue
                status = str(run_info.get("status", "")).upper()
                submission_id = run_info.get("submission_id")
                if not (group_matches_filter or normalized_filter in status):
                    logger.info("Skipping run %s/%s with status %s", group_name, run_id, status)  # noqa: G004
                    continue
                assert run_id not in runs, f"Duplicate run_id {run_id} found in submission file."
                file_path = run_info.get("file_path")
                runs[run_id] = SubmissionRun(
                    group_name=group_name,
                    run_key=normalized_run_key,
                    run_id=run_id,
                    status=status,
                    submission_name=submission_name,
                    file_path=file_path,
                    submission_id=submission_id,
                )
    return runs


def find_run_directory(run_key: str, run_id: str, *, search_paths: Sequence[Path] | None = None) -> Path:
    """
    Locate the directory for the given run using the ``*run_key*/*/*run_id`` glob pattern.

    Args:
        run_key: Dataset/environment key of the run.
        run_id: Unique run identifier.
        search_paths: Paths to scan; defaults to ``PATHS``.

    Returns:
        The directory containing the run.

    Raises:
        FileNotFoundError: If no directory matches the pattern.
        ValueError: If multiple directories match the pattern.
    """
    base_paths = tuple(search_paths) if search_paths is not None else tuple(PATHS)
    pattern = f"*{run_key}*/*/*{run_id}"
    matches: list[Path] = []
    for base in base_paths:
        base_path = Path(base)
        if not base_path.exists():
            continue
        matches.extend(base_path.glob(pattern))
    unique_matches: list[Path] = list(dict.fromkeys(matches))
    if not unique_matches:
        raise FileNotFoundError(f"No run directory found for run_id={run_id} (run_key={run_key}).")
    if len(unique_matches) > 1:
        raise ValueError(f"Multiple run directories found for run_id={run_id} (run_key={run_key}): {unique_matches}")
    return unique_matches[0]


def get_run_directories_from_submission(
    submission_file: str | Path,
    *,
    status_filter: str = "SUCCEEDED",
    search_paths: Sequence[Path] | None = None,
    test: Literal[False] | int = False,
) -> Iterator[tuple[str, Path]]:
    """
    Yield directories for all runs matching ``status_filter`` in a submission file.

    Args:
        submission_file: Path to the submission YAML file.
        status_filter: Filter applied to group names and statuses.
        search_paths: Paths to scan for the runs; defaults to ``PATHS``.

    Yields:
        Tuples containing the run_id and the discovered directory.
    """
    run_infos = get_runs_from_submission(submission_file, status_filter=status_filter)
    collected = 0
    if not test:
        limit = float("inf")
    elif test is True:  # type: ignore[comparison-overlap]
        limit = 5
    else:
        limit = int(test)
    for run_id, info in run_infos.items():
        assert run_id == info["run_id"]
        if re.match(r"^.*-v\d$", run_id):
            continue

        # Check if file_path exists in cached data and is valid
        cached_path = info.get("file_path")
        if cached_path and Path(cached_path).exists():
            directory = Path(cached_path)
        else:
            # Find the directory and write back to YAML
            directory = find_run_directory(info["run_key"], run_id, search_paths=search_paths)

            # Write back the discovered path to the YAML file
            if write_back is not None:
                try:
                    # Construct job_id from group_name and run_key for write_back
                    job_id = f"{info['group_name']}_{info['run_key']}"
                    # Pass the file_path as part of the run_info dict
                    run_info_update = {
                        run_id: {
                            "status": info["status"],
                            "submission_id": info["submission_id"],
                            "file_path": str(directory),
                        }
                    }
                    write_back(
                        group=info["group_name"],
                        job_id=job_id,  # split into group_runkey
                        run_id=run_info_update,
                        file=submission_file,
                        overwrite=True,
                    )
                except Exception as e:
                    logger.warning("Failed to write back file_path for run %s: %s", run_id, e)

        yield run_id, directory
        collected += 1
        if collected >= limit:
            return


class _NoReprDict(dict):
    """Dict for faster debugging"""

    def __repr__(self) -> str:
        return "{...}"


class _NoReprList(list):
    """List for faster debugging"""

    def __repr__(self) -> str:
        return "{...}"


_NoReprList = cast("type[list]", _NoReprList)


def load_run_data(offline_run: str | Path, experiment_dir=None, *, use_cache=True) -> pd.DataFrame | dict[Any, Any]:
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
    # Check if already processed file exists:
    df = None
    if use_cache:
        if (offline_run / FULL_EXPERIMENT_FILE.with_suffix(".parquet")).exists():
            try:
                df = pd.read_parquet(offline_run / FULL_EXPERIMENT_FILE.with_suffix(".parquet"))
            except Exception as e:
                logger.error(  # noqa: G004
                    f"Failed to read parquet file for run {offline_run}: {e!r}. "
                    "This may be due to mixed data types in columns. "
                    "Please delete the parquet file to regenerate it."
                )
        if (offline_run / FULL_EXPERIMENT_FILE.with_suffix(".csv")).exists():
            df = pd.read_csv(
                offline_run / FULL_EXPERIMENT_FILE.with_suffix(".csv"), header=[0, 1, 2, 3], index_col=[0, 1]
            )
        if df is not None:
            logger.info("Loaded cached full experiment data for run %s from %s", offline_run, FULL_EXPERIMENT_FILE)
            return df
    result_files = offline_run.glob("*/result*.json")
    run_data = _NoReprDict()
    num_errors = 0
    for result_file in tqdm(result_files):
        if result_file.name.endswith(".parent.json"):
            continue
        try:
            df = pd.read_json(result_file, lines=True, dtype={"trial_id": str})
            # Drop unwanted columns if they exist
            # Flatten nested dict columns and convert to MultiIndex
            # ptimes = df[("config", "fork_from", "parent_time")]
            # mask = ~ptimes.isna()

            # df.loc[mask.values, ("config", "fork_from", "parent_time")] = ptimes[mask.values].map(str).values
            df = pd.json_normalize(df.to_dict(orient="records"), sep="/")
            # Convert columns to MultiIndex, replacing NaN with Placeholder
            tuples = [
                tuple(Placeholder if (isinstance(part, float) and np.isnan(part)) else part for part in col.split("/"))
                for col in df.columns
            ]
            if not tuples:
                logger.error(f"No columns found in result file {result_file}")
                continue
            longest_tuple_length = max(len(t) for t in tuples)
            tuples = [ifill(*tup, n=longest_tuple_length) for tup in tuples]
            multiindex = pd.MultiIndex.from_tuples(tuples)
            df.columns = multiindex
            df = df.sort_index(axis=1).drop(columns=DROP_COLUMNS, errors="ignore").sort_index(axis=1)
            # Drop levels that are only Placeholder
            levels_to_drop = [
                level for level in range(df.columns.nlevels) if all(df.columns.get_level_values(level) == Placeholder)
            ]
            if levels_to_drop:
                df.columns = df.columns.droplevel(level=levels_to_drop)
            if "timers" in df:
                remote_breakpoint()
            # df.columns = pd.MultiIndex.from_tuples(cast("list[tuple[str, ...]]", df.columns))
            try:
                experiment_key = df.config.experiment_key.iloc[-1]
                if not isinstance(experiment_key, str):
                    experiment_key = experiment_key.item()
                if result_file.name != "result.json" and experiment_key not in result_file.name:
                    # Due to a bug where orignal experiment was added to config.trial_id_history on continue
                    if experiment_key.endswith(ExperimentKey.RIGHT_PAD_CHAR):
                        # Fix history which should be broken too
                        # if main branch it is likely a bug and we can trust the result_file as experiment key
                        main_branch = df.iloc[-1].config.__pbt_main_branch__.item()
                        if "trial_id_history" in df.config:
                            history = df.iloc[-1].config.trial_id_history
                            history.index = history.index.get_level_values(0)  # remove all Placeholder
                            indices = history.index[history == experiment_key].map(_cast_int_or_nan)
                            idx = indices.max()
                            if idx > 0:
                                # Note: clean history allows experiment_key fix
                                assert (
                                    idx > 1
                                )  # as this only happens when a fork is continued the index should be at least 2
                                max_index = (
                                    history.index[history == experiment_key].map(_cast_int_or_nan).fillna(0).max()
                                )
                                if idx == max_index:
                                    # drop the column
                                    df = df.drop(
                                        [
                                            # Try int and string
                                            ("config", "trial_id_history", str(idx)),
                                            ("config", "trial_id_history", idx),
                                        ],
                                        axis=1,
                                        errors="ignore",
                                    )
                                else:
                                    # drop and shift the other columns
                                    histories = df.config.trial_id_history.sort_index(axis=1).copy()
                                    original_experiment_keys = histories.pop("original_experiment_keys")
                                    epoch_columns = histories.columns.get_level_values(0)
                                    col_type = type(epoch_columns[0])
                                    epoch_columns = epoch_columns.map(int)
                                    levels_before = histories.columns.nlevels
                                    histories.columns = pd.RangeIndex(epoch_columns.min(), epoch_columns.max() + 1)
                                    cols_after = histories[list(range(idx + 1, len(histories.columns)))]
                                    histories = histories.drop(histories.columns[-1], axis=1)
                                    histories[list(range(idx, len(histories.columns)))] = cols_after
                                    # Wirte back multiindex columns
                                    histories["original_experiment_key"] = original_experiment_keys
                                    histories.columns = pd.MultiIndex.from_product(
                                        [
                                            ["config"],
                                            ["trial_id_history"],
                                            list(map(col_type, histories.columns)),
                                            *[[Placeholder]] * (levels_before - 1),
                                        ]
                                    )
                                    df = pd.concat(
                                        [df.drop(("config", "trial_id_history"), axis=1), histories], axis=1
                                    ).sort_index(axis=1)
                            else:
                                # Its possibly that the initial trial ID is written without being put into the history, which is correct
                                latest_history_idx = history.index.map(_cast_int_or_nan).dropna().astype(int).max()
                                if latest_history_idx > 0 and history[latest_history_idx] in result_file.name:
                                    new_experiment_key = history[latest_history_idx]
                                else:
                                    # This can somehow happen if a wrong ID was written in a prior place
                                    remote_breakpoint()
                                    raise ValueError(
                                        f"Experiment key {experiment_key} does not match filename. Possibly the history is incomplete or due to a bug written wrongly."
                                    )
                        elif main_branch:
                            # no trial_id history but we are a main branch, believe this is a bug
                            new_experiment_key = result_file.name.split("-")[-1].split(".")[0]
                        else:
                            remote_breakpoint()
                            raise ValueError(
                                f"Experiment key {experiment_key}does not match filename. Possibly the history is incomplete."
                            )
                        experiment_key_from_file = result_file.name.split("-")[-1].split(".")[0]
                        if "trial_id_history" in df.config:
                            history = df.iloc[-1].config.trial_id_history.drop(
                                "original_experiment_key", errors="ignore"
                            )
                            index = history.index.get_level_values(0)
                            col_type = type(index[0])
                            max_entry_idx = index.map(int).max()
                            new_experiment_key = history[col_type(max_entry_idx)]
                            if not isinstance(new_experiment_key, str):
                                new_experiment_key = new_experiment_key.item()
                            assert experiment_key_from_file == new_experiment_key
                        # Cannot replace all occurances as it could be A, B, A
                        mask = df.config.experiment_key == experiment_key
                        # replace the last True block
                        last_true_idx = mask[::-1].idxmin().item()
                        mask = mask.to_numpy().flatten()
                        mask[:last_true_idx] = False
                        df.loc[mask, ("config", "experiment_key")] = new_experiment_key
                    else:
                        logger.error("error with experiment_key cannot restore")
                        remote_breakpoint()  # some other bug
                        raise ValueError(
                            f"Experiment key {experiment_key} does not match id in file name {result_file}"
                        )
                    experiment_key = df.config.experiment_key.iloc[-1]
                    if not isinstance(experiment_key, str):
                        experiment_key = experiment_key.item()
                    assert result_file.name == "result.json" or experiment_key in result_file.name, (
                        f"Experiment key {experiment_key} does does not match id in file name {result_file}"
                    )
            except AttributeError as ae:
                if "experiment_key" not in str(ae):
                    raise
                # Older versions without this field, take from filename
                if result_file.name == "result.json":
                    assert df.trial_id.nunique().item() == 1
                    experiment_key = df.iloc[0].trial_id.item()
                else:
                    experiment_key = result_file.stem.split("-")[-1]
                    if "fork_from" in df.config:
                        # can be nan if it is a continued fork and we did not include this info
                        last_non_nan: int | None = df.config.fork_from["fork_id_this_trial"].last_valid_index()
                        key_info_from_config = "Not fork_id_this_trial found"
                        if last_non_nan is not None:
                            key_info_from_config = df.iloc[last_non_nan].config.fork_from["fork_id_this_trial"].item()
                        assert last_non_nan is None or experiment_key == key_info_from_config, (
                            f"Experiment key {experiment_key} does not match fork_from key {key_info_from_config}"
                        )
                # experiment_key = df.config.experiment_id.values.item()+"_"+df.config.trial_id.values.item()
                if experiment_key in run_data:
                    remote_breakpoint()
                    raise ValueError(f"Duplicate experiment_key {experiment_key} already present in {offline_run}")  # noqa: B904
                if result_file.name != "result.json" and experiment_key not in result_file.name:
                    logger.error(f"Experiment key {experiment_key} does does not match id in file name {result_file}")
                    remote_breakpoint()
                    continue
        except Exception as e:  # noqa: PERF203
            if "patched.json" in str(result_file):
                continue
            num_errors += 1
            logger.error(f"Failed to load run data for {result_file}: {e!r}", exc_info=num_errors <= 2)  # noqa: G004
            if num_errors >= 2:
                logger.error("Too many errors encountered while loading run data; aborting further attempts.")
                raise
        else:
            if experiment_key in run_data:
                # how can this be, we check file names, then the first file must have? Backup or parent file
                logger.warning(
                    f"Duplicate experiment_key {experiment_key} found in {offline_run} for file {result_file}. "
                    "This may be due to a bug where continued trials incorrectly recorded the original experiment key. "
                    "Please verify the data integrity."
                )
                if experiment_key not in result_file.name:
                    # take this one
                    logger.error(
                        f"Not taking data from file {result_file} for experiment_key {experiment_key}. Filename does not match."
                    )
                    continue
                # this one seems correct
                # take the one that is longer
                present = run_data[experiment_key]
                if len(present) >= len(df):
                    logger.warning(
                        "Existing data is longer, keeping existing data for experiment_key %s", experiment_key
                    )
                    continue
                del present
            group_stat = _get_group_stat(df, "pbt_group_key")
            if group_stat == "grad_clip":
                # replace NaN with inf
                if group_stat in df:
                    df[group_stat] = df[group_stat].fillna(float("inf"))
                if group_stat in df.config:
                    df[ifill("config", group_stat, n=df.columns.nlevels)] = df[
                        ifill("config", group_stat, n=df.columns.nlevels)
                    ].fillna(float("inf"))

            run_data[experiment_key] = df

    logger.info(f"Loaded data for run {offline_run} with {len(run_data.keys())} experiments")  # noqa: G004
    return run_data


def save_run_data(offline_run: str | Path, data: pd.DataFrame, format: Literal["parquet", "csv"] = "parquet"):
    """Stores the combined dataframe of the experiment"""
    if isinstance(offline_run, str):
        offline_run = Path(offline_run)
    if offline_run.is_dir():
        offline_run = offline_run / FULL_EXPERIMENT_FILE.with_suffix(f".{format}")
    offline_run.parent.mkdir(parents=True, exist_ok=True)
    # check suffix:
    if format == "parquet" and offline_run.suffix != ".parquet":
        offline_run = offline_run.with_suffix(".parquet")
    elif format == "csv" and offline_run.suffix != ".csv":
        offline_run = offline_run.with_suffix(".csv")
    if format == "parquet":
        try:
            data.to_parquet(offline_run, index=True)
        except Exception as e:
            if "Conversion failed for column run_id" in str(e):
                # some might have an int there
                assert data.index.names == ["run_id", "training_iteration"]
                data.index = pd.MultiIndex.from_arrays(
                    [data.index.get_level_values(0).map(str), data.index.get_level_values(1)]
                )
                data.to_parquet(offline_run, index=True)
            else:
                raise
    elif format == "csv":
        data.to_csv(offline_run, index=True)


__logged_base62_value_error = False


def __base62_sort_key(s: str) -> tuple[int, int]:
    if not isinstance(s, str):
        return s, 0
    if s.endswith("Z") or "S" not in s:
        secondary = 0
        try:
            if "_" in s:
                # trial_00012
                secondary = int(s.rsplit("_", 1)[-1])
            elif "C" in s:
                _, part2 = s.split("C", 1)
                secondary = int(part2[:2])
        except ValueError:
            global __logged_base62_value_error  # noqa: PLW0603
            if not __logged_base62_value_error:
                logger.exception(f"Failed to parse secondary sort key for {s!r}")
                __logged_base62_value_error = True
        return 0, secondary
    # We should not need a secondary here as they should be already ordered.
    return base62.decode(s.split("S", 1)[-1]), 0


def _base62_sort_key(s: pd.Index):
    return s.map(__base62_sort_key)


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
        return df.drop(columns=["run_id"], errors="ignore")

    dfs: list[pd.DataFrame] = _NoReprList()
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
    if len(dfs) == 0:
        raise ValueError("No DataFrames to combine.")
    try:
        # If we tracked timers at some points the df levels might not match
        combined_df = pd.concat(
            cast("list[pd.DataFrame]", dfs), keys=dataframes.keys(), names=["run_id", "training_iteration"]
        ).sort_index(key=_base62_sort_key)
        # Replace NaN in MultiIndex with the Placeholder sentinel
        if isinstance(combined_df.columns, pd.MultiIndex):
            new_tuples = [
                tuple(Placeholder if (isinstance(v, float) and np.isnan(v)) else v for v in idx)
                for idx in combined_df.columns
            ]
            combined_df.columns = pd.MultiIndex.from_tuples(new_tuples, names=combined_df.columns.names)
    except Exception as e:  # noqa: PERF203
        logger.error(f"Failed to concatenate DataFrames: {e!r}", exc_info=True)  # noqa: G004
        breakpoint()
        raise
    shape_before = combined_df.shape
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
        combined_df = combined_df.sort_index(axis=1)
    logger.debug("Removing duplicates changed the shape from %s to %s", shape_before, combined_df.shape)

    # Reset only the outermost 'run_id' index, not all levels
    # combined_df = combined_df.reset_index(level=0)
    # Ensure 'run_id' is the first index level, and avoid duplicate index names
    # index_names = [n for n in combined_df.index.names if n != "run_id"]
    # combined_df = combined_df.set_index(["run_id", "training_iteration"], drop=True)
    # combined_df = combined_df.reorder_levels(["run_id", *index_names])
    depth = len(combined_df.columns[0])

    def ifill(*cols):
        return (*cols, *(Placeholder,) * (depth - len(cols)))

    combined_df, perturbation_interval = add_missing_pbt_epoch(combined_df)
    # if perturbation_inveral is None we possibly do not not have pbt_epoch entry which can cause a failure below
    combined_df = _drop_duplicate_steps(combined_df)
    continued_runs, combined_df = which_continued(combined_df)
    if continued_runs.empty:
        # likely because failed before first perturb
        combined_df.loc[:, ifill("config", "__pbt_main_branch__")] = False
        combined_df.loc[:, ifill("config", "pbt_epoch")] = 0
        # If it is a baseline / aborted early still add pbt epoch = 0
        return combined_df.sort_index(axis=1).copy()
    # if ("config", "__pbt_main_branch__") in combined_df and not combined_df[("config", "__pbt_main_branch__")].isna().any():
    #    # already present
    #    return combined_df.sort_index(axis=1)
    main_branch_mask = np.zeros_like(
        combined_df.index, dtype=bool
    )  # combined_df.index.get_level_values("run_id").isin(continued_runs.index)
    # BUT when the run a fork is no longer continued we may NOT set it to True in their last pbt_epoch
    last_pbt_epoch = combined_df[("config", "pbt_epoch")].groupby(level="run_id").max()
    for run_id in continued_runs.index:
        if run_id in last_pbt_epoch.index:
            # Set to true were we have a continued run but not where it has its max pbt_epoch
            run_mask = (combined_df.index.get_level_values("run_id") == run_id) & (
                combined_df.config.pbt_epoch != last_pbt_epoch.loc[run_id]
            ).to_numpy().flatten()
            main_branch_mask[run_mask] = True
        else:
            assert False

    assert (
        combined_df.groupby([ifill("config", "pbt_epoch")])[[ifill("config", "__pbt_main_branch__")]]
        .max()
        .iloc[:-1]
        .all()
        .item()
    ), "Some epoch is missing main branch True values"
    main_branch_mask = pd.Series(main_branch_mask, index=combined_df.index, name=ifill("config", "__pbt_main_branch__"))
    # try:
    #    combined_df.loc[:, ("config", "__pbt_main_branch__")] = main_branch_mask
    # except KeyError:
    #    combined_df.loc[:, ifill("config", "__pbt_main_branch__")] = main_branch_mask
    combined_df = pd.concat(
        [combined_df.drop(ifill("config", "__pbt_main_branch__"), axis=1, errors="ignore"), main_branch_mask], axis=1
    ).sort_index(axis=1)
    # This fixes a potential missing end step
    _get_epoch_end_steps(combined_df)
    return combined_df


def add_missing_pbt_epoch(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """Adds missing pbt_epoch column if not present based on fork_from parent_env_steps"""
    if "pbt_epoch" in df.config:
        return df, None
    if "fork_from" not in df.config:
        logger.warning("Cannot add pbt_epoch as no fork_from information is present.")
        unique_index = df.index.get_level_values("run_id").unique()
        if not unique_index.str.contains(ExperimentKey.FORK_SEPARATOR).any():
            # no forked runs
            indexer = ifill("config", "pbt_epoch", n=df.columns.nlevels)
            df = df.copy()
            df.loc[:, indexer] = 0
        return df, None
    depth = df.config.fork_from.columns.nlevels
    fork_from = df.config.fork_from.droplevel(axis=1, level=list(range(1, depth)))
    epoch_end_steps = np.nan_to_num(fork_from.parent_env_steps.unique()).astype(int)
    diffs = np.diff(epoch_end_steps)
    if (diffs[0] == diffs).all():
        perturbation_interval = diffs[0]
    elif (diffs[0] == diffs[:-1]).all():
        perturbation_interval = diffs[0]
    else:
        # There kan be rare bug that a parent overstepped and therefore not all are unique
        unique, counts = np.unique(diffs, return_counts=True)
        cleaner_diffs = unique[counts > 1]
        if len(cleaner_diffs) == 1:
            perturbation_interval = cleaner_diffs[0]
        else:
            perturbation_interval = None
            logger.warning(
                "Cannot make a good guess on perturbation interval from fork_from parent_env_steps %s", epoch_end_steps
            )
    if perturbation_interval is not None:
        # Estimate pbt_epochs
        df = df.copy()
        indexer = ifill("config", "pbt_epoch", n=df.columns.nlevels)
        pbt_epochs = (df["current_step"] - 1) // perturbation_interval
        # Fix large overstepping:
        if "fork_from" in df.config:
            perturbation_steps = df[ifill("config", "fork_from", "parent_env_steps", n=df.columns.nlevels)]
            epochs = perturbation_steps.fillna(0).unique() // perturbation_interval
            max_epoch = epochs.max() + 1
            pbt_epochs = pbt_epochs.astype(int).clip(upper=max_epoch)
        df.loc[:, indexer] = pbt_epochs.astype(int)
        df = df.sort_index(axis=1)
    # else cannot make a good guess on the real pbt epochs
    return df, perturbation_interval


def _which_continued_legacy(df: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.str_]]:
    # When we have no pbt_epoch we need other methods.
    try:
        depth = df.config.fork_from.columns.nlevels
    except AttributeError as ae:
        if "fork_from" in str(ae):
            # no fork_from at all - maybe aborted early, or was never forked
            return np.array([], dtype=str)
        raise
    fork_from = df.config.fork_from.droplevel(axis=1, level=list(range(1, depth)))
    parent_ids = fork_from.parent_fork_id.dropna().unique()
    return parent_ids


def _get_group_stat(df: pd.DataFrame | pd.Series, group_key: str | Hashable):
    if isinstance(df, pd.DataFrame):
        first_row = df.iloc[0]
    else:
        first_row = df
    try:
        return first_row.config[group_key].str.split("=").iloc[0][0]
    except KeyError:
        pass
    # If this failed then we have no group_by[-1] (pbt_group_key) column with key=value; other parts might then fail too
    # Possibly this run was never forked
    tune = first_row.config.cli_args.get("tune")
    if not tune:
        tune = first_row.config.experiment_group.values.item().split(":")[-1]
        if tune == "batch_size" and ("batch_size" not in df.config and "train_batch_size_per_learner" not in df.config):
            try:
                from experiments.create_tune_parameters import default_distributions

                tune_candidates = set(default_distributions.keys())
                tune_candidates.pop("minibatch_scale")
            except ImportError:
                tune_candidates = {
                    "lr",
                    "minibatch_size",
                    "num_envs_per_env_runner",
                    "accumulate_gradients_every",
                    "clip_param",
                    "vf_loss_coeff",
                    "gamma",
                    "grad_clip",
                    "entropy_coeff",
                }
            # experiment group can be helpful BUT batch_size was added to some that did NOT tune batch_size
            if isinstance(df.config, pd.Series):
                columns = set(df.config.index.get_level_values(0))
            else:
                columns = set(df.config.columns.get_level_values(0))
            candidates = columns & tune_candidates
            if candidates:
                if len(candidates) == 1:
                    return candidates.pop()
                logger.warning(f"Multiple possible group keys found {candidates}, using 'lr'")
                remote_breakpoint()
                return candidates.pop()
            logger.error("No suitable group key found - 'batch_size' is likely incorrect")
            remote_breakpoint()
            raise ValueError("No suitable group key detected. Is batch_size correct?")
        return tune
    return tune[0]


def which_continued(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Note for now we do not trust the __pbt_main_branch__
    """
    # there is no continued key we need to check which run_id (in the index) spans over multiple pbt_epochs
    df = df.copy()
    continued_legacy = None
    try:
        pbt_epochs = df[("config", "pbt_epoch")]
    except KeyError as ke:
        if "pbt_epoch" not in str(ke):
            raise
        try:
            df, perturbation_interval = add_missing_pbt_epoch(df)
            pbt_epochs = df[("config", "pbt_epoch")]
        except KeyError as e:
            if "pbt_epoch" not in str(e):
                # will only happen if fork_from is not in df.config
                logger.warning(
                    "pbt_epoch not found in df.config, likely also not df.config.fork_from. Assuming no forks."
                )
                return pd.DataFrame(index=[], columns=["continued"]), df
            # Note this also adds pbt_epoch
            continued_ids = _which_continued_legacy(df)
            # bring into expected format
            continued = pd.DataFrame(index=continued_ids, columns=["continued"])
            if continued.empty:
                if df.shape[0] != df.index.nunique():
                    logger.warning(
                        "Could not determine continued runs - no pbt_epoch and no fork_from parent_fork_id found."
                    )
                else:
                    # Do not return all, else main_only will not work
                    logger.info("Run without any forks")
                return continued, df
            continued_legacy = continued
    if "pbt_group_key" not in df.config:
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler  # noqa: PLC0415

        def map_group_key(row: pd.Series) -> str:
            groups = row.config.cli_args.get("tune")
            if not groups:
                # this might say batch_size but this could be wrong
                groups = [_get_group_stat(row, "pbt_group_key")]  # noqa: PD011
            mutations = dict.fromkeys(groups, None)

            ns = SimpleNamespace(_hyperparam_mutations=mutations)
            if groups and groups[0] not in row and groups[0] in row.config:
                row = row.config
            return GroupedTopPBTTrialScheduler._build_group_key_from_config(
                ns,
                row.droplevel(level=list(range(1, row.index.nlevels))),  # pyright: ignore[reportArgumentType]
            )

        group_keys = df.apply(map_group_key, axis=1)
        df.loc[:, ifill("config", "pbt_group_key", n=df.columns.nlevels)] = group_keys

    df.sort_index(axis=1, inplace=True)
    # Ignore "training_iteration" in the index by resetting it if present
    idx_names = list(df.index.names)
    # if "training_iteration" in idx_names:
    #    pbt_epochs = pbt_epochs.reset_index("training_iteration", drop=True)
    epoch_counts = pbt_epochs.groupby(level="run_id").nunique()

    # Method 1: TODO: Maybe not correct for older runs that do not set it on perturb
    continued = epoch_counts[(epoch_counts > 1).values]  # type: ignore  # noqa: PD011
    continued.columns = ["continued"]
    # Method 2
    if continued_legacy is not None:
        if set(continued_legacy.index) != set(continued.index):
            logger.warning("A: Discrepancy between continued runs detected between methods.")
            remote_breakpoint()
        if (continued_legacy != continued).any().item():
            logger.warning("Discrepancy between continued runs detected between methods.")
            remote_breakpoint()
    if "__pbt_main_branch__" in df.config:
        main_branch_info = (
            df[(df.config.__pbt_main_branch__).fillna(False).values].index.get_level_values("run_id").unique()
        )
        if (
            not set(continued.index) == set(main_branch_info)
            and not (df.config.__pbt_main_branch__).fillna(False).any().item()
        ):
            logger.warning("Continued runs do not match main branch runs.")
            remote_breakpoint()
    return continued, df


def _connect_groups(last: pd.DataFrame, now: pd.DataFrame, stat_value: str | None) -> pd.DataFrame:
    last_entries = last.xs(
        last.index.get_level_values("training_iteration").max(),
        level=1,
        drop_level=False,
        axis=0,
    )
    # Add immediate connection to start value
    if stat_value is None:
        return pd.concat(
            [
                last_entries,
                now,
            ],
            axis=0,
        )
    new_value = now.iloc[0][stat_value]
    last_entries2 = last_entries.copy()
    last_entries2["current_step"] += 128
    last_entries2[stat_value] = new_value
    return pd.concat(
        [
            last_entries,
            last_entries2,
            now,
        ],
        axis=0,
    )


def check_metric_backport(df: pd.DataFrame, metric: str | tuple[str, ...]) -> str | tuple[str, ...]:
    if metric == "episode_reward_mean" and (metric not in df or df[metric].isna().all().item()):
        from ray_utilities.constants import EPISODE_RETURN_MEAN, EPISODE_RETURN_MEAN_EMA  # noqa: PLC0415

        # introduced it later
        if EPISODE_RETURN_MEAN_EMA in df.evaluation and not df.evaluation[EPISODE_RETURN_MEAN_EMA].isna().all().item():
            backport_metric = ("evaluation", EPISODE_RETURN_MEAN_EMA)
        else:
            backport_metric = ("evaluation", EPISODE_RETURN_MEAN)
        logger.info("Metric 'episode_reward_mean' was not found in the data, changing metric to: %s", backport_metric)
        return backport_metric
    return metric


def _get_epoch_end_steps(df) -> pd.DataFrame | None:
    def ifill(*cols, n=df.columns.nlevels):
        return (*cols, *(Placeholder,) * (n - len(cols)))

    main_branch_data = df[df[ifill("config", "__pbt_main_branch__")]]
    try:
        epoch_grouper = main_branch_data.groupby([ifill("config", "pbt_epoch")])["current_step"]
    except KeyError as ke:
        if main_branch_data.empty:
            logger.error("No main branch data found for calculating hyperparam metrics.")
            return None
        remote_breakpoint()
        raise
    # Does not contain last epoch as there is no main branch
    epoch_end_steps = epoch_grouper.last().current_step.astype(int).copy()
    expected_length = epoch_end_steps.index.max() - epoch_end_steps.index.min() + 1
    if pd.isna(expected_length) and len(epoch_end_steps) == 0:
        # When we have a baseline there is no max
        return df
    if len(epoch_end_steps) != expected_length:
        logger.warning(
            "Some pbt_epochs are missing __pbt_main_branch__ likely not correct: expected length %s but got %s",
            expected_length,
            len(epoch_end_steps),
        )
        missing = set(epoch_end_steps.index) ^ set(
            range(int(epoch_end_steps.index.min()), int(epoch_end_steps.index.max()) + 1)
        )
        # Use heuristic of most count
        candidates, counts = np.unique(epoch_end_steps.diff().dropna().values, return_counts=True)
        perturbation_interval = candidates[np.argmax(counts)]
        for miss in sorted(missing):
            epoch_end_steps.loc[miss] = epoch_end_steps.loc[miss - 1] + perturbation_interval
            try:
                epoch_mask = df.config.pbt_epoch == miss
                if not epoch_mask.any().item():
                    epoch_mask = (perturbation_interval * (miss) < df.current_step) & (
                        df.current_step <= perturbation_interval * (miss + 1)
                    )
                    df.loc[epoch_mask.to_numpy().flatten(), ifill("config", "pbt_epoch")] = miss
                missing_epoch = df[epoch_mask.values]
                run_ids_in_missing = missing_epoch.index.get_level_values("run_id").unique()
                # check that they are a parent in the next epoch
                next_epoch = df[(df.config.pbt_epoch == miss + 1).values]
                parents_next_epoch = next_epoch[ifill("config", "fork_from", "parent_fork_id")].dropna().unique()
                if set(parents_next_epoch) & set(run_ids_in_missing) == set(parents_next_epoch):
                    # set main branch in miss; however likely we wont get there else it would already be correct
                    missing_epoch.loc[parents_next_epoch, ifill("config", "__pbt_main_branch__")] = True
                    main_branch_info = df[ifill("config", "__pbt_main_branch__")].copy()
                    main_branch_info[
                        (df.config.pbt_epoch == miss).to_numpy().flatten()
                        & df.index.get_level_values("run_id").isin(parents_next_epoch)  # ,
                        # ifill("config", "__pbt_main_branch__"),  # this is a series
                    ] = True
                    df[ifill("config", "__pbt_main_branch__")] = main_branch_info
                else:
                    logger.warning(
                        "Could not fix __pbt_main_branch__ for missing epoch %s "
                        "as parents could not determined but could determine epoch end steps. "
                        "Parents next epoch: %s, runs in missing epoch: %s",
                        miss,
                        parents_next_epoch,
                        run_ids_in_missing,
                        stacklevel=2,
                    )
            except Exception:
                logger.exception("Could not fix __pbt_main_branch__ for missing epoch %s", miss)
                remote_breakpoint(5679)

        epoch_end_steps = epoch_end_steps.sort_index()
        # Check which step is missing and fill it in
        # remote_breakpoint()
        # FIXME:
        # df and main_branch_data are wrong at miss steps
    epoch_end_steps.index.names = ["pbt_epoch"]
    epoch_end_steps.columns = ["current_step"]
    # Find the ending position. This might be wrong as some trials have been trained for longer accidentially => use harded max step, but it should align with the last_step + perturb interval
    max_epoch = epoch_end_steps.index.max()
    try:
        if pd.isna(max_epoch) and epoch_end_steps.empty:
            max_epoch = -1
        epoch_end_steps.loc[max_epoch + 1] = int(
            min(
                df.current_step.max().item(),
                (epoch_end_steps.loc[max_epoch] + epoch_end_steps.diff().iloc[-1]).iloc[-1],
                1_179_648,
            )
        )
    except (KeyError, ValueError) as e:
        if max_epoch != -1 or isinstance(e, ValueError):
            # A value error happens if we cast int to nan
            remote_breakpoint()
        epoch_end_steps.loc[max_epoch + 1] = int(
            min(
                df.current_step.max().item(),
                1_179_648,
            )
        )
    return epoch_end_steps.drop_duplicates(keep="first")


def get_epoch_stats(
    df, metric, *, individual_runs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    # region Variance
    # backup = df
    def ifill(*cols, n=df.columns.nlevels):
        return (*cols, *(Placeholder,) * (n - len(cols)))

    epoch_end_steps = _get_epoch_end_steps(df)
    main_branch_data = df[df[ifill("config", "__pbt_main_branch__")]]
    if main_branch_data.empty:
        return None, None, (None, None)
    metric_key = metric if isinstance(metric, str) else "-".join(metric)
    # NOTE: main_branch data is not backfilled here!
    main_branch_data = main_branch_data.set_index(ifill("current_step"), append=True)
    main_branch_data.index.names = [*main_branch_data.index.names[:-1], "current_step"]
    df = df.set_index(
        [ifill("config", "pbt_epoch"), ifill("config", "pbt_group_key"), ifill("current_step")], append=True
    )
    df.index.names = [*df.index.names[:-3], "pbt_epoch", "pbt_group_key", "current_step"]
    if not individual_runs:
        last_epoch_values = (
            # Exclude final step from list as not pinned on main branch
            df.loc[pd.IndexSlice[:, :, :, :, epoch_end_steps.T.values.tolist()[0]]]
            .groupby(["pbt_epoch"])[metric if isinstance(metric, str) else [ifill(*metric)]]
            .agg(["mean", "std"])
        )
    else:
        last_epoch_values = (
            # Exclude final step from list as not pinned on main branch
            df.loc[pd.IndexSlice[:, :, :, :, epoch_end_steps.T.values.tolist()[0]]]
            .groupby(["pbt_epoch", ifill("config", "experiment_key")])[
                metric if isinstance(metric, str) else [ifill(*metric)]
            ]
            # std will be NaN here
            .agg(["mean", "std"])
        )
    last_epoch_values.columns = [metric_key, "metric_std"]
    last_epoch_values.index.names = ["pbt_epoch"] if not individual_runs else ["pbt_epoch", "experiment_key"]
    last_epoch_values["current_step"] = last_epoch_values.index.get_level_values("pbt_epoch").map(
        epoch_end_steps.current_step
    )
    if not individual_runs:
        # we want to add this as -1; but possibly the minium is 0?
        if last_epoch_values.index.min().item() != 0:
            remote_breakpoint()
        last_epoch_values.loc[last_epoch_values.index.min() - 1] = 0
    else:
        new_head = pd.DataFrame(
            [[0] * len(last_epoch_values.columns)]
            * len(last_epoch_values.loc[0].index.get_level_values("experiment_key")),
            index=pd.MultiIndex.from_product(
                [
                    [last_epoch_values.index.min() - 1],
                    last_epoch_values.loc[0].index.get_level_values("experiment_key"),
                ],
                names=last_epoch_values.index.names,
            ),
        )
        new_head.columns = last_epoch_values.columns
        last_epoch_values = pd.concat([new_head, last_epoch_values], axis=0)
        last_epoch_values = last_epoch_values.sort_index()

    # df.groupby([pd.Grouper(level="current_step"), pd.Grouper(level="pbt_group_key"), "pbt_epoch"])[
    #    [ifill(metric, n=4), ifill("config", "pbt_epoch", n=4)]
    # ]
    group_by = ["current_step", "pbt_group_key", "pbt_epoch"] if not individual_runs else ["current_step", "pbt_epoch"]
    if not individual_runs:
        metric_values = (
            df.groupby(level=group_by)[metric if isinstance(metric, str) else [ifill(*metric)]]
            .agg(["mean", "std"])
            .droplevel(level=list(range(1, df.columns.nlevels)), axis=1)
        )
    else:
        metric_values = (
            df.groupby(["run_id", *group_by])[metric if isinstance(metric, str) else [ifill(*metric)]]
            .agg(["mean", "std"])
            .droplevel(level=list(range(1, df.columns.nlevels)), axis=1)
        )
    metric_values.columns = [metric_key, "metric_std"]
    # Shift by for for subtracting initial value
    value_shifter = last_epoch_values.copy()
    if not isinstance(value_shifter.index, pd.MultiIndex):
        value_shifter.index = last_epoch_values.index + 1
    else:
        assert "pbt_epoch" in value_shifter.index.names
        value_shifter.index = pd.MultiIndex.from_tuples(
            [
                tuple(
                    idx + 1 if idx_name == "pbt_epoch" else idx
                    for idx_name, idx in zip(value_shifter.index.names, idx_tuple)
                )
                for idx_tuple in value_shifter.index.to_list()
            ],
            names=value_shifter.index.names,
        )
    value_shifter_with_first = value_shifter.drop("current_step", axis=1).copy()
    try:
        if not individual_runs:
            metric_agg = (
                metric_values[metric_values.index.get_level_values("pbt_epoch") == 0]
                .groupby(["pbt_epoch", "current_step"])[metric_key]
                .agg(["mean", "std"])
            )
            first_loc = metric_agg.isna().idxmin()["mean"]
            # if we tune batch size the first entry weill be 0
            value_shifter_with_first.loc[0] = metric_agg.loc[first_loc]["mean"]
        else:
            metric_agg = (
                metric_values[metric_values.index.get_level_values("pbt_epoch") == 0]
                .groupby(["pbt_epoch", "run_id", "current_step"])[metric_key]
                .agg(["mean", "std"])  # nothing to aggregate
            )
            # Shift everyone by mean or shift every individually to 0?
            # Problem with individually is if there are large batch sizes.
            for idx in last_epoch_values.index[last_epoch_values.index.get_level_values("pbt_epoch") == 0]:
                first_loc = metric_agg.loc[idx].isna().idxmin()["mean"]
                value_shifter_with_first.loc[idx, :] = metric_agg.loc[(*idx, first_loc)].values
    except (ValueError, KeyError):
        remote_breakpoint()

    return metric_values, epoch_end_steps, (value_shifter.sort_index(), value_shifter_with_first.sort_index())


def calculate_hyperparam_metrics(
    df: pd.DataFrame, metric: str | tuple[str, ...], *, per_epoch: bool = False
) -> tuple[pd.DataFrame, str, tuple[pd.DataFrame, pd.DataFrame]] | None:
    """
    Calculate additional metrics for the DataFrame.
    This calculates:
        - variance between data grouped by pbt_group_key
        - Gini coefficient between data grouped by pbt_group_key
        - Kendal Tau metric at the end of each pbt_epoch between data grouped by pbt_group_key

    Args:
        df: DataFrame containing the experiment data.
        metric: Metric to calculate the statistics for.
        per_epoch: Whether to calculate metrics per epoch or per step.
    """

    def ifill(*cols, n=df.columns.nlevels):
        return (*cols, *(Placeholder,) * (n - len(cols)))

    metric_key = metric if isinstance(metric, str) else "-".join(metric)
    metric_values, epoch_end_steps, (value_shifter, value_shifter_with_first) = get_epoch_stats(df, metric)
    # Subtract initial mean from each group to get variance around start value
    centered_metric_values = metric_values.sub(value_shifter[metric_key].to_frame(), level="pbt_epoch")[
        metric_key
    ].dropna()
    # Rough start value for all groups; not good when we tune batch_size
    normed_metric_values = metric_values.divide(value_shifter_with_first.replace(0, nan), level="pbt_epoch")[
        metric_key
    ].dropna()
    # Also center the first epoch
    centered_metric_values2 = metric_values.sub(value_shifter_with_first, level="pbt_epoch")[metric_key].dropna()

    groupby = "current_step" if not per_epoch else ["current_step", "pbt_epoch"]

    centered_metrics = centered_metric_values.groupby(level=groupby).aggregate(["var", "std", "mean"])
    centered_metrics2 = centered_metric_values2.groupby(level=groupby).aggregate(["var", "std", "mean"])
    # For mean we need to subtract mean of pbt_epoch 0
    normed_metrics = normed_metric_values.groupby(level=groupby).aggregate(["var", "std", "mean"])

    # Normed metrics 2: Normalize: subtract mean and divide by stddev of epoch begin from value_shifter_with_first
    normalized_metric_values = (
        metric_values[metric_key]
        .sub(value_shifter_with_first[metric_key], level="pbt_epoch")
        .div(value_shifter_with_first["metric_std"], level="pbt_epoch")
    ).dropna()
    normalized_metric = normalized_metric_values.groupby(level=groupby).aggregate(["var", "std", "mean"])
    # endregion

    # region Gini
    # Calculate the gini coefficient for each point in time
    def calc_gini(x: pd.Series | pd.DataFrame, metric=metric_key) -> float:
        # Gini coefficient calculation
        if x.empty:
            return nan
        if isinstance(x, pd.DataFrame):
            x = x[metric]
        sorted_x = x.sort_values() if isinstance(x, pd.Series) else x.sort_values(metric)
        n = len(x)
        cumulative_x = sorted_x.cumsum()
        gini_coeff = (2 * (np.arange(1, n + 1) * sorted_x).sum()) / (n * cumulative_x.iloc[-1]) - (n + 1) / n
        return gini_coeff

    # TODO: Want to know how robust a configuration is over time, stable increase vs volatile
    # How to do this?
    # In metric values we have the std over each group key
    # metric_values.groupby(["pbt_group_key", "pbt_epoch"])["metric_std"].mean().groupby("pbt_group_key").mean()

    # TODO: or do group by key and get gini within the epoch for each group -> how does group differ over epoch?
    # metric_values.groupby(["pbt_epoch", "pbt_group_key"] if per_epoch else "current_step")[metric_key].agg(calc_gini)
    gini_metrics = metric_values.groupby(["current_step", "pbt_epoch"] if per_epoch else "current_step")[
        metric_key
    ].agg(calc_gini)
    # Clip gini_metrics to 99% of the data to reduce the effect of outliers

    # Kendallal Tau
    from scipy.stats import kendalltau

    try:
        B: pd.Series = metric_values.loc[epoch_end_steps.current_step, metric_key].drop(epoch_end_steps.iloc[-1])
        A: pd.Series = metric_values.loc[epoch_end_steps.current_step, metric_key].drop(epoch_end_steps.iloc[0])
        try:
            B.index = A.index
        except ValueError:
            # This wont work if some runs are dropped, need a better matcher
            try:
                max_step_B = B.index.get_level_values("current_step").max()
                max_step_A = A.index.get_level_values("current_step").max()
                keys_B = B[B.index.get_level_values("current_step") == max_step_B].index.get_level_values(
                    "pbt_group_key"
                )
                keys_A = A[A.index.get_level_values("current_step") == max_step_A].index.get_level_values(
                    "pbt_group_key"
                )
                not_in_A = keys_B.difference(keys_A)
                if not_in_A.size == 1:
                    B.drop((max_step_B, not_in_A.item()), inplace=True, axis=0)
                else:
                    B = B.drop([(max_step_B, a) for a in not_in_A.to_numpy()], axis=0)
                B.index = A.index
                AB = pd.concat([A, B], axis=1, names=[metric_key, "rank2"])
                # try to throw away those in the last epoch that are not present in A, ignore the step
            except Exception:
                logger.error("Failed to align A and B for kendall tau calculation")
                B: pd.Series = metric_values.loc[epoch_end_steps.current_step, metric_key].drop(
                    epoch_end_steps.iloc[-1]
                )
                A: pd.Series = metric_values.loc[epoch_end_steps.current_step, metric_key].drop(epoch_end_steps.iloc[0])
                # Slicing without accounting for level can disrupt alignment
                # Fill A with NaN for missing keys in B, cast steps to categorical to align
                B.name = "rank2"
                AB = pd.concat([A.reset_index(), B.reset_index()], axis=1, names=[metric_key, "rank2"])
                AB = AB.loc[:, ~AB.columns.duplicated(keep="first")]
                AB = AB.set_index(metric_values.index.names)
                logger.info("Found a reduced way to calculate kendall tau")
        else:
            # ignore this error for short aborted runs, kendall will be NaN then
            AB = pd.concat([A, B], axis=1, names=[metric_key, "rank2"])
    except Exception:
        remote_breakpoint(5679)
    AB.columns = [metric_key, "rank2"]
    # tau ~0 no consistent ordering, ~1 consistent ordering
    # high p value, no evidence of monotonic relationship. Tau is compatible with random change
    kendall_metrics = (
        AB.groupby("current_step")
        .rank()
        .groupby("current_step")
        .apply(lambda x: pd.Series(kendalltau(x[metric_key], x["rank2"]), index=["tau", "pvalue"]))
    )
    if per_epoch:
        # if we want epoch in the index and not the current_step:
        # step_to_epoch = epoch_end_steps.reset_index().set_index("current_step")["pbt_epoch"].to_dict()
        # kendall_metrics.index = kendall_metrics.index.map(step_to_epoch)
        gini_metrics = gini_metrics.groupby("pbt_epoch").mean()
        centered_metrics = centered_metrics.groupby("pbt_epoch").mean()
        centered_metrics2 = centered_metrics2.groupby("pbt_epoch").mean()
        normed_metrics = normed_metrics.groupby("pbt_epoch").mean()
        normalized_metric = normalized_metric.groupby("pbt_epoch").mean()
        # TODO: However this does not tell us how robust a configuration is over time
        # Cast epochs to step
        epoch_to_steps = epoch_end_steps["current_step"].to_dict()
        try:
            for metrics_df in (gini_metrics, centered_metrics, centered_metrics2, normed_metrics, normalized_metric):
                metrics_df.index = metrics_df.index.map(epoch_to_steps)
        except ValueError:
            remote_breakpoint(5680)
    try:
        intra_group_variance = df.groupby([ifill("config", "pbt_epoch"), ifill("config", "pbt_group_key")])[
            metric_key
        ].agg(["var", "std", "mean"])
    except (ValueError, KeyError) as e:
        logger.exception("Failed to calculate intra group variance.")
        remote_breakpoint(5680)
        raise
    intra_group_variance.index.names = ["pbt_epoch", "pbt_group_key"]
    intra_group_variance_global = df.groupby([ifill("config", "pbt_group_key")])[metric_key].agg(["var", "std", "mean"])
    intra_group_variance_global.index.names = ["pbt_group_key"]

    # AllA = metric_values.loc[metric_values.index.get_level_values("current_step") <= epoch_end_steps.iloc[-1].item(), metric].to_frame()
    # AllA["rank2"] = AllA.shift(len(AllA.loc[2048])*8)
    # kendall_metrics_all = AllA.groupby("current_step").rank().groupby("current_step").apply(lambda x: pd.Series(kendalltau(x[metric], x["rank2"]), index=["tau", "pvalue"]))
    # Add a MultiIndex column for each DataFrame/Series, using the variable name as the first level
    def to_multiindex(df, name):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if not isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([[name], df.columns])
        else:
            df = df.copy()
            df.columns = pd.MultiIndex.from_tuples(
                [(name, *col) if isinstance(col, tuple) else (name, col) for col in df.columns]
            )
        return df

    stats = pd.concat(
        [
            to_multiindex(
                centered_metrics.clip(
                    lower=centered_metrics.quantile(0.02) - 1, upper=centered_metrics.quantile(0.98) + 1, axis=1
                ),
                "centered_metrics",
            ),
            to_multiindex(
                centered_metrics2.clip(
                    lower=centered_metrics2.quantile(0.02) - 1, upper=centered_metrics2.quantile(0.98) + 1, axis=1
                ),
                "centered_metrics2",
            ),
            to_multiindex(
                normed_metrics.clip(
                    lower=normed_metrics.quantile(0.01) - 1, upper=normed_metrics.quantile(0.98) + 1, axis=1
                ),
                "normed_metrics",
            ),
            to_multiindex(
                normalized_metric.clip(
                    lower=normalized_metric.quantile(0.01) - 1, upper=normalized_metric.quantile(0.98) + 1, axis=1
                ),
                "normalized_metrics",
            ),
            to_multiindex(
                gini_metrics.clip(lower=gini_metrics.quantile(0.01) - 1, upper=gini_metrics.quantile(0.98) + 1),
                "gini_metrics",
            ),
            to_multiindex(kendall_metrics, "kendall_metrics"),
        ],
        axis=1,
    )
    # mean and std of the variance and metrics
    stats_reduced = stats.agg(["mean", "std"]).T.drop(["mean", "std"], axis=0, level=1).sort_index()
    stats_reduced = pd.concat(
        [
            stats_reduced.drop("normed_metrics"),
            to_multiindex(normed_metrics, "normed_metrics").agg(["mean", "std"]).T.drop("std", level=1),
        ],
        axis=0,
    )
    return stats, stats_reduced.to_string(), (intra_group_variance, intra_group_variance_global)


def bin_metric(values: pd.Series, bins: int = 10) -> pd.Series:
    """Bin a continuous metric into *bins* equal-width intervals."""
    if values.dropna().empty:
        return pd.Series(index=values.index, dtype="object")
    # pd.cut preserves the original index, which keeps time linkage intact
    binned = pd.cut(values, bins=bins, include_lowest=True, duplicates="drop")
    return binned.astype(str)


def _drop_duplicate_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for possibly duplicated steps and returns a cleaned version of the DataFrame.

    As pbt_epoch can be incorrect for restored and duplicated steps, this function
    determines the perturbation_interval and sets the pbt_epoch based on current_step
    """

    # When restored the pbt_epoch might be wrong and we might have duplicated steps, use first non-duplicated to get interval
    # use last to continue values
    def ifill(*cols, n=df.columns.nlevels):
        return (*cols, *(Placeholder,) * (n - len(cols)))

    df = df.sort_index(key=_base62_sort_key)
    try:
        first_change: tuple[str, int] | pd.Series = df[ifill("config", "pbt_epoch")].diff().iloc[1:].ne(0).idxmax()  # pyright: ignore[reportAssignmentType]
    except ValueError as ve:
        if "argmax of an empty sequence" not in str(ve):
            raise
        # no changes; likely early terminated experiment
        remote_breakpoint()
        raise
    except KeyError:
        # When this is a baseline run we do not have pbt_epoch. Other duplicate cleaning should take
        assert not df.index.duplicated().any()
        assert not df.reset_index("training_iteration").set_index(ifill("current_step"), append=True).duplicated().any()
        return df

    if isinstance(first_change, (pd.Series, pd.DataFrame)):
        # rare case can be a series
        first_change = first_change.item()
    try:
        perturbation_interval = df.current_step.loc[(first_change[0], first_change[1] - 1)].item()
    except AttributeError as ae:
        if "item" not in str(ae):
            raise
        return df  # should be when item() does not work but then there is no KeyError
    except KeyError:
        pass  # duplicated steps handle below
    else:
        df.attrs["perturbation_interval"] = perturbation_interval
        return df
    duplicated_steps = cast("pd.DataFrame | pd.Series", df.loc[first_change[0], "current_step"]).nunique() < len(
        df.current_step
    )
    try:
        duplicated_steps = bool(duplicated_steps)
    except ValueError:
        duplicated_steps = duplicated_steps.item()  # pyright: ignore[reportAttributeAccessIssue]
    if not duplicated_steps:
        raise
        # if we keep last the metrics will align better but we might not get the correct interval
    no_duplicates = df[~df.index.duplicated(keep="first")]
    nd_min_epoch_per_step = no_duplicates.groupby(ifill("current_step"))[[ifill("config", "pbt_epoch")]].min()
    # We cannot just look at the first change restored trials might be started with pbt_epoch 1
    loc_alt = (
        no_duplicates[ifill("current_step")]
        .map(nd_min_epoch_per_step[ifill("config", "pbt_epoch")])
        .diff()
        .iloc[1:]
        .ne(0)
        .idxmax()
    )
    try:
        perturbation_interval = int(no_duplicates.loc[(loc_alt[0], loc_alt[1] - 1)].current_step.iloc[0])  # item() ?
    except KeyError:
        # bad order so that fork comes first
        try:
            perturbation_interval = int(
                no_duplicates.loc[loc_alt].current_step
                - no_duplicates.loc[loc_alt]
                .get(
                    "batch_size",
                    no_duplicates.loc[loc_alt].config.get(
                        "train_batch_size_per_learner", no_duplicates.loc[loc_alt].config.get("batch_size")
                    ),
                )
                .iloc[0]
            )
        except TypeError:
            perturbation_interval = int(
                (
                    no_duplicates.loc[loc_alt].current_step
                    - no_duplicates.loc[loc_alt].get(
                        "batch_size",
                        no_duplicates.loc[loc_alt].config.get(
                            "train_batch_size_per_learner", no_duplicates.loc[loc_alt].config.get("batch_size")
                        ),
                    )
                ).iloc[0]
            )

    perturbation_interval_alt = None
    try:
        if "__pbt_main_branch__" in df.config:
            df[ifill("config", "__pbt_main_branch__")] = (
                df[ifill("config", "__pbt_main_branch__")].infer_objects(copy=False).fillna(False)
            )
            epoch_end_steps = _get_epoch_end_steps(df)
            unique, counts = np.unique(epoch_end_steps.diff().dropna().values, return_counts=True)
            perturbation_interval_alt = unique[np.argmax(counts)]
    except Exception:
        logger.exception("Could not get epoch end steps")
    else:
        if perturbation_interval_alt != perturbation_interval:
            logger.warning(
                "Disagreement between perturbation interval detection methods: %d vs %d",
                perturbation_interval,
                perturbation_interval_alt,
            )
            if perturbation_interval_alt in (32768, 147456):
                # common interval trust that one.
                perturbation_interval = perturbation_interval_alt
    if perturbation_interval > 200_000:
        logger.warning("Too large perturbation interval detected: %d", perturbation_interval)
        epoch_end_steps = _get_epoch_end_steps(df)
        remote_breakpoint()
        if perturbation_interval_alt is not None:
            perturbation_interval = perturbation_interval_alt
    df.attrs["perturbation_interval"] = perturbation_interval
    # TODO: Does pbt_epoch need to be fixed?
    # This disrupts seed, pbt_epoch
    df = df[~df.index.duplicated(keep="last")]
    # Need to fix pbt_epoch
    # Method 1 take min
    min_epoch_per_step = df.groupby(ifill("current_step"))[[ifill("config", "pbt_epoch")]].min()
    # Method 2 divide current step

    # Apply method 2
    method1 = ((df["current_step"] - 4) // perturbation_interval).values.flatten()
    # Apply method 1
    method2 = df[ifill("current_step")].map(min_epoch_per_step[ifill("config", "pbt_epoch")])
    if not (method1 == method2).all():
        # Method 1 can disagree when there is large overstepping
        logger.warning(
            "Disagreement between pbt_epoch fixing methods, using method 1. Overstepping: %s",
            (df.current_step.max() > perturbation_interval * 8).item(),
        )
        if (df.current_step.max() < perturbation_interval * 8).item():
            remote_breakpoint()
    df.loc[:, ifill("config", "pbt_epoch")] = method2.to_numpy()
    return df.sort_index(axis=1).copy()


def plot_n_save(
    df: pd.DataFrame,
    metrics: Sequence[str | tuple[str, ...]],
    plot_option: PlotOption,
    save_path: str | Path,
    experiment_keys: list[str] | None = None,
    group_stat: str | None = None,
    group_by=DEFAULT_GROUP_BY,
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
    close: bool = True,
    plot_errors: bool | Literal["only"] | str | Sequence[str] = True,
    plot_errors_type: Literal["box", "violin"] = "box",
    use_cached_figures: bool = False,
    **kwargs,
) -> list[Path] | None:
    # for metric in metrics: [metric]
    plot_errors = kwargs.pop("plot_errors", plot_errors)
    plot_errors_type = kwargs.pop("plot_errors_type", plot_errors_type)
    if "plot_error" in kwargs:
        plot_errors = kwargs.pop("plot_error")
    # cyclic import
    from ray_utilities.visualization.plot import plot_run_data  # noqa: PLC0415

    if (plot_option.main_only or plot_option.main_vs_rest or plot_option.main_vs_second_best) and not df.config[
        "__pbt_main_branch__"
    ].any().item():
        logger.warning("Cannot plot main branch only data as no main branch found in DataFrame.")
        return None
    save_path = Path(save_path)
    cached_fig = None
    if use_cached_figures and save_path.with_suffix(".pkl").exists():
        with open(save_path.with_suffix(".pkl"), "wb") as f:
            cached_fig = pickle.load(f)
    fig, error_figures = plot_run_data(
        df,
        metrics,
        experiment_keys,
        figsize,
        group_stat,
        group_by,
        plot_option=plot_option,
        log=log,
        show=False,
        plot_errors=plot_errors,
        plot_errors_type=plot_errors_type,
        fig=cached_fig,
        **kwargs,
    )
    if fig is None:  # pyright: ignore[reportUnnecessaryComparison]
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not save_path.is_dir():
        # check extension
        assert save_path.suffix == f".{format}", (
            f"File {save_path} already exists with different format "
            f"({save_path.suffix} != .{format}). Change the name or format,"
        )
    fig.savefig(save_path, format=format, bbox_inches="tight")
    # pickle figure
    with open(save_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(fig, f)
    if close:
        plt.close(fig)
    outfiles: list[Path] = [save_path]
    if error_figures:
        for metric_name, error_fig in error_figures.items():
            error_path = save_path.with_name(
                f"{save_path.stem}_{metric_name.replace('/', '_') if metric_name not in save_path.stem else ''}_errors{save_path.suffix}"
            )
            error_fig.savefig(error_path, format=format, bbox_inches="tight")
            outfiles.append(error_path)
            if close:
                plt.close(error_fig)
            print(f"Saved error plot to {error_path}")
    logger.info(f"Saved plot to {save_path}")  # noqa: G004
    return outfiles


def _join_nested(m, on="_"):
    if isinstance(m, tuple):
        return on.join(m)
    return m


def calculate_experiment_stats(
    combined_df: pd.DataFrame,
    metrics: Sequence[str | tuple[str, ...]],
    out_path: Path,
    format="pdf",
    *,
    plot_option: PlotOption,
    large: bool,
) -> list[Path]:
    if not out_path.is_dir():
        # Single file names are normally to convoluted with extra keys
        base_dir = out_path.parent
        path_base = base_dir.stem
    else:
        path_base = out_path.stem
    outfiles: list[Path] = []
    for per_epoch in (True, False):
        per_epoch_str = "-per_epoch" if per_epoch else ""
        for metric in metrics:
            metric = check_metric_backport(combined_df, metric)  # noqa: PLW2901
            metric_str = _join_nested(metric)
            if metric_str not in path_base and _join_nested(metric, on="-") not in path_base:
                metric_str = "-" + metric_str
            else:
                metric_str = ""
            # if per_epoch:
            #    remote_breakpoint()
            calculated_stats = calculate_hyperparam_metrics(combined_df, metric, per_epoch=per_epoch)
            if calculated_stats is None:
                continue
            stats, stats_reduced, (group_variance_per_epoch, group_variance_global) = calculated_stats

            # 1. Save reduced stats
            large_str = "-large" if large else ""
            out_path_name = out_path.stem
            stats_reduced_path = out_path.with_name(
                f"{out_path_name}{metric_str}{large_str}-hyperparam_stats{per_epoch_str}_reduced.txt"
            )
            with open(stats_reduced_path, "w") as f:
                f.write(stats_reduced)
            outfiles.append(stats_reduced_path)
            if per_epoch:
                pd.concat([group_variance_per_epoch, group_variance_global]).to_csv(
                    out_path.with_name(f"{out_path_name}{metric_str}{large_str}-hyperparam_variance_per_epoch.txt"),
                    index=True,
                )
                from ray_utilities.visualization.plot import plot_intra_group_variances  # cyclic  # noqa: PLC0415

                var_files = plot_intra_group_variances(
                    group_variance_global,
                    group_variance_per_epoch,
                    out_path,
                    out_path_name,
                    metric_str,
                    per_epoch_str,
                    format=format,
                    plot_option=plot_option,
                    large=large,
                )
                outfiles.extend(var_files)
            else:
                from ray_utilities.visualization import plot as _plot  # will set theme; cyclic import  # noqa: F401

            logger.info("Saved reduced hyperparameter stats to '%s'", stats_reduced_path)
            # 2nd plot the stats
            stats_path = out_path.with_name(
                f"{out_path_name}{metric_str}{large_str}_hyperparam_stats{per_epoch_str}.{format}"
            )
            stats_to_plot = stats.drop(
                [
                    c
                    for c in stats.columns
                    if "var" not in c[-1]
                    and "gini" not in c[0]
                    and "kendall" not in c[0]
                    and "normed_metrics" not in c[0]
                ],
                axis=1,
            )
            fig, axes = plt.subplots(nrows=len(stats_to_plot.columns), sharex=True)
            for col, ax in zip(stats_to_plot.columns, axes.flatten(), strict=True):
                stats_to_plot[col].sort_index().dropna().plot(
                    ax=ax,
                    kind="line",
                    legend=False,
                    fontsize=11,
                )
            columns_ax: "Axes"
            fig.set_size_inches(10, 1.25 * len(stats_to_plot.columns))
            for i, (columns_ax, column) in enumerate(zip(axes.flatten(), stats_to_plot.columns, strict=True)):
                if i != len(stats_to_plot.columns) - 1:
                    columns_ax.set_xlabel("")
                    columns_ax.set_xticks([])
                    columns_ax.set_xticklabels([])
                if plot_option.title:
                    columns_ax.set_title(column, fontsize=11, fontweight="light")
                if "kendall" in column[0]:
                    stats_to_plot[column].dropna().plot(ax=columns_ax, label=column)
            # for stat in stats.columns.get_level_values(0).unique():
            #    stats[stat].plot(ax=ax, label=stat)
            fig.savefig(stats_path, format=format, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved hyperparameter stats to '%s'", stats_path)
            stats_csv_paths = stats_path.with_suffix(".csv")
            stats.to_csv(stats_csv_paths, index=True)
            outfiles.extend([stats_path, stats_csv_paths])
    return outfiles


def get_and_check_group_stat(
    df, group_stat: str | None = None, group_by: Sequence[str] = DEFAULT_GROUP_BY
) -> tuple[str, pd.DataFrame]:
    if group_stat is None:
        # assumes a name=... format
        group_stat = _get_group_stat(df, group_by[-1])
        assert group_stat is not None
    if group_stat == "train_batch_size_per_learner" and "train_batch_size_per_learner" not in df:
        group_stat = "batch_size"
    if group_stat == "batch_size" and "batch_size" not in df:
        df = df.assign(batch_size=df.config.train_batch_size_per_learner)
    elif group_stat not in df:
        try:
            df = df.assign(**{group_stat: df.config[group_stat]})
        except TypeError:
            df = df.assign(**{group_stat: df.config[group_stat].values})
        except KeyError:
            remote_breakpoint()
            from_cli_args = df.config.cli_args[group_stat]
            if from_cli_args.empty or (from_cli_args.iloc[0] == from_cli_args).all():
                # wrong group_stat key
                raise
            df = df.assign(**{group_stat: df.config.cli_args[group_stat].values})
            raise
    if group_stat not in df.config and group_stat in df:
        stat_column = df[group_stat].to_frame() if isinstance(df[group_stat], pd.Series) else df[group_stat]
        stat_column.columns = pd.MultiIndex.from_tuples([ifill("config", group_stat, n=df.columns.nlevels)])
        df = pd.concat([df, stat_column], axis=1)
        # df.loc[:, ifill("config", group_stat)] = df[group_stat]
    if (df[group_stat] == df[group_stat].iloc[0]).all().item():
        logger.warning("Group stat '%s' has the same value for all runs.", group_stat)
        # mabye an old run with repeated runs per metric
    return group_stat, df.sort_index(axis=1)


def export_run_data(
    experiment_path: str | Path | pd.DataFrame,
    plot_option: PlotOption,
    experiment_keys: list[str] | None = None,
    metrics: Sequence[str | tuple[str, ...]] = ("episode_reward_mean",),
    group_stat: str | None = None,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
    save_path: str | Path | None = None,
    calc_stats: bool = True,
    use_cache: bool = True,
    **kwargs,
) -> list[Path] | None:
    """
    TODO:
        - Some older runs do not have a pbt_group_key, and likely also no pbt_epoch
    """
    metric_str: str = metrics if isinstance(metrics, str) else "-".join(map(_join_nested, metrics))
    if not isinstance(experiment_path, pd.DataFrame):
        experiment_path = Path(experiment_path)
        if save_path is None:
            save_path = experiment_path / "plots"
        else:
            save_path = Path(save_path)
        logger.info("Loading run data %s ...", experiment_path.name)
        data = load_run_data(experiment_path, use_cache=use_cache)
        if not isinstance(data, pd.DataFrame):
            logger.debug("Combining dataframes...")
            combined_df = combine_df(data)
            save_run_data(experiment_path, combined_df)
        else:
            combined_df = data
            if "pbt_epoch" not in combined_df.config and use_cache:
                logger.warning("Loaded old cache but pbt_epoch not in data, reloading without cache.")
                data = load_run_data(experiment_path, use_cache=False)
                if not isinstance(data, pd.DataFrame):
                    combined_df = combine_df(data)
                    save_run_data(experiment_path, combined_df)
        out_path = None
    elif save_path is None:
        raise ValueError("When providing a DataFrame directly, save_path must be specified.")
    else:
        out_path = Path(save_path)
        combined_df = experiment_path
    # Mark large experiments:
    large = ""
    if group_stat is None:
        group_stat = _get_group_stat(combined_df, group_by[-1])
    if group_stat not in ("batch_size", "train_batch_size_per_learner"):
        batch_size = combined_df.iloc[0].config.get(
            "train_batch_size_per_learner",
            combined_df.iloc[0].config.get("batch_size", combined_df.iloc[0].get("batch_size", None)),
        )
        if not isinstance(batch_size, (int, float, np.integer, np.floating)):
            if isinstance(batch_size, pd.Series):
                batch_size = batch_size.iloc[0]
            else:
                batch_size = batch_size.item()
        if batch_size >= 8192:
            large = "(large)"
    if out_path is None:
        out_path = Path(save_path, f"{experiment_path.name}_{metric_str}{large}.{format}")
    logger.info("Plotting and saving...")

    # If group_stat is None, do what?
    outfiles: list[Path] = []
    if calc_stats:
        try:
            outfiles = calculate_experiment_stats(
                combined_df=combined_df,
                metrics=metrics,
                out_path=out_path,
                format=format,
                plot_option=plot_option,
                large=bool(large),
            )
        except Exception as e:  # noqa: PERF203
            logger.exception("Failed to calculate experiment stats: %r", e)  # noqa: G004
    files = plot_n_save(
        combined_df,
        plot_option=plot_option,
        metrics=metrics,
        save_path=out_path,
        experiment_keys=experiment_keys,
        group_stat=group_stat,
        group_by=group_by,
        figsize=figsize,
        log=log,
        format=format,
        **kwargs,
    )
    if not files:
        # possibly no main data
        return outfiles or None
    return [*files, *outfiles]


def _export_one(args, group_by, figsize, format, **kwargs):
    experiment_path, metric = args
    try:
        file_path = export_run_data(
            experiment_path,
            metrics=(metric,),
            group_by=group_by,
            figsize=figsize,
            format=format,
            **kwargs,
        )
    except Exception as e:  # noqa: PERF203
        tb = traceback.format_exc()
        logger.error("Failed to export run data for %s: %r", experiment_path, e, exc_info=True)  # noqa: G004
        return e, tb, experiment_path
    else:
        if file_path is not None:
            logger.info("Exported run data for %s to %s", experiment_path, file_path)  # noqa: G004
        return file_path


def clean_placeholder_keys(
    d: dict[str | tuple[str, ...], dict | Any], *, flatten: bool = False
) -> dict[str | tuple[str, ...], dict | Any]:
    def clean_key(key):
        if isinstance(key, tuple):
            # Remove Placeholder values
            cleaned = tuple(k for k in key if k not in (Placeholder, "<Placeholder>"))
            if len(cleaned) == 1:
                return cleaned[0]
            if flatten:
                return "/".join(map(str, cleaned))
            return cleaned
        return key

    cleaned = {}
    for k, v in d.items():
        new_key = clean_key(k)
        if isinstance(v, dict):
            v = clean_placeholder_keys(v)
        elif isinstance(v, (pd.Series, pd.DataFrame)) and v.size == 1:
            v = v.item()
        cleaned[new_key] = v
    return cleaned


def _export_multiple(
    experiment_path: Path,
    plot_options: Sequence[PlotOption],
    metrics: Sequence[str | tuple[str, ...]],
    group_by: Sequence[str],
    figsize: tuple[int, int],
    format: str,
    **kwargs: OptionalRepeat[Any],
) -> tuple[list[Path], list[tuple[Exception, str, Path]]]:
    experiment_path = Path(experiment_path)
    saved_files: list[Path] = []
    try:
        data = load_run_data(experiment_path)
    except Exception as e:  # noqa: PERF203
        tb = traceback.format_exc()
        logger.error(f"Failed to load run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
        return saved_files, [(e, tb, experiment_path)]

    if not isinstance(data, pd.DataFrame):
        try:
            combined_df = combine_df(data)
        except Exception as e:  # noqa: PERF203
            tb = traceback.format_exc()
            logger.error(f"Failed to combine dataframes for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
            return saved_files, [(e, tb, experiment_path)]
        save_run_data(experiment_path, combined_df)
    else:
        combined_df = data
    # Expand kwargs if any value is an Options instance (cross product over all Options)
    large = False
    if (group_stat := kwargs.get("group_stat")) is None:
        group_stat = _get_group_stat(combined_df, group_by[-1])
    if group_stat not in ("batch_size", "train_batch_size_per_learner"):
        batch_size = combined_df.iloc[0].config.get(
            "train_batch_size_per_learner",
            combined_df.iloc[0].config.get("batch_size", combined_df.iloc[0].get("batch_size", None)),
        )
        try:
            if batch_size >= 8192:
                large = True
        except ValueError:
            if not batch_size.empty:
                if isinstance(batch_size, pd.Series):
                    if batch_size.iloc[0] >= 8192:
                        large = True
                elif batch_size.iloc[0, 0] >= 8192:
                    large = True

    # Identify keys in kwargs whose values are Options
    kwargs["group_by"] = group_by
    del group_by
    option_keys = [k for k, v in kwargs.items() if isinstance(v, Repeat)]
    errors: list[tuple[Exception, str, Path]] = []
    if option_keys:
        # Build a list of lists for each Options value, or [v] if not Options
        option_values: list[Repeat[object]] = [kwargs[k] for k in option_keys]  # pyright: ignore[reportAssignmentType]
        # For each combination in the cross product, build a dict of overrides
        # remote_breakpoint()
        calced_stat_for_metric = set()
        for combo in product(*option_values):
            combo_kwargs = kwargs.copy()
            for k, v in zip(option_keys, combo, strict=True):
                combo_kwargs[k] = v
            combo_kwargs = cast("dict[str, Any]", combo_kwargs)
            assert not any(isinstance(v, Repeat) for v in combo_kwargs.values()), (
                "All Repeat values should have been expanded."
            )
            if combo_kwargs.get("main_only", False) and combo_kwargs.get("plot_reduced", False):
                logger.info(
                    "Skipping combination with main_only=True and plot_reduced=True as they are mutually exclusive."
                )
                continue
            for metric in metrics:
                for option in plot_options:
                    groupby_option = combo_kwargs.get("group_by", kwargs.get("group_by", DEFAULT_GROUP_BY))
                    if option.exclude(metric, groupby_option):
                        continue
                    metric_str = metrics if isinstance(metrics, str) else "-".join(map(_join_nested, metrics))
                    out_path = _create_image_output_path(
                        experiment_path,
                        metric,
                        plot_option=option,
                        output_dir=experiment_path,
                        format=format,
                        large=large,
                        group_by=groupby_option,
                    )
                    logger.debug(
                        "Exporting metric '%s' with options %s to '%s' ...", metric_str, combo_kwargs, out_path
                    )
                    try:
                        file_paths = export_run_data(
                            combined_df,
                            plot_option=option,
                            metrics=(metric,),
                            figsize=figsize,
                            format=format,
                            save_path=out_path,
                            calc_stats=metric not in calced_stat_for_metric,
                            **combo_kwargs,
                        )
                        calced_stat_for_metric.add(metric)
                    except Exception as e:  # noqa: PERF203
                        tb = traceback.format_exc()
                        logger.error(f"Failed to export run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
                        errors.append((e, tb, experiment_path))
                    else:
                        if file_paths is not None:
                            dump_str = "\n".join(map(str, file_paths))
                            logger.info(f"Exported run data for '{experiment_path}' to '{dump_str}'\n-----")  # noqa: G004
                            saved_files.extend(file_paths)
        return saved_files, errors
    kws = cast("dict[str, Any]", kwargs)
    for metric in metrics:
        calc_stats = True
        for option in plot_options:
            if option.exclude(metric, kws.get("group_by", DEFAULT_GROUP_BY)):
                continue
            out_path = _create_image_output_path(
                experiment_path,
                metric,
                plot_option=option,
                output_dir=experiment_path,
                format=format,
                large=large,
                group_by=kws.get("group_by", DEFAULT_GROUP_BY),
            )
            try:
                file_paths = export_run_data(
                    combined_df,
                    plot_option=option,
                    metrics=(metric,),
                    figsize=figsize,
                    format=format,
                    save_path=out_path,
                    calc_stats=calc_stats,
                    **kws,
                )
            except Exception as e:  # noqa: PERF203
                tb = traceback.format_exc()
                logger.error(f"Failed to export run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
                errors.append((e, tb, experiment_path))
            else:
                # Calc stats are only important for metric, plot options do not influence it (only title true false)
                calc_stats = False
                if file_paths is not None:
                    dump_str = "\n".join(map(str, file_paths))
                    logger.info(f"Exported run data for {experiment_path} to {dump_str}\n-----")  # noqa: G004
                    saved_files.extend(file_paths)
    return saved_files, errors


T = TypeVar("T")


class Repeat(list[T]):
    pass


OptionalRepeat: TypeAlias = Repeat[T] | T


def _create_image_output_path(
    experiment_path: Path,
    metric: str | tuple[str, ...],
    plot_option: PlotOption,
    *,
    output_dir: Path,
    format: str,
    large: bool = False,
    group_by: Sequence[str],
) -> Path:
    metric_str = "-".join(metric) if isinstance(metric, tuple) else metric
    subdir = metric_str
    suffixes = []
    if plot_option.main_only:
        suffixes.append("main_only")
    # Main not compatible with reduced
    elif plot_option.plot_reduced:
        suffixes.append("reduced")
    elif plot_option.main_vs_second_best:
        suffixes.append("main_vs_second_best")
    elif plot_option.main_vs_rest:
        suffixes.append("main_vs_rest")
    else:
        suffixes.append("all")
    if group_by == DEFAULT_GROUP_BY:
        group_by_str = ""
    else:
        group_by_str = "_".join(group_by) if not isinstance(group_by, str) else group_by
        suffixes.append(f"_groupby_{group_by_str}")
    suffix_str = ("-" + "_".join(suffixes)) if suffixes else ""
    if large:
        suffix_str += "(large)"
    file_path = output_dir / "plots" / subdir / f"{experiment_path.name}_{metric_str}{suffix_str}.{format}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def export_all_runs(
    output_dir: str | Path | Iterable[str | Path],
    plot_options: Sequence[PlotOption],
    *,
    single: bool = False,
    test=TEST,
    max_workers: int = 4,
    zip_plots: bool = False,
    excludes: Collection[str] = (),
    redo: bool = False,
    # need to be passed on
    format="pdf",
    figsize=(14, 10),
    group_by=DEFAULT_GROUP_BY,
    metrics=("episode_reward_mean",),
    plot_errors: OptionalRepeat[bool] = True,
    plot_errors_type: OptionalRepeat[Literal["box", "violin"]] = "box",
    zip_file: Optional[ZipFile] = None,
    run_infos: Optional[dict[str, SubmissionRun]] = None,
    **kwargs,
):
    """
    Export runs using multiple processes.

    If `output_dir` is a single path, runs are discovered by globbing
    `*/*/*` under that directory (unless `single=True`, in which case
    only the provided directory is processed).

    If `output_dir` is a sequence of paths, those paths are treated as the
    complete set of run directories to process. In this mode, no globbing is
    performed and `single` is ignored to allow parallel processing of the
    provided runs.

    Args:
        output_dir: Directory containing experiment runs, or a list of explicit
            run directories to process.
        format: Output file format.
        figsize: Figure size.
        group_by: Grouping columns.
        metrics: Metrics to plot.
        test: If set, limits the number of runs processed.
        max_workers: Number of parallel processes to use.
        **kwargs: Additional keyword arguments for export_run_data.

    Returns:
        List of saved file paths.
    """
    # Normalize input into a list of experiment directories to process.
    # Also record a base root for relative zip paths when applicable.
    base_root: Path | None
    if isinstance(output_dir, (str, Path)):
        base_root = Path(output_dir)
        # Discover runs under the base directory unless single is set
        experiment_dirs: Iterable[Path] = base_root.glob("*/*/*") if not single else [base_root]
    else:
        # Explicit list of run directories provided; do not glob and ignore `single`
        assert not single
        experiment_dirs = (Path(p) for p in output_dir)
        base_root = None
    saved_files: list[Path] = []
    test = int(test) - 1 if test else float("inf")
    # Submit tasks as soon as we get experiment paths from glob
    file_paths: list[Path] = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        i = 0
        skip_dirs = set()
        for experiment_path in experiment_dirs:
            if any(excl in str(experiment_path) for excl in excludes) or "cometml" in str(experiment_path):
                if "cometml" not in str(experiment_path):
                    logger.info(f"Excluding experiment path {experiment_path} due to exclude patterns.")  # noqa: G004
                continue
            logger.info("Processing experiment path %s ...", experiment_path)
            if not experiment_path.is_dir():
                # possibly not yet sorted into groups
                experiment_path = experiment_path.parent  # noqa: PLW2901
                if not (experiment_path / ".validate_storage_marker").exists():
                    logger.error(f"Missing .validate_storage_marker in {experiment_path}, skipping.")  # noqa: G004
                    skip_dirs.add(experiment_path)
                    continue
                if experiment_path in skip_dirs:
                    logger.info(f"Skipping previously failed experiment path {experiment_path}.")  # noqa: G004
                    continue
                skip_dirs.add(experiment_path)
                # raise ValueError(f"Experiment path {experiment_path} is not a directory.")
            filtered_metrics = metrics.copy()
            # Prepare all combinations of main_only and plot_reduced if they are Repeat, else just use as is
            main_only_options = [o.main_only for o in plot_options]
            plot_reduced_options = [o.plot_reduced for o in plot_options]
            # If not redo, check for all combinations and remove metrics that already exist for all combinations
            redo = True  # XXX - # Currently this check does not inclue all plots so we always redo for now
            if not redo:
                metrics_to_remove = set()
                for metric in metrics:
                    for option in plot_options:
                        if option.exclude(metric, group_by):
                            continue
                        mo = option.main_only
                        pr = option.plot_reduced
                        if mo and pr:
                            continue
                        out_path = _create_image_output_path(
                            experiment_path,
                            metric,
                            plot_option=option,
                            output_dir=output_dir,
                            format=format,
                            group_by=group_by,
                        )  # noqa: E501
                        if out_path.exists():
                            metrics_to_remove.add((metric, mo, pr))
                            # Include in zip file paths even if skipping
                            file_paths.append(out_path)
                            logger.info(
                                f"Plot for metric {metric} (main_only={mo}, plot_reduced={pr}) already exists at {out_path}, skipping."  # noqa: G004
                            )
                # Remove metrics for which all combinations exist
                # If all combinations for a metric exist, remove it from filtered_metrics
                for metric in metrics:
                    if any(
                        options.exclude(metric, group_by) for options in plot_options if options.exclude_metric
                    ) and all(options.exclude(metric, group_by) for options in plot_options if options.exclude_metric):
                        filtered_metrics.remove(metric)
                        continue
                    all_exist = all(
                        (metric, mo, pr) in metrics_to_remove
                        for mo in main_only_options
                        for pr in plot_reduced_options
                    )  # fmt: skip
                    if all_exist and metric in filtered_metrics:
                        filtered_metrics.remove(metric)
            base_kwargs = kwargs.copy()
            base_kwargs.setdefault("plot_errors", plot_errors)
            base_kwargs.setdefault("plot_errors_type", plot_errors_type)
            futures.append(
                executor.submit(
                    _export_multiple,
                    experiment_path=experiment_path,
                    metrics=filtered_metrics,
                    group_by=group_by,
                    figsize=figsize,
                    format=format,
                    plot_options=plot_options,
                    **base_kwargs,
                )
            )
            if i >= test:
                break
            i += 1

        # Collect results as they complete
        tracebacks = []
        zipf: ZipFile | None = zip_file
        try:
            if zip_plots:  # zip skipped files as we go
                if not zip_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if base_root is not None:
                        zip_path = base_root / f"exported_plots_{timestamp}.zip"
                    else:
                        zip_path = Path.cwd() / f"exported_plots_{timestamp}.zip"
                    zipf = ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=3)
                else:
                    zip_path = zipf.fp
                assert zipf
                # If we not redo add files already present
                for file_path in file_paths:
                    # Derive archive name relative to any known base (base_root and PATHS)
                    base_candidates: list[Path] = ([base_root] if base_root is not None else []) + PATHS
                    arcname_str = make_zip_arcname(file_path, base_candidates, run_infos, use_dir_flags=True)
                    zipf.write(file_path, arcname=arcname_str)
            try:
                # Collect results as they complete
                file_paths: list[Path]
                errors: list[tuple[Exception, str, Path]]
                print("---------- Waiting and collecting results ----------")
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    file_paths, errors = future.result()
                    if isinstance(file_paths, (Path, str)):
                        file_paths = [Path(file_paths)]
                    print("Exported files:", file_paths)
                    saved_files.extend(file_paths)
                    if zipf is not None:
                        base_candidates: list[Path] = ([base_root] if base_root is not None else []) + PATHS
                        for file_path in file_paths:
                            if not file_path.exists():
                                logger.error(f"File {file_path} does not exist, cannot add to zip.")  # noqa: G004
                                continue
                            arcname_str = make_zip_arcname(file_path, base_candidates, run_infos, use_dir_flags=True)
                            zipf.write(file_path, arcname=arcname_str)
                    for error_return in errors:
                        error, tb, failed_path = error_return
                        logger.error(f"Error during export of {failed_path} : {error!r}\n{tb}")  # noqa: G004
                        tracebacks.append((failed_path, tb))
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received: cancelling all running futures.")
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        finally:
            if zip_file is None and zipf is not None:
                zipf.close()

    if tracebacks:
        logger.error("Some exports failed with errors:")
        for failed_path, tb in tracebacks:
            print("\n--------------------------------\n")
            logger.error(f"Failed export for {failed_path}:\n{tb}")  # noqa: G004
    end = time.time()
    logger.info(
        "Exported %d files in %s:\n%s",
        len(saved_files),
        f"{(end - start) / 60:.1f} min",
        "\n".join(map(str, {f.parent.parent for f in saved_files})),
    )  # noqa: G004
    if zip_plots:
        logger.info(f"Zipped plots saved to {zip_path}")  # pyright: ignore[reportPossiblyUnboundVariable] # noqa: G004
    return saved_files


def get_running_experiments(experiment_dir: str | Path = "./experiments", *, include_restore: bool = False):
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise ValueError(f"Experiment directory {experiment_dir} does not exist or is not a directory.")
    submission_files = list(experiment_dir.glob("submission*.yaml"))
    running_experiments = set()
    for submission_file in submission_files:
        with open(submission_file, "r") as f:
            submission_data = yaml_load(f)
        if not isinstance(submission_data, dict):
            logger.warning(f"Submission file {submission_file} is not a dict, skipping.")  # noqa: G004
            continue
        for group in submission_data.values():
            if not isinstance(group, dict) or "run_ids" not in group:
                continue
            for _run_key, runs in group["run_ids"].items():
                for run_id, run_info in runs.items():
                    if isinstance(run_info, dict):
                        if run_info.get("status") == "RUNNING":
                            running_experiments.add(run_id)
                        if include_restore and "RESTORE" in (run_info.get("status") or ""):  # status might be None
                            running_experiments.add(run_id)
    # Clean some wrong keys
    return {k for k in running_experiments if "-v" not in k and "_restore" not in k}


def load_excludes():
    try:
        with open("plot_exclude.txt", "r") as f:
            file_excludes: set[str] = {line.strip() for line in f if line.strip() and not line.startswith("#")}
            logger.info("Loaded plot excludes: %s", file_excludes)
    except FileNotFoundError:
        logger.info("No plot excludes found.")
        file_excludes = set()
    return file_excludes


if __name__ == "__main__":
    # cd /path/to/parent
    # find dir1 dir2 -type f -iname '*.pdf' -printf '%P\n' | sed 's|^[^/]*/||' | zip -@ all_pdfs.zip
    import argparse
    from datetime import datetime

    from ray_utilities import nice_logger
    from ray_utilities.visualization.plot import logger as plot_logger

    file_excludes = load_excludes()
    running_experiments = get_running_experiments(Path(__file__).parent.parent.parent / "experiments")

    logger = nice_logger(logger, logging.INFO)
    plot_logger.setLevel(logging.INFO)
    # logging.info("Set handle")  # noqa: LOG015
    # logger.setLevel()
    logger.info("Running data export script...")

    parser = argparse.ArgumentParser(description="Export and plot RLlib experiment results.")
    parser.add_argument("path", nargs="*", type=str, help="Experiment output directory or run path.")
    parser.add_argument("--all", action="store_true", help="Export all runs in the directory.")
    parser.add_argument("--main_only", action="store_true", help="Plot only the main branch runs.")
    parser.add_argument("--format", type=str, default="pdf", help="Output file format (default: pdf).")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics to plot.")
    parser.add_argument(
        "--figsize", type=int, nargs=2, default=[14, 10], help="Figure size as two integers (width height)."
    )
    parser.add_argument(
        "--test", nargs="?", const=True, default=False, type=int, help="Run in test mode with limited data."
    )
    parser.add_argument("--pbt_metric", type=str, default="episode_reward_mean", help="PBT selection metric.")
    parser.add_argument(
        "--workers", "-w", "-p", type=int, default=4, help="Number of parallel workers for exporting all runs."
    )
    parser.add_argument("--zip", "-z", action="store_true", default=False, help="Whether to zip the output plots.")
    parser.add_argument(
        "--excludes", nargs="*", type=str, default=["def-workspace", "TESTING"], help="Experiment keys to exclude."
    )
    parser.add_argument("--redo", action="store_true", help="Redo existing plots.")
    parser.add_argument(
        "--load-figures", "-f", action="store_true", help="Load cached figure data for faster re-plotting."
    )
    parser.add_argument("--single", "-s", action="store_true", help="Export a single run instead of all runs.")
    parser.add_argument("--no_error", "-nb", action="store_true", help="Disable error bars in plots.")
    parser.add_argument(
        "--error-plot-type",
        choices=["box", "violin"],
        default="box",
        help="Error plot type (box or violin).",
    )
    parser.add_argument("--title", action="store_true", help="Disable titles in plots.")
    args = parser.parse_args()

    assert not args.all or not args.main_only, "Cannot use --all and --main_only together."
    if args.metrics is None:
        args.metrics = [
            "episode_reward_mean",
        ]  # ("training", "episode_return_mean")]  # , ("learners", "total_loss")]  # XXX
    else:
        args.metrics = list(map(literal_eval, args.metrics))
    excludes: set[str] = file_excludes.union(set(args.excludes)).union(running_experiments)
    plot_options = [
        PlotOption(plot_reduced=False, title=args.title),  # plot all
        PlotOption(plot_reduced=True, title=args.title),
        PlotOption(main_only=True, plot_reduced=False, title=args.title),
        PlotOption(main_vs_second_best=True, plot_reduced=False, title=args.title),
        PlotOption(main_vs_rest=True, plot_reduced=False, title=args.title),
    ]
    if args.test:
        plot_options = plot_options[1:2]
    paths = list(args.path)
    if not paths:
        paths = [
            "outputs/shared/experiments/",
            "outputs/shared_backup/",
            "outputs/shared_backup/",
        ]
    zipfile = None
    run_infos: dict[str, SubmissionRun] = {}
    processing_submissions = any(p.endswith(".yaml") for p in paths)
    run_directory_iter: Iterable[Path] | None = None
    if processing_submissions:
        assert all(p.endswith(".yaml") for p in paths), (
            f"Either all or none of the paths must be YAML files. Got: {paths}"
        )
        # Load experiment groups from YAML files
        yaml_paths = [Path(p) for p in paths]
        # Do not use single-run mode when aggregating from submissions;
        # we'll parallelize across runs via common parent directories.
        assert not args.single
        run_directory_iter = (
            directory
            for yaml_path in yaml_paths
            for _, directory in get_run_directories_from_submission(yaml_path, test=args.test)
        )
        for yaml_path in yaml_paths:
            run_infos.update(get_runs_from_submission(yaml_path))
        # need a common zipfile
        submission_str = "+".join(
            [re.sub(r"submissions?_?", "", yaml_path.stem).replace("pbt_", "").strip("_- ") for yaml_path in yaml_paths]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = Path("outputs/shared/experiments") / f"exported_plots_{submission_str}_{timestamp}.zip"
        zipfile = ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=3)
        # TODO: do not do for path in paths + single! No parallelism
        paths = [run_directory_iter]  # one iteration to parallelise

    try:
        for path in paths:
            export_all_runs(
                path,
                single=args.single,
                plot_options=plot_options,
                test=args.test,
                metrics=args.metrics,
                group_by=DEFAULT_GROUP_BY,
                figsize=tuple(args.figsize),
                format=args.format,
                pbt_metric=args.pbt_metric,
                max_workers=args.workers,
                zip_plots=args.zip,
                excludes=excludes,
                redo=args.redo,
                plot_errors=not args.no_error,
                plot_errors_type=args.error_plot_type,
                zip_file=zipfile,
                run_infos=run_infos,
            )
    finally:
        if zipfile:
            logger.info("Zipped plots saved to %s", zipfile.fp)
            zipfile.close()


# Example for grouping by MultiIndex columns:
# group_cols = [("config", "pbt_epoch"), ("config", "pbt_group_key")]
# group_values = [df[col] for col in group_cols]
# group_df = pd.concat(group_values, axis=1)
# group_df.columns = ["pbt_epoch", "pbt_group_key"]
# grouped = df.groupby([group_df["pbt_epoch"], group_df["pbt_group_key"]])
