#!/usr/bin/bash
# PYTHON_ARGCOMPLETE_OK

# ruff: noqa: T201

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, cast

import argcomplete
import ray
import wandb

from ray_utilities.callbacks.wandb import (
    FailureDictType,
    RunNotFound,
    VerificationFailure,
    find_experiment_dir,
    wandb_api,
)
from ray_utilities.callbacks.wandb import logger as ru_wandb_logger
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import ExperimentKey, RE_GET_TRIAL_ID

if TYPE_CHECKING:
    import pandas as pd
    from wandb.apis.public.runs import Run as RunApi

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload logs to WandB for a given run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser):
        subparser.add_argument(
            "project", type=str, help="WandB project name (required). Need to know which project to monitor."
        )
        subparser.add_argument(
            "run_id",
            type=str,
            nargs="?",
            default=None,
            help="WandB run ID (optional positional; required for upload/verify).",
        )
        subparser.add_argument(
            "--experiment_path",
            type=str,
            default=os.environ.get("RAY_UTILITIES_STORAGE_PATH", "./outputs/experiments/shared"),
            help="Path to experiment outputs (default: RAY_UTILITIES_STORAGE_PATH or ./outputs/experiments/shared).",
        )
        subparser.add_argument("--entity", type=str, default=None, help="WandB entity (optional).")
        subparser.add_argument(
            "--experiment_key", type=str, default=None, help="WandB to upload check a single experiment key (optional)."
        )
        subparser.add_argument(
            "wandb_args",
            nargs=argparse.REMAINDER,
            help="Capture all arguments following '--'.",
        )

    # Upload subparser
    upload_parser = subparsers.add_parser("upload", help="Upload WandB logs for a run.")
    add_common_arguments(upload_parser)
    upload_parser.add_argument("--no-monitor", action="store_true", help="Do not start the monitor thread.")
    upload_parser.add_argument(
        "--first-verify", action="store_true", help="Perform verification before uploading files."
    )

    # Verify subparser
    verify_parser = subparsers.add_parser("verify", help="Verify WandB run without uploading files.")
    add_common_arguments(verify_parser)
    verify_parser.add_argument(
        "--skip-checked", action="store_true", help="Skip runs in YAML that already have a 'failures' key set."
    )

    # Patch subparser
    patch_parser = subparsers.add_parser(
        "patch", help="Patch an incomplete offline history JSON file using WandB online data."
    )
    add_common_arguments(patch_parser)

    argcomplete.autocomplete(parser)
    return parser


def experiment_ids_from_submissions_yaml(
    file: Path, experiments: list[str] | str | None = None, *, skip_checked: bool = False
):
    """
    Layout:
        group_name:
            (project):
                run_id:
                    status: ...
                    failures:
                       - none  # single entry if we are fine
                       - experiment_id: [list of failures]
    """
    from ray_submit import yaml_load  # noqa: PLC0415

    with open(file, "r") as f:
        data = yaml_load(f)
    groups_to_check: dict[str, dict[str, dict]] = {}
    if experiments:
        if isinstance(experiments, str):
            groups_to_check[experiments] = data[experiments]["run_ids"]
        else:
            for group in experiments:
                runs_to_add = {}
                for environment in data[group]["run_ids"]:
                    entry = data[group]["run_ids"][environment]
                    for run_id, run_status in entry.copy().items():
                        # Skip if previously no failures were detected - and explicitly added as marker
                        if "failures" in run_status and run_status["failures"] in (["none"], "none"):
                            del entry[run_id]
                        elif skip_checked and "failures" in run_status:
                            del entry[run_id]
                    if entry:
                        runs_to_add[environment] = entry
                if runs_to_add:
                    groups_to_check[group] = runs_to_add
    else:
        # get all groups that have a run_ids section
        for group_name, group in data.items():
            if "run_ids" not in group:
                continue
            runs_to_add = {}
            for environment, entry in group["run_ids"].items():
                for run_id, run_status in entry.copy().items():
                    if "failures" in run_status and run_status["failures"] in (["none"], "none"):
                        del entry[run_id]
                    elif skip_checked and "failures" in run_status:
                        del entry[run_id]
                if entry:
                    runs_to_add[environment] = entry
            if runs_to_add:
                groups_to_check[group_name] = runs_to_add
    for group_name, run_ids in groups_to_check.items():
        runs: dict[str, dict[str, str] | str]
        for env, runs in run_ids.items():
            experiment_ids = runs.keys()
            # TODO currently we put hyperparameters inside () in the group name, currently this is only the env
            # so it is fine but not future proof.
            yield from ((group_name, f"Default-mlp-{env.strip('()')}", exp_id) for exp_id in experiment_ids)


def _filter_keys_from_config(config: dict) -> dict:
    from ray_utilities.callbacks.tuner._log_result_grouping import exclude_results  # noqa: PLC0415

    for key in (*exclude_results, "log_stats", "offline_loggers", "_config_files"):
        if "/" in key and key not in config:
            # Nested key
            parts = key.split("/")
            sub_dict = config
            for part in parts[:-1]:
                sub_dict = sub_dict.get(part, {})
                if not isinstance(sub_dict, dict):
                    break
            else:
                sub_dict.pop(parts[-1], None)
        else:
            config.pop(key, None)
    return config


def _get_parents_from_json(offline_path: Path | str, parents: dict | None = None) -> dict:
    parents = {} if parents is None else parents
    with open(offline_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if FORK_FROM in data.get("config", {}):
                parents[data["config"][FORK_FROM]["parent_fork_id"]] = data["config"][FORK_FROM]["fork_id"]
    return parents


def _find_parent_json_file(offline_path: Path | str, parent_run_id: str) -> Path | None:
    # Get trial_number

    offline_path = Path(offline_path)
    if ExperimentKey.FORK_SEPARATOR in parent_run_id:
        parent_search = offline_path.parent.parent.glob(f"*/result*{parent_run_id}.json")
    else:
        parent_trial_part = parent_run_id.split(ExperimentKey.RUN_ID_SEPARATOR)[-1]
        if ExperimentKey.COUNT_SEPARATOR in parent_trial_part:
            trial_part, count_part = parent_trial_part.rstrip(ExperimentKey.RIGHT_PAD_CHAR).split(
                ExperimentKey.COUNT_SEPARATOR
            )
            if ExperimentKey.REPLACE_3ZEROS != "" and ExperimentKey.REPLACE_3ZEROS in count_part:
                count_part = count_part.replace(ExperimentKey.REPLACE_3ZEROS, "000")
            else:
                count_part = "000" + count_part
        else:
            trial_part, count_part = parent_trial_part.split("_")
        parent_run_id = f"{trial_part}_{count_part}"  # noqa: PLW2901
        trial_number = int(count_part)
        if id_match := RE_GET_TRIAL_ID.search(str(offline_path)):
            path_id = int(id_match["trial_number"])
            if path_id != trial_number:
                # We are looking in the wrong path - which is the normal case
                # assume offline_path points to the json file
                parent_search = offline_path.parent.parent.glob(f"*id={parent_run_id}*/result.json")
            else:
                # same dir
                parent_search = offline_path.parent.glob("result.json")
        else:
            logger.error("Cannot identify which result.json file to select for partent run id %s", parent_run_id)
            # TODO: Need to check all for the id, but we should not end up here normally
            return None
        # check trial number
    parent_files = list(parent_search)
    if len(parent_files) > 1:
        logger.warning("Found multiple parent json files for parent run id %s: %s", parent_run_id, parent_files)
        return parent_files[-1]
    if len(parent_files) == 1:
        return parent_files[0]
    # possibly looking for result.json
    if parent_run_id.endswith(ExperimentKey.RIGHT_PAD_CHAR) and ExperimentKey.FORK_SEPARATOR not in parent_run_id:
        parent_search = offline_path.parent.glob("result.json")
        parent_files = list(parent_search)
        if len(parent_files) == 1:
            return parent_files[0]
    logger.error("Did not find parent json file for parent run id %s in %s", parent_run_id, offline_path.parent)
    return None


def patch_offline_history(
    offline_path: Path | str,
    run: Optional[RunApi | str] = None,
    *,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    experiment_key: Optional[str] = None,
    patch_with_offline: bool = False,
) -> None:
    """
    Patch an incomplete offline history JSON file with online WandB data.

    Args:
        offline_path: Path to the incomplete offline history JSON file.
        run: Optional WandB run object or run ID.
        entity: Optional WandB entity.
        project: Optional WandB project name.
        run_id: Optional WandB run ID.

    Returns:
        None
    """
    from ray.rllib.utils import unflatten_dict  # noqa: PLC0415

    api = wandb_api()
    if run is not None:
        if isinstance(run, str):
            run = api.run(run)
    else:
        assert (project, experiment_key) != (None, None), "Either run or both project and run_id must be provided."
        api = wandb_api()
        if entity:
            run = api.run(f"{entity}/{project}/{experiment_key}")
        else:
            run = api.run(f"{project}/{experiment_key}")
    # load or create offline_path
    offline_iteration_data = {}
    offline_path = Path(offline_path)
    if not offline_path.exists():
        offline_path.mkdir(parents=True, exist_ok=True)
        offline_path.touch()
    else:
        # check which lines exist
        # NOTE: The restored offline data cannot restore the config accurately only the metrics.
        with offline_path.open("r+") as f:
            for line in f:
                data = json.loads(line)
                offline_iteration_data[data["training_iteration"]] = data
    run = cast("RunApi", run)
    run_config = _filter_keys_from_config(run.config)
    online_history = run.history(samples=8000, pandas=False)
    online_iteration_data = {
        int(entry.get("training_iteration", entry.get("_step"))): unflatten_dict(entry) | {"config": run_config}
        for entry in online_history
    }
    # FIXME: Problem if a forked run is a top trial during perturbation it does lose its FORK_FROM info from the config
    # We might overwrite the config in the AdvWandbLogger loosing the config information.
    # We can use the new trial_id_history if available to find the parent runs.
    # CAREFUL: If we load a parent it means it has continued training, only load data from before the fork_point
    parent_runs: list[RunApi] = []
    parent_offline_files = []
    if "trial_id_history" in run.config:
        # do we have a clue about the fork point here?
        parent_run_ids = [
            run_id
            for i, run_id in sorted(
                run.config["trial_id_history"].items(),
                key=lambda x: int(x[0]) if x[0].isdigit() else -1,
            )
            if i != "original_experiment_key" and run_id != run.id
        ]
        for i, parent_run_id in enumerate(parent_run_ids):
            # TODO: There was a bug that inserted a wrong parent into the trial_id_history
            # when a forked run was continued
            if i != 0 and (
                parent_run_id.endswith(ExperimentKey.RIGHT_PAD_CHAR)
                or ExperimentKey.FORK_SEPARATOR not in parent_run_id
            ):
                logger.warning(
                    "parent_run_id in slot %s seems might be wrong due to a bug. Check carefully. Skipping. All Id: %s",
                    i,
                    parent_run_id,
                )
                if sys.argv[0] == "":
                    logger.info("Require input...")
                    while (choice := input("Skip this parent? (y/n): ").lower()) not in ("y", "n"):
                        pass
                    if choice == "y":
                        continue
                else:
                    continue
            parent_file = _find_parent_json_file(offline_path, parent_run_id)
            # if parent_file:
            # parent_offline_files.append(parent_file)
            try:
                try:
                    if entity:
                        parent_run = api.run(f"{entity}/{project}/{parent_run_id}")
                    else:
                        parent_run = api.run(f"{project}/{parent_run_id}")
                except wandb.errors.errors.CommError:
                    if ExperimentKey.RIGHT_PAD_CHAR not in parent_run_id:
                        raise
                    # Convert to abc13_0000 format
                    id_part, count_part = (
                        parent_run_id.split(ExperimentKey.RUN_ID_SEPARATOR)[-1]
                        .rstrip(ExperimentKey.RIGHT_PAD_CHAR)
                        .split(ExperimentKey.COUNT_SEPARATOR)
                    )
                    if ExperimentKey.REPLACE_3ZEROS != "" and ExperimentKey.REPLACE_3ZEROS in count_part:
                        count_part = count_part.replace(ExperimentKey.REPLACE_3ZEROS, "000")
                    else:
                        count_part = "000" + count_part
                    parent_run_id = f"{id_part}_{count_part}"  # noqa: PLW2901
                    if entity:
                        parent_run = api.run(f"{entity}/{project}/{parent_run_id}")
                    else:
                        parent_run = api.run(f"{project}/{parent_run_id}")

                parent_runs.append(parent_run)
            except Exception as e:
                logger.warning("Could not find parent run %s: %s", parent_run_id, e)
    elif FORK_FROM in run.config:
        parent_run_ids = []
        parent_run_ids.insert(0, run.config[FORK_FROM]["parent_fork_id"])
        parent_run = (
            api.run(f"{entity}/{project}/{parent_run_ids[0]}") if entity else api.run(f"{project}/{parent_run_ids[0]}")
        )
        # oldest parent is last
        parent_runs.insert(0, parent_run)
        try:
            while FORK_FROM in parent_run.config:
                parent_run_ids.insert(0, parent_run.config[FORK_FROM]["parent_fork_id"])
                parent_run = (
                    api.run(f"{entity}/{project}/{parent_run_ids[0]}")
                    if entity
                    else api.run(f"{project}/{parent_run_ids[0]}")
                )
                parent_runs.insert(0, parent_run)
        except Exception as e:
            logger.warning("Could not find parent run %s: %s", parent_run_ids[0], e)
    parent_histories = {}
    if parent_runs:
        min_step = min(online_iteration_data.keys())

        # get online history of parents, trim to fork point
        def insert_parent_config(run, record):
            record["config"] = run.config
            record["config"]["__patched_from_wandb__"] = True
            return record

        for parent_run in reversed(parent_runs):
            parent_online_history = parent_run.history(samples=8000, pandas=True)  # pyright: ignore[reportAssignmentType]
            parent_online_history: pd.DataFrame = parent_online_history[
                parent_online_history["training_iteration"] < min_step
            ]
            if parent_online_history.empty:  # we should always have some parent data if we iterate correctly.
                continue
            parent_config = _filter_keys_from_config(run.config)
            parent_histories.update(
                {
                    record["training_iteration"]: insert_parent_config(
                        parent_run, unflatten_dict(cast("dict", record)) | {"config": parent_config}
                    )
                    for record in parent_online_history.to_dict(orient="records")
                    if record["training_iteration"] < min_step
                }
            )
            min_step = parent_online_history["training_iteration"].min()
    # If the online history is a fork we must query all parents as well.
    del online_history
    new_data_file = offline_path.with_suffix(".patched.json")
    merge_data = {**parent_histories, **online_iteration_data, **offline_iteration_data}
    with new_data_file.open("w") as f:
        for iteration in sorted(merge_data.keys()):
            data = merge_data[iteration]
            f.write(json.dumps(data) + "\n")


def _write_failures_to_submission_file(
    file: Path,
    experiment_failures: FailureDictType,
    group_mapping: dict[str, dict[str, str]],
    no_failures: Iterable[str] = (),
):
    from ray_submit import yaml_dump, yaml_load  # noqa: PLC0415

    with open(file, "r") as f:
        data = yaml_load(f)
    failure_count = defaultdict(int)
    for no_failed_experiment in no_failures:
        submission_group = group_mapping[no_failed_experiment]["submission_group"]
        project = group_mapping[no_failed_experiment]["project"]
        experiment_id = group_mapping[no_failed_experiment]["experiment_id"]
        group_data = data[submission_group]["run_ids"][f"({project})"][experiment_id]
        if isinstance(group_data, str):
            group_data = data[submission_group]["run_ids"][f"({project})"][experiment_id] = {
                "value": group_data,
                "failures": ["none"],
            }
        else:
            group_data["failures"] = ["none"]
        experiment_failures.pop(no_failed_experiment, None)  # pyright: ignore[reportArgumentType, reportCallIssue]
    for run, failures in experiment_failures.items():
        try:
            if run.id not in group_mapping:
                # Early write
                logger.debug("Skipping run %s not in group mapping.", run.id)
                continue
            submission_group = group_mapping[run.id]["submission_group"]
            project = group_mapping[run.id]["project"]
            experiment_id = group_mapping[run.id]["experiment_id"]
            # Find the entry in the yaml file
            group_data = data[submission_group]["run_ids"][f"({project})"][experiment_id]
            try:
                group_data.setdefault("failures", {})
            except AttributeError:
                group_data = data[submission_group]["run_ids"][f"({project})"][experiment_id] = {
                    "value": group_data,
                    "failures": {},
                }
            if "multiple" in group_data["failures"]:
                continue
            failure_count[(submission_group, project, experiment_id)] += 1
            if failure_count[(submission_group, project, experiment_id)] >= 5:
                # Avoid flooding the yaml file with too many failures
                group_data["failures"]["multiple"] = ", ".join(
                    {fail for run_id in group_data["failures"] for fail in group_data["failures"][run_id]}
                )
                continue

            if not group_data["failures"].get(run.id):
                group_data["failures"][run.id] = []
            if isinstance(failures, Exception):
                group_data["failures"][run.id].append(str(failures))
            else:
                group_data["failures"][run.id] = [failure.type.value for failure in failures] if failures else ["none"]
            # remove duplicates
            group_data["failures"][run.id] = list(set(group_data["failures"][run.id]))
        except Exception:
            logger.exception("Error writing failure for run %s:", run.id)
    with open(file, "w") as f:
        yaml_dump(data, f)


def _verify_one(group_name: str, project: str, experiment_id: str):
    uploader_local = WandbUploaderMixin()
    uploader_local.project = project
    project_key = project.removeprefix("Default-mlp-")
    local_group_mapping = {
        experiment_id: {
            "submission_group": group_name,
            "project": project_key,
            "experiment_id": experiment_id,
        }
    }
    print("\nVerifying project:", f"{project}/{experiment_id}")
    experiment_results = {experiment_id: {}}
    experiment_failures = uploader_local.verify_wandb_uploads(
        experiment_id=experiment_id, output_dir=None, single_experiment=None, experiment_results=experiment_results
    )
    for run in experiment_failures.keys():
        local_group_mapping[run.id] = {
            "submission_group": group_name,
            "project": project_key,
            "experiment_id": experiment_id,
        }
    local_failures = {
        run: failure for run, failure in experiment_failures.items() if failure or isinstance(run, RunNotFound)
    }
    return (project, group_name, experiment_id, local_failures, local_group_mapping)


parser = get_parser()
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ru_wandb_logger.setLevel(logging.INFO)
    args = parser.parse_args()

    logger.info("Parsed arguments: run_id=%s, project=%s, entity=%s", args.run_id, args.project, args.entity)

    # If project is a path split it
    first_arg_path = Path(args.project)
    # fork_file_present
    if args.command == "patch":
        # Patch an incomplete offline history JSON file using WandB online data.

        # do we have a json file
        if first_arg_path.exists():
            offline_json_path = first_arg_path
            project = None
        else:
            # we got passed a project
            project = args.project
            assert args.run_id is not None, "run_id is required for patch command if no file is given."
            # Find json file in experiment path
            if ExperimentKey.RUN_ID_SEPARATOR in args.run_id:
                args.experiment_key = args.experiment_key or args.run_id
                args.run_id = args.run_id.split(ExperimentKey.RUN_ID_SEPARATOR, 1)[0]
                single_experiment = args.experiment_key
            else:
                if not args.experiment_key:
                    raise ValueError(
                        f"experiment_key is required if run_id does not contain the RUN_ID_SEPARATOR '{ExperimentKey.RUN_ID_SEPARATOR}', "  # noqa: E501
                        "For non-forked runs pass the --experiment_key explicitly",
                    )
                single_experiment = args.experiment_key
            group_glob = "*"
            output_dir, offline_results = find_experiment_dir(
                args.experiment_path,
                "*"
                + args.run_id
                + (
                    f"/**/result*{single_experiment}.json"
                    if ExperimentKey.FORK_SEPARATOR in single_experiment
                    else "result.json"
                ),
                project=project,
                group_glob=group_glob,
            )
            assert len(offline_results) == 1, (
                f"Could not find unique offline results for run_id {args.run_id} "
                f"and experiment_key {single_experiment} in {output_dir}."
            )
            offline_json_path = offline_results[0]

        # Try to infer project and run_id from filename if not given
        experiment_key = args.experiment_key
        if project is None or experiment_key is None:
            # Example filename: offline-run-<run_id>-<project>.json or similar
            stem = offline_json_path.stem
            # Try to extract run_id and project from filename
            # Accepts: offline-run-<run_id>-<project>
            if not experiment_key and offline_json_path.name != "results.json" and "-" in stem:
                parts = stem.split("-")
                experiment_key = parts[-1]
            # If still not found, try parent directory
            if project is None:
                project = offline_json_path.parent.name.rsplit("-", 1)[0]
            if experiment_key is None:
                experiment_key = offline_json_path.parent.name.split("-")[-1]
        patch_offline_history(
            offline_path=offline_json_path,
            entity=args.entity,
            project=project,
            experiment_key=experiment_key,
        )
        print(f"Patched offline history written to {offline_json_path.with_suffix('.patched.json')}")
        sys.exit(0)

    if first_arg_path.is_dir() and first_arg_path.exists():
        args.project = first_arg_path.name.rsplit("-", 1)[0]
        if args.project == "driver_artifacts":
            args.project = first_arg_path.parent.name.rsplit("-", 1)[0]
            args.run_id = first_arg_path.parent.name.split("-")[-1]
        else:
            args.run_id = first_arg_path.name.split("-")[-1]
        assert args.experiment_path == os.environ.get("RAY_UTILITIES_STORAGE_PATH", "./outputs/experiments"), (
            "When project is a path, experiment_path must be in RAY_UTILITIES_STORAGE_PATH, ./outputs/experiments"
        )  # noqa: E501
        args.experiment_path = str(first_arg_path.parent)
        fork_file_present = bool(list(first_arg_path.glob("pbt_fork_data*")))
        if fork_file_present:
            args.no_monitor = True
        # New directory format that differs from wanb project names
    elif first_arg_path.suffix == ".yaml":
        assert first_arg_path.exists(), f"YAML file {first_arg_path} does not exist."
    assert args.run_id is not None or (args.command == "verify" and first_arg_path.suffix == ".yaml"), (
        "run_id is required - or for verify command, use a submissions.yaml file."
    )
    from ray_utilities.callbacks.wandb import WandbUploaderMixin

    uploader = WandbUploaderMixin()
    uploader.project = args.project
    if args.command == "verify":
        # Allow a submissions.yaml file to be used
        if first_arg_path.suffix == ".yaml":
            logger.info(
                "Using submissions.yaml file for verification: %s. "
                "Check that RAY_UTILITIES_STORAGE_PATH: '%s' points to the right location",
                args.experiment_path,
                os.environ.get("RAY_UTILITIES_STORAGE_PATH", ""),
            )
            # Read single experiment keys from the yaml file
            failures: FailureDictType = {}
            group_mapping = {}
            last_experiment = None

            jobs = list(
                experiment_ids_from_submissions_yaml(
                    first_arg_path,
                    args.run_id,
                    skip_checked=args.skip_checked,
                )
            )
            batch_size = 4
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_job = {}
                job_iter = iter(jobs)
                # Start initial batch
                for _ in range(batch_size):
                    try:
                        job = next(job_iter)
                        future = executor.submit(_verify_one, *job)
                        future_to_job[future] = job
                    except StopIteration:  # noqa: PERF203
                        break
                while future_to_job:
                    done, _ = concurrent.futures.wait(
                        future_to_job.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    local_failures = {}  # just to be bound
                    for future in done:
                        try:
                            project, group_name, experiment_id, local_failures, local_group_mapping = future.result()
                        except Exception as e:
                            logger.error("Error occurred while verifying job %s: %s", future_to_job[future], e)
                            continue
                        group_mapping.update(local_group_mapping)
                        if not local_failures:
                            local_failures = {experiment_id: "none"}
                        failures.update(local_failures)
                        if last_experiment is None:
                            last_experiment = experiment_id
                        if last_experiment != experiment_id:
                            logger.info(
                                "Finished verification for project {%s}/{%s}/{%s}. "
                                "Writing results and moving to next group {%s}.\n",
                                group_name,
                                project,
                                last_experiment,
                                group_name,
                            )
                            _write_failures_to_submission_file(
                                first_arg_path,
                                failures,
                                {
                                    run_id: mapping
                                    for run_id, mapping in group_mapping.items()
                                    if mapping["experiment_id"] == experiment_id
                                },
                                no_failures=[
                                    exp_id
                                    for exp_id, failures in local_failures.items()
                                    if not failures or failures == "none"
                                ],
                            )
                            last_experiment = experiment_id
                        # Remove completed future
                        del future_to_job[future]
                        # Submit next job if available
                        try:
                            job = next(job_iter)
                            next_future = executor.submit(_verify_one, *job)
                            future_to_job[next_future] = job
                        except StopIteration:
                            pass
                    # write last
                    _write_failures_to_submission_file(
                        first_arg_path,
                        failures,
                        {
                            run_id: mapping
                            for run_id, mapping in group_mapping.items()
                            if mapping["experiment_id"] == last_experiment
                        },
                        no_failures=[
                            exp_id for exp_id, failures in local_failures.items() if not failures or failures == "none"
                        ],
                    )
            # Check which runs where not uploaded yet:
            for run, failure in failures.items():
                if isinstance(failure, Exception):
                    print(f"Error during verification for {group_mapping.get(run.id, {})} {run.id}: {failure}")
                elif isinstance(run, RunNotFound) or (
                    len(failure) > 0 and failure[0].type == VerificationFailure.NO_ONLINE_RUN_FOUND
                ):
                    print(f"Run not found, not uploaded yet: {group_mapping.get(run.id, {})} {run.id}")
                elif len(failure):
                    print(f"Verification failures for {group_mapping.get(run.id, {})} {run.id}: {failure}")
            _write_failures_to_submission_file(first_arg_path, failures, group_mapping)
        else:
            if ExperimentKey.RUN_ID_SEPARATOR in args.run_id:
                args.experiment_key = args.experiment_key or args.run_id
                args.run_id = args.run_id.split(ExperimentKey.RUN_ID_SEPARATOR, 1)[0]
            failures = uploader.verify_wandb_uploads(
                experiment_id=args.run_id, output_dir=args.experiment_path, single_experiment=args.experiment_key
            )
        success = not failures or all(
            not (isinstance(run, RunNotFound) or isinstance(failure, Exception) or any(not f.minor for f in failure))
            for run, failure in failures.items()
        )
        sys.exit(0 if success else 1)
    try:
        experiment_dir = Path(args.experiment_path)
        if args.experiment_key:
            glob_pattern = f"*{args.run_id}/**/wandb/*run-*-{args.experiment_key}*"
        else:
            glob_pattern = f"*{args.run_id}/**/wandb/offline-run-*"
        wandb_paths = experiment_dir.glob(glob_pattern)
        wandb_paths = list(wandb_paths)
        if not wandb_paths:
            print(
                "No WandB paths found for the given run ID in",
                experiment_dir,
                "glob:",
                glob_pattern,
            )
            sys.exit(1)
        print("Found", len(wandb_paths), "WandB paths to upload.")
        if not args.no_monitor:
            uploader._start_monitor_safe(args.project, entity=args.entity)
        if args.first_verify:
            print("Performing initial verification before upload...")
            try:
                failures = uploader.verify_wandb_uploads(
                    experiment_id=args.run_id,
                    output_dir=args.experiment_path,
                    single_experiment=args.experiment_key,
                    verbose=2,
                    run_per_page=64,
                )
            except KeyboardInterrupt:
                print("Verification interrupted by user.")
            else:
                # Skip those paths that did not report an error
                wandb_paths = [
                    path
                    for path in wandb_paths
                    # skip minor failures as likely offline data is missing
                    if any(
                        isinstance(failure, Exception) or any(not f.minor for f in failure)
                        for run, failure in failures.items()
                        if run.id in str(path)
                    )
                ]
        uploader.upload_paths(
            wandb_paths=wandb_paths, use_tqdm=True, wait=True, skip_synced=False, wandb_args=args.wandb_args
        )
        print("All project uploads done. Verifying...")
        try:
            time.sleep(30)
            uploader.verify_wandb_uploads(
                experiment_id=args.run_id, output_dir=args.experiment_path, single_experiment=args.experiment_key
            )
        except KeyboardInterrupt:
            print("Verification interrupted by user.")
    finally:
        print("Cleanup do not interrupt.")
        if not args.no_monitor and ray.is_initialized():
            uploader._stop_monitor()
        if ray.is_initialized():
            ray.shutdown()
