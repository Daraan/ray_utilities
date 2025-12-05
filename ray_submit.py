"""
A script to submit and monitor Ray jobs in bulk based on a YAML configuration file.
Uses the Ray Job Submission API to submit jobs, track their logs, and update their status in the YAML file.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import re
import shlex
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import AsyncIterator, Collection, cast

from ray.job_submission import JobStatus, JobSubmissionClient

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

CLIENT: JobSubmissionClient | None = None
JOB_END_STATES = {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED, "UNKNOWN (Deleted)"}
IGNORE_KEYS = {"comment"}

TIMESTAMP_SUFFIX = time.strftime("%Y-%m-%d_%H:%M:%S")
"""Suffix used for repetition of job IDs to avoid name clashes."""

AUTOSCALER_WARNING_PATTERN = re.compile(
    r"\(autoscaler [^\)]*\) Warning: The following resource request cannot be scheduled right now: .*? This is likely due to all cluster resources being claimed by actors\.\s*Consider creating fewer actors or adding more nodes to this Ray cluster\.\s*"  # noqa: E501
)

EXCLUDE_WDIR_FILES = [
    ".git",
    ".vscode",
    "docs",
    "outputs",
    "default_arguments",
    ".github",
    "get_ray_address.py",
    "ray_submit.py",
    "test",
]


def resolve_substitution_value(value: str | list[str], yaml_data: dict) -> list[str]:
    """
    Resolve a substitution value that may reference a global entry.

    Args:
        value: Either a list of strings or a string key reference (e.g., "<mujoco_envs>")
        yaml_data: The full YAML data to look up global references

    Returns:
        A list of resolved string values
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        key = value
        if key not in yaml_data:
            raise KeyError(f"Global reference '{key}' not found in YAML data")
        return yaml_data[key]

    return [value] if isinstance(value, str) else value


def get_submissions(
    group: str,
    *,
    file,
    failed_only: bool = False,
    ignore_node_ids: Collection[str] = (),
    ignore_hostnames: Collection[str] = (),
    excludes: list[str] | None = None,
) -> dict[str, dict]:
    """
    Get job submissions for a group.

    Returns:
        Dictionary mapping job_id to settings dict. Settings dict now includes a 'group' key.
    """
    with open(file, "r") as f:
        data = yaml_load(f)
    group_data = data.get(group)
    if group_data is None:
        raise KeyError(f"Group '{group}' not found in the submissions file.")
    if "entrypoint_pattern" not in group_data:
        return {k: {**v, "group": group} for k, v in data[group].items() if k not in IGNORE_KEYS}

    entrypoint_pattern_config = group_data["entrypoint_pattern"]

    if isinstance(entrypoint_pattern_config, str):
        pattern = entrypoint_pattern_config
        substitutions = {}
    elif isinstance(entrypoint_pattern_config, dict):
        pattern = entrypoint_pattern_config.get("pattern")
        substitutions = entrypoint_pattern_config.get("substitutions", {})

        if not pattern:
            raise ValueError(f"Dictionary entrypoint_pattern in group '{group}' must have a 'pattern' key")

        if pattern.startswith("<") or not pattern.strip().startswith("python"):
            pattern_key = pattern if pattern in data else f"<{pattern}>" if f"<{pattern}>" in data else pattern
            if pattern_key not in data:
                raise KeyError(f"Pattern reference '{pattern}' not found in global YAML data")
            pattern = data[pattern_key]
    else:
        raise TypeError(f"entrypoint_pattern must be either a string or dict, got {type(entrypoint_pattern_config)}")

    # NOTE: For hostname/node_id selector removes the string wrapping
    parts = shlex.split(pattern)
    env_ignore_nodes = os.environ.get("SUBMIT_IGNORE_NODES", "").split(",")
    env_ignore_hostnames = os.environ.get("SUBMIT_IGNORE_HOSTNAMES", "").split(",")
    ignore_nodes = set(ignore_node_ids).union({n for n in env_ignore_nodes if n})
    ignore_hostnames = set(ignore_hostnames).union({h for h in env_ignore_hostnames if h})
    try:
        pbt_index = parts.index("pbt")
    except ValueError:
        pbt_index = -1
    for arg, ignores in zip(["--node_id_selector", "--hostname_selector"], [ignore_nodes, ignore_hostnames]):
        if not ignores:
            # still need to take care of ' removal
            continue
        if arg not in parts:
            parts[pbt_index:pbt_index] = [arg, "'!in(" + ",".join(ignores) + ")'"]
        elif not (selector := parts[parts.index(arg) + 1]).startswith("!"):
            raise ValueError("Cannot combine positive hostname selector with ignoring hostnames.", selector)
        elif "!in(" in selector:
            for label in ignores:
                selector = selector.replace("in(", "in(" + label + ",")
            parts[parts.index(arg) + 1] = selector
        else:
            # single hostname put with others into !in(...)
            ignores.add(selector.lstrip("!"))
            parts[parts.index(arg) + 1] = "'!in(" + ",".join(ignores) + ")'"
    for arg in ("--node_id_selector", "--hostname_selector"):
        if arg not in parts:
            continue
        selector_idx = parts.index(arg) + 1
        selector = parts[selector_idx]
        if selector[0] not in ("'", '"'):
            selector = "'" + selector
        if selector[-1] not in ("'", '"'):
            selector += "'"
        parts[selector_idx] = selector
    pattern = " ".join(parts)

    # Parse optional substitution keys with defaults (format: <KEY:default_value>)
    optional_subs = {}
    for match in re.finditer(r"<([^>:]+):([^>]*)>", pattern):
        key_name = match.group(1)
        default_value = match.group(2)
        full_key = f"<{key_name}:>"
        optional_subs[full_key] = default_value

    # Resolve and apply substitutions from the substitutions dict
    # Handle global references in substitution values
    resolved_substitutions = {}
    for sub_key, sub_value in substitutions.items():
        if isinstance(sub_value, str) and sub_value.startswith("<") and sub_value.endswith(">"):
            # This is a global reference - resolve it and add to other_keys instead
            resolved_values = resolve_substitution_value(sub_value, data)
            # Add the target key with resolved values to group_data for later processing
            if sub_key not in group_data:
                group_data[sub_key] = resolved_values
        else:
            # Direct substitution value - apply to pattern immediately
            resolved_substitutions[sub_key] = sub_value

    # Apply direct substitutions to pattern
    for sub_key, sub_value in resolved_substitutions.items():
        if sub_key in pattern:
            if isinstance(sub_value, str):
                pattern = pattern.replace(sub_key, sub_value)
        else:
            # Also replace keys with defaults, e.g. <NUM_ENVS:3>
            for key_with_default in re.findall(r"<([^>:]+):[^>]*>", pattern):
                sub_key_plain = f"<{key_with_default}>"
                if sub_key_plain in resolved_substitutions:
                    pattern = re.sub(
                        rf"<{key_with_default}:[^>]*>",
                        resolved_substitutions[sub_key_plain],
                        pattern,
                    )

    # Replace optional substitutions with their defaults if not already substituted
    for opt_key, default_val in optional_subs.items():
        if opt_key in pattern:
            pattern = pattern.replace(opt_key, default_val)
    # Resolve all linebreaks in the pattern to be a single line
    pattern = re.sub(r"\s*\n\s*", " ", pattern)
    pattern = re.sub(r"\s+", " ", pattern).strip()

    # Find all replacement keys in the pattern (keys like <ENV_TYPE>)
    other_keys: dict[str, list[str]] = {
        k: v for k, v in group_data.items() if k not in IGNORE_KEYS and k.startswith("<") and k.endswith(">")
    }
    # Build all combinations of replacement values for keys present in the pattern
    replace_keys = [k for k in other_keys if k in pattern]
    if not replace_keys:
        raise ValueError(
            f"No replacement keys found in the entrypoint pattern.\n"
            f"Pattern: {pattern}\n"
            f"Available keys in group_data: {list(other_keys.keys())}\n"
            f"Substitutions applied: {substitutions}"
        )

    # Resolve replacement values (may reference global entries)
    replace_values_lists = []
    for k in replace_keys:
        resolved_values = resolve_substitution_value(other_keys[k], data)
        filtered_values = [v for v in resolved_values if v and not v.startswith("#")]
        replace_values_lists.append(filtered_values)

    submissions = {}
    all_combinations = list(itertools.product(*replace_values_lists))

    if failed_only:
        run_ids_data = group_data.get("run_ids", {})
        if not run_ids_data:
            return {}

        failed_combinations = set()
        for combo in all_combinations:
            run_key = "(" + ", ".join(combo).rstrip(", ") + ")"
            has_failed = False
            if run_key in run_ids_data:
                for run_info in run_ids_data[run_key].values():
                    if isinstance(run_info, dict):
                        if run_info.get("status") == "FAILED":
                            has_failed = True
                        # If there is a success after a failure, do not consider it failed
                        if has_failed and run_info.get("status") == JobStatus.SUCCEEDED.value:
                            has_failed = False
            else:
                # declare missing as failed as well
                has_failed = True
            if has_failed:
                failed_combinations.add(combo)
        all_combinations = [combo for combo in all_combinations if combo in failed_combinations]

    run_ids_data = group_data.get("run_ids", {})
    for values in all_combinations:
        job_id = "_".join(values)
        entry = dict(zip(replace_keys, values))
        entrypoint = pattern
        for k, v in entry.items():
            entrypoint = entrypoint.replace(k, v)

        # Exclude if entrypoint matches any exclude pattern
        if excludes:
            if any(excl in entrypoint or excl in job_id for excl in excludes):
                continue

        # Check run_ids for this combination: skip if any SUCCEEDED
        run_key = "(" + ", ".join(values).rstrip(", ") + ")"
        already_succeeded = False
        if run_key in run_ids_data:
            for run_info in run_ids_data[run_key].values():
                if isinstance(run_info, dict) and run_info.get("status") == JobStatus.SUCCEEDED.value:
                    already_succeeded = True
                    break
        if already_succeeded:
            continue

        remaining_placeholders = re.findall(r"<[^>]+>", entrypoint)
        if remaining_placeholders:
            raise ValueError(
                f"Unresolved placeholders in entrypoint pattern: {remaining_placeholders}\n"
                f"Pattern: {pattern}\n"
                f"After substitution: {entrypoint}"
            )

        submissions[job_id] = {
            "entrypoint": entrypoint,
            "group": group,
            **{
                k: v
                for k, v in group_data.items()
                if k not in ("entrypoint_pattern", *IGNORE_KEYS) and not (k.startswith("<") and k.endswith(">"))
            },
        }
    return submissions


def is_valid_group(group_data: dict) -> bool:
    """Check if a group has a valid entrypoint_pattern configuration."""
    return isinstance(group_data, dict) and "entrypoint_pattern" in group_data


def get_environment_types(group_data: dict, yaml_data: dict) -> list[str]:
    """
    Extract environment types needed for a group.

    Args:
        group_data: The group configuration dictionary
        yaml_data: The full YAML data for resolving references

    Returns:
        List of environment type values needed for this group
    """
    entrypoint_pattern_config = group_data["entrypoint_pattern"]

    # Find <ENV_TYPE> in substitutions or group keys
    env_type_key = "<ENV_TYPE>"

    # Check group-level keys first (higher priority)
    if env_type_key in group_data:
        return resolve_substitution_value(group_data[env_type_key], yaml_data)

    # Check substitutions
    if isinstance(entrypoint_pattern_config, dict):
        substitutions = entrypoint_pattern_config.get("substitutions", {})
        if env_type_key in substitutions:
            return resolve_substitution_value(substitutions[env_type_key], yaml_data)

    return []


def should_submit_group(group_name: str, group_data: dict, yaml_data: dict) -> bool:
    """
    Determine if a group should have submissions based on run_ids status.

    Args:
        group_name: Name of the group
        group_data: The group configuration dictionary
        yaml_data: The full YAML data

    Returns:
        True if the group needs submissions (no run_ids or missing/failed environments)
    """
    run_ids_data = group_data.get("run_ids")

    # If no run_ids section or run_ids is None/empty dict, submit all
    if not run_ids_data:
        return True

    # Get required environments
    env_types = get_environment_types(group_data, yaml_data)

    # If no environment types defined, always submit
    if not env_types:
        return True

    # Check each environment for at least one SUCCEEDED entry
    for env_type in env_types:
        # Look for run_key containing this environment
        env_has_success = False
        env_found = False

        for run_key, runs in run_ids_data.items():
            # Check if this run_key contains the environment
            # run_key format: "(<val1>, <val2>, ...)"
            if env_type in run_key:
                env_found = True
                # Check if any run for this environment succeeded
                if runs:
                    for run_info in runs.values():
                        if isinstance(run_info, dict) and run_info.get("status") == JobStatus.SUCCEEDED.value:
                            env_has_success = True
                            break

            if env_has_success:
                break

        # If environment not found in run_ids or has no success, need to submit
        if not env_found or not env_has_success:
            return True

    # All environments have at least one success
    return False


def get_all_submissions(
    *,
    file,
    failed_only: bool = False,
    ignore_node_ids: Collection[str] = (),
    ignore_hostnames: Collection[str] = (),
    excludes: list[str] | None = None,
) -> dict[str, dict]:
    """
    Get submissions from all valid groups in the YAML file.

    Returns:
        Dictionary mapping job_id to settings dict (including 'group' key)
    """
    with open(file, "r") as f:
        yaml_data = yaml_load(f)

    all_submissions = {}

    for group_name, group_data in yaml_data.items():
        # Skip non-group entries and invalid groups
        if not is_valid_group(group_data):
            continue

        # Check if this group should be submitted
        if not should_submit_group(group_name, group_data, yaml_data):
            print(f"Skipping group '{group_name}': all environments have succeeded")
            continue

        try:
            group_submissions = get_submissions(
                group_name,
                file=file,
                failed_only=failed_only,
                ignore_node_ids=ignore_node_ids,
                ignore_hostnames=ignore_hostnames,
                excludes=excludes,
            )

            # Prefix job_id with group name to avoid collisions
            for job_id, settings in group_submissions.items():
                prefixed_job_id = f"{group_name}_{job_id}"
                if excludes and any(excl in prefixed_job_id for excl in excludes):
                    continue
                all_submissions[prefixed_job_id] = settings

        except Exception as e:
            print(f"Warning: Failed to get submissions for group '{group_name}': {e}")
            continue

    return all_submissions


def deep_update(original: dict, updates: dict) -> dict:
    """
    Recursively update a dictionary with another dictionary.
    Nested dictionaries are merged, not replaced.

    Args:
        original: The dictionary to update.
        updates: The dictionary with updates.

    Returns:
        The updated dictionary.
    """
    for key, value in updates.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def write_back(group: str, job_id: str, run_id: str | dict[str, str | dict[str, str]], *, file: str | Path):
    file_path = Path(file)
    lock_file = file_path.with_suffix(file_path.suffix + ".lock")

    # Wait for lock to be released
    timeout = 60  # seconds
    start_time = time.time()
    while lock_file.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Could not acquire lock on {file_path} after {timeout} seconds. Lock file {lock_file} still exists."
            )
        print(f"Waiting for lock file {lock_file} to be released...")
        time.sleep(1)

    try:
        # Acquire lock
        lock_file.touch()

        with open(file_path, "r") as f:
            data = yaml_load(f)
        job_id = job_id.removesuffix(TIMESTAMP_SUFFIX)

        # If group is "all", try to extract the actual group from job_id prefix or from the submission data
        actual_group = group
        if group == "all":
            # Try to get the group from the job_id prefix
            if "_" in job_id:
                possible_group = job_id.split("_")[0]
                if possible_group in data:
                    actual_group = possible_group
            # If not found, try to get the group from the submission data
            if actual_group == "all":
                # Try to find the group from the job's data if present
                for g in data:
                    if job_id in data[g]:
                        actual_group = g
                        break

        if "entrypoint_pattern" not in data.get(actual_group, {}):
            data[actual_group][job_id].setdefault("run_ids", {})
            if isinstance(run_id, dict):
                for experiment_id, run_info in run_id.items():
                    if isinstance(run_info, dict):
                        if (
                            _current_status := data[actual_group][job_id]["run_ids"]
                            .get(experiment_id, {})
                            .get("status", JobStatus.RUNNING)
                        ) not in (JobStatus.RUNNING, JobStatus.PENDING, *JOB_END_STATES):
                            run_id = run_id.copy()
                            run_id[experiment_id] = run_info = run_info.copy()  # noqa: PLW2901
                            run_info.pop("status", None)
                deep_update(data[actual_group][job_id]["run_ids"], run_id)
            else:
                data[actual_group][job_id]["run_ids"][run_id] = "RUNNING"
        else:
            data[actual_group].setdefault("run_ids", {})
            replacement_parts = job_id.removeprefix(actual_group + "_").split("_")
            run_key = "(" + ", ".join(replacement_parts).rstrip(", ") + ")"
            if isinstance(run_id, dict):
                data[actual_group]["run_ids"].setdefault(run_key, {}).update(run_id)
            else:
                data[actual_group]["run_ids"].setdefault(run_key, {})[run_id] = "RUNNING"
        with open(file_path, "w") as f:
            yaml_dump(data, f)
    finally:
        # Release lock
        if lock_file.exists():
            os.remove(lock_file)


def wait_until_status(job_id, status_to_wait_for, timeout_seconds=5, client: JobSubmissionClient | None = None):
    # copied from ray documentation
    client = client or CLIENT
    assert client is not None, "A JobSubmissionClient must be provided either via argument or global CLIENT."
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(1)


def get_tmux_log_command(job_id: str) -> str:
    ray_command = f"ray job logs {job_id} -f"
    tmux_command = (
        f'tmux new-session -s "{job_id}" -n "{job_id}" -d '
        f"\"bash -c 'source ../env/bin/activate && {ray_command}; exec bash'\""
    )
    return tmux_command


_submission_counter = 0


def submit_single_job(job_id: str, settings: dict, args: argparse.Namespace) -> str | None:
    if args.test:
        global _submission_counter  # noqa: PLW0603
        _submission_counter += 1
        settings = settings.copy()
        settings.pop("run_ids", None)
        settings.pop("group", None)
        print(f"\n-----------------------\nTest mode: would submit job {job_id} with settings:\n{pformat(settings)}")
        return None

    print(f"Submitting job: {job_id}")
    assert CLIENT

    # Extract group from settings
    group = settings.pop("group", args.group)
    try:
        submission_id_out = CLIENT.submit_job(
            entrypoint=settings["entrypoint"],
            submission_id=settings.get(
                "submission_id", group + "_" + job_id.removeprefix(group + "_") + "_" + TIMESTAMP_SUFFIX
            ),
            runtime_env=settings.get(
                "runtime_env",
                {"working_dir": ".", "excludes": EXCLUDE_WDIR_FILES},
            ),
            entrypoint_num_cpus=settings.get("entrypoint_num_cpus", 0.33),
            entrypoint_num_gpus=settings.get("entrypoint_num_gpus", 0),
            # While most of the time the jobs do not need that much memory there is a spike at the end
            # possibly related to Wandb & Comet logging finalization.
            entrypoint_memory=int(settings.get("entrypoint_memory", 4.5 * 1000 * 1000 * 1000)),
            entrypoint_resources=settings.get("entrypoint_resources", {"persistent_node": 1}),
            metadata=settings.get("metadata", None),
        )
        print(f"Submitted job {job_id} with job ID: {submission_id_out}")
        return submission_id_out  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        print(f"Failed to submit job {job_id}: {e}, settings: {settings}")
        return None


async def monitor_job_statuses(
    jobs_tracked: dict[str, tuple[str, str]],
    task_run_ids: dict[str, str],
    args: argparse.Namespace,
    pending_submissions: list[tuple[str, dict]] | None = None,
):
    """Monitors the statuses of jobs and writes back the final status."""
    assert CLIENT
    print("Performing only monitoring of job statuses...")
    loop = asyncio.get_running_loop()
    user_input = AsyncInput(loop)
    try:
        jobs_tracked_left = jobs_tracked.copy()
        final_states: dict[str, JobStatus] = {}
        while jobs_tracked_left or (pending_submissions and len(pending_submissions) > 0):
            print("-" * 80)
            jobs_to_delete = []
            for job_id, group_and_job_id in jobs_tracked_left.items():
                try:
                    job_status = CLIENT.get_job_status(job_id)
                except RuntimeError as e:
                    # If we delete a job it does not show up anymore
                    if "does not exist" not in str(e):
                        raise
                    job_status = final_states.get(job_id, "UNKNOWN (Deleted)")
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Job {job_id} status: {job_status}.")
                if job_status in JOB_END_STATES:
                    job_status = cast("JobStatus", job_status)
                    final_states[job_id] = job_status
                    if run_id := task_run_ids.get(job_id):
                        group, original_job_id = group_and_job_id
                        write_back(
                            group,
                            original_job_id,
                            {run_id: {"status": job_status.name, "submission_id": job_id}},
                            file=args.submissions_file,
                        )
                    jobs_to_delete.append(job_id)
            for job_id in jobs_to_delete:
                jobs_tracked_left.pop(job_id, None)

            if len(jobs_tracked_left) < args.max_jobs and pending_submissions:
                submission_id_out = None
                while pending_submissions and submission_id_out is None:
                    next_job_id, next_settings = pending_submissions.pop(0)
                    print("No running jobs left. Automatically submitting next pending job:", next_job_id)
                    submission_id_out = submit_single_job(next_job_id, next_settings, args)
                    if submission_id_out:
                        # Extract group from settings
                        group = next_settings.get("group", args.group)
                        jobs_tracked_left[submission_id_out] = (group, next_job_id)

            if not jobs_tracked_left and not pending_submissions:
                break

            interval = 180
            if pending_submissions:
                print(
                    f"Pending submissions: {len(pending_submissions)} / Running {len(jobs_tracked_left)}. "
                    "Submit next job? (y/n): ",
                    end="",
                    flush=True,
                )
                input_future = user_input.start()
                try:
                    done, _ = await asyncio.wait([input_future], timeout=interval, return_when=asyncio.FIRST_COMPLETED)
                    if input_future in done:
                        result = input_future.result()
                        if result and result.strip().lower() == "y":
                            print("Submitting next job manually...")
                            next_job_id, next_settings = pending_submissions.pop(0)
                            submission_id_out = submit_single_job(next_job_id, next_settings, args)
                            if submission_id_out:
                                # Extract group from settings
                                group = next_settings.get("group", args.group)
                                jobs_tracked_left[submission_id_out] = (group, next_job_id)
                        print("\r\033[K", end="", flush=True)
                    else:
                        input_future.cancel()
                        print()
                finally:
                    user_input.stop()
            else:
                await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n\n\n########################## Second Keyboard Interrupt Detected #########################\n\n\n\n")
        print("Exiting monitoring.")
        sys.exit(1)


class AsyncInput:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue = asyncio.Queue()
        self.fd = sys.stdin.fileno()
        self.future = None

    def _on_input(self):
        try:
            line = sys.stdin.readline()
            if line:
                if self.future and not self.future.done():
                    self.future.set_result(line)
        except Exception as e:  # noqa: BLE001
            if self.future and not self.future.done():
                self.future.set_exception(e)

    def start(self):
        self.future = self.loop.create_future()
        self.loop.add_reader(self.fd, self._on_input)
        return self.future

    def stop(self):
        self.loop.remove_reader(self.fd)
        if self.future and not self.future.done():
            self.future.cancel()


if __name__ == "__main__":
    os.environ["RAY_UTILITIES_NO_TQDM"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "group",
        nargs="?",
        help="The group key in the yaml file to run, 'all' for all groups, or 'monitor'.",
        type=str,
    )
    parser.add_argument("submissions_file", help="The submissions yaml file.")
    parser.add_argument("monitor_group", nargs="*")
    parser.add_argument(
        "--address",
        type=str,
        help="The address of the Ray cluster.",
        default="http://" + os.environ.get("DASHBOARD_ADDRESS", "localhost:8265"),
    )
    parser.add_argument("--test", action="store_true", help="If set, runs in test mode without submitting jobs.")
    parser.add_argument(
        "--failed-only", action="store_true", help="If set, only submits jobs that have previously failed."
    )
    parser.add_argument(
        "--ignore_hostnames",
        nargs="*",
        default=[],
        help="Hostnames to ignore when submitting jobs.",
    )
    parser.add_argument(
        "--ignore_nodes",
        nargs="*",
        default=[],
        help="Node IDs to ignore when submitting jobs.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=5,
        help="Maximum number of concurrent jobs to run.",
    )
    parser.add_argument(
        "--excludes",
        nargs="+",
        default=[],
        help="Exclude submissions whose entrypoint contains any of these patterns.",
    )

    args = parser.parse_args()
    assert args.address, (
        "Please provide the Ray cluster address via --address or DASHBOARD_ADDRESS environment variable."
    )

    try:
        CLIENT = JobSubmissionClient(args.address)
    except ConnectionError as e:
        print(f"Failed to create JobSubmissionClient: {e}")
        if not args.test:
            import sys

            sys.exit(1)
        CLIENT = None

    jobs_tracked: dict[str, AsyncIterator[str]] = {}
    # For monitor mode, this will be dict[submission_id, tuple[group, original_job_id]]
    # For submission mode, this will be dict[submission_id, tuple[group, original_job_id]]
    monitor_jobs_tracked: dict[str, tuple[str, str]] = {}
    task_run_ids = {}
    pending_submissions: list[tuple[str, dict]] = []

    if args.group == "monitor":
        monitor_groups = set(args.monitor_group)
        print("Monitor mode: Scanning for running jobs...")
        with open(args.submissions_file, "r") as f:
            data = yaml_load(f)
        for group_name, group_data in data.items():
            if monitor_groups and group_name not in monitor_groups:
                continue
            if not isinstance(group_data, dict) or "run_ids" not in group_data:
                continue
            run_ids_data = group_data.get("run_ids", {})
            for run_key, runs in run_ids_data.items():
                for run_id, run_info in runs.items():
                    if isinstance(run_info, dict) and run_info.get("status") == "RUNNING":
                        submission_id = run_info.get("submission_id")
                        if not submission_id:
                            continue
                        print(f"Found running job: group={group_name}, run_id={run_id}, submission_id={submission_id}")
                        # Reconstruct original job_id for write_back
                        # run_key is like '(<val1>, <val2>)'
                        replacement_parts = [p.strip() for p in run_key.strip("()").split(",") if p.strip()]
                        original_job_id = "_".join(replacement_parts)
                        monitor_jobs_tracked[submission_id] = (group_name, original_job_id)
                        task_run_ids[submission_id] = run_id
        if not monitor_jobs_tracked:
            print("No running jobs found to monitor.")
            sys.exit(0)

    else:
        assert args.group, "A group must be specified if not in monitor mode."

        if args.group == "all":
            submissions_dict = get_all_submissions(
                file=args.submissions_file,
                failed_only=args.failed_only,
                ignore_node_ids=args.ignore_nodes,
                ignore_hostnames=args.ignore_hostnames,
                excludes=args.excludes,
            )
        else:
            submissions_dict = get_submissions(
                args.group,
                file=args.submissions_file,
                failed_only=args.failed_only,
                ignore_node_ids=args.ignore_nodes,
                ignore_hostnames=args.ignore_hostnames,
                excludes=args.excludes,
            )
        submissions = list(submissions_dict.items())

        if not args.test and args.max_jobs is not None and args.max_jobs > 0:
            pending_submissions = submissions[args.max_jobs :]
            submissions = submissions[: args.max_jobs]

        for job_id, settings in submissions:
            submission_id_out = submit_single_job(job_id, settings, args)
            if submission_id_out:
                jobs_tracked[submission_id_out] = CLIENT.tail_job_logs(submission_id_out)  # pyright: ignore[reportOptionalMemberAccess]
                # Extract group from settings or use args.group as fallback
                group = settings.get("group", args.group)
                monitor_jobs_tracked[submission_id_out] = (group, job_id)
                time.sleep(10)

    if args.test:
        import sys

        print(f"\n-----------------------\nTest mode: would submit {_submission_counter} jobs.")
        sys.exit(0)
    assert CLIENT

    if not jobs_tracked and not monitor_jobs_tracked:
        print("No jobs to track or monitor.")
        sys.exit(0)

    if args.group == "monitor":
        asyncio.run(monitor_job_statuses(monitor_jobs_tracked, task_run_ids, args))
        sys.exit(0)

    async def gather_and_print_job_outputs(
        jobs_tracked: dict[str, AsyncIterator[str]],
        interval: float = 5.0,
        pending_submissions: list[tuple[str, dict]] | None = None,
    ):
        assert CLIENT

        async def get_next(aiterator):
            return await aiterator.__anext__()

        def submit_next():
            assert CLIENT
            while pending_submissions:
                next_job_id, next_settings = pending_submissions.pop(0)
                submission_id_out = submit_single_job(next_job_id, next_settings, args)
                if submission_id_out:
                    jobs_tracked[submission_id_out] = CLIENT.tail_job_logs(submission_id_out)
                    # Extract group from settings
                    group = next_settings.get("group", args.group)
                    monitor_jobs_tracked[submission_id_out] = (group, next_job_id)
                    async_read_tasks[submission_id_out] = asyncio.create_task(get_next(jobs_tracked[submission_id_out]))
                    last_outputs[submission_id_out] = []
                    return True
            return False

        last_outputs = {job_id: [] for job_id in jobs_tracked}
        async_read_tasks = {
            job_id: asyncio.create_task(get_next(aiterator)) for job_id, aiterator in jobs_tracked.items()
        }

        loop = asyncio.get_running_loop()
        user_input = AsyncInput(loop)

        last_tmux_print = time.time()
        while async_read_tasks:
            start = time.time()
            collected: dict[str, list[str]] = {job_id: [] for job_id in async_read_tasks}

            # Start input wait
            if pending_submissions:
                print(
                    f"Pending submissions: {len(pending_submissions)} / Running {len(async_read_tasks)}. "
                    "Submit next job? (y/n): ",
                    end="",
                    flush=True,
                )
                input_future = user_input.start()
            else:
                input_future = asyncio.Future()
                input_future.set_result("n")

            while time.time() - start < interval and async_read_tasks:
                wait_tasks = (
                    [*async_read_tasks.values(), input_future] if pending_submissions else [*async_read_tasks.values()]
                )
                done, _ = await asyncio.wait(
                    wait_tasks, timeout=interval - (time.time() - start), return_when=asyncio.FIRST_COMPLETED
                )

                if input_future in done:
                    try:
                        result = input_future.result()
                        if result and result.strip().lower() == "y":
                            print("Submitting next job manually...")
                            if not submit_next():
                                print("No more pending jobs.")
                        # Restart input wait
                        print(
                            f"Pending submissions: {len(pending_submissions or [])} / Running {len(async_read_tasks)}. "
                            "Submit next job? (y/n): ",
                            end="",
                            flush=True,
                        )
                        input_future = user_input.start()
                    except Exception:  # noqa: BLE001
                        pass

                for done_task in done:
                    if done_task == input_future:
                        continue

                    # Get job_id of the done task
                    for job_id, t in list(async_read_tasks.items()):
                        if t == done_task:
                            try:
                                output = done_task.result()
                                collected[job_id].append(output)
                                # save last 1000 lines
                                last_outputs[job_id] = [*last_outputs[job_id][-1000:], output]
                                async_read_tasks[job_id] = asyncio.create_task(get_next(jobs_tracked[job_id]))
                            except StopAsyncIteration:
                                async_read_tasks.pop(job_id, None)

                            try:
                                job_status = CLIENT.get_job_status(job_id)
                            except Exception as e:  # noqa: BLE001
                                print(f"Failed to get status for job {job_id}: {e}")
                                break

                            if job_status in JOB_END_STATES:
                                if run_id := task_run_ids.get(job_id):
                                    try:
                                        group, original_job_id = monitor_jobs_tracked[job_id]
                                        write_back(
                                            group,
                                            original_job_id,
                                            {
                                                run_id: {
                                                    "status": (
                                                        job_status.name if hasattr(job_status, "name") else job_status
                                                    ),
                                                    "submission_id": job_id,
                                                }
                                            },
                                            file=args.submissions_file,
                                        )
                                    except TimeoutError:
                                        pass
                                async_read_tasks.pop(job_id, None)

                                submit_next()
                            break
            # Print outputs for all jobs after interval
            for job_id, lines in collected.items():
                if lines:
                    filtered_lines = []
                    for line in lines:
                        if AUTOSCALER_WARNING_PATTERN.search(line):
                            line = AUTOSCALER_WARNING_PATTERN.sub("", line)  # noqa: PLW2901
                            if not line.strip():
                                continue
                        filtered_lines.append(line)
                    lines = filtered_lines  # noqa: PLW2901

                    if not lines:
                        continue

                    if job_id not in task_run_ids:
                        # Check line for Run ID: <run_id>
                        for line in lines:
                            if "Run ID:" in line:
                                # NOTE: Currently the ID always ends with 4 the version number
                                run_id_match = re.search(r"Run ID:\s*([a-zA-Z0-9]+)", line)
                                if run_id_match:
                                    run_id = run_id_match.group(1)
                                    task_run_ids[job_id] = run_id
                                    group, original_job_id = monitor_jobs_tracked[job_id]
                                    try:
                                        write_back(
                                            group,
                                            original_job_id,
                                            {run_id: {"status": "RUNNING", "submission_id": job_id}},
                                            file=args.submissions_file,
                                        )
                                    except TimeoutError:
                                        continue
                                    break
                    print(f"\n\n ============= Out: {job_id} =============\n\n")
                    print("".join(lines))
            if last_tmux_print + 240 < time.time():
                print("You can follow all jobs individually in separate tmux sessions using the following commands:")
                tmux_commands = [get_tmux_log_command(job_id) for job_id in jobs_tracked]
                print("\n".join(tmux_commands))
                last_tmux_print = time.time()

            user_input.stop()
            # Clear the prompt line
            print("\r\033[K", end="", flush=True)

        # Final output after all jobs are done
        for job_id, lines in last_outputs.items():
            print(f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Final Out: {job_id} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
            print("".join(lines[-1900 // len(last_outputs) :]))

    print("Starting job output in 10 seconds...")
    print("You can follow all jobs individually in separate tmux sessions using the following commands:")
    tmux_commands = [get_tmux_log_command(job_id) for job_id in jobs_tracked]
    print("\n".join(tmux_commands))
    time.sleep(10)

    try:
        # Run the async loop to gather and print outputs
        asyncio.run(gather_and_print_job_outputs(jobs_tracked, pending_submissions=pending_submissions))
    except KeyboardInterrupt:
        print("\n\n\n\n########################## Keyboard Interrupt Detected #########################\n\n\n\n")
        choice = input(
            "Stop all runs or exit log streaming only? "
            "Ctrl+C to exit streaming, "
            "Press any key to monitor statuses and submit next job when ready. "
            "'clear' to monitor but to not submit new jobs. "
            "'end' to stop all runs. "
            "'x2' to send two interrupts: "
        )
        if choice == "end":
            for job_id in jobs_tracked:
                print(f"Stopping job: {job_id}")
                CLIENT.stop_job(job_id)
            sys.exit(1)
        elif choice == "x2":
            for job_id in jobs_tracked:
                print(f"Stopping job: {job_id}")
                CLIENT.stop_job(job_id)
            time.sleep(0.5)
            for job_id in jobs_tracked:
                print(f"Stopping job: {job_id}")
                CLIENT.stop_job(job_id)
            sys.exit(1)
        if choice == "clear":
            pending_submissions = []
        asyncio.run(
            monitor_job_statuses(monitor_jobs_tracked, task_run_ids, args, pending_submissions=pending_submissions)
        )
