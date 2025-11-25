"""
A script to submit and monitor Ray jobs in bulk based on a YAML configuration file.
Uses the Ray Job Submission API to submit jobs, track their logs, and update their status in the YAML file.
"""

from __future__ import annotations

import sys
import argparse
import asyncio
import itertools
import os
import re
import time
from pprint import pformat
from typing import AsyncIterator, cast

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
JOB_END_STATES = {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}
IGNORE_KEYS = {"comment"}

RANDOM_SUFFIX = str(int(time.time()))[-4:]
"""Suffixed used for repetition of job IDs to avoid name clashes."""


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


def get_submissions(group: str, *, file, failed_only: bool = False) -> dict[str, dict]:
    with open(file, "r") as f:
        data = yaml_load(f)
    group_data = data.get(group)
    if group_data is None:
        raise KeyError(f"Group '{group}' not found in the submissions file.")
    if "entrypoint_pattern" not in group_data:
        return {k: v for k, v in data[group].items() if k not in IGNORE_KEYS}

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
    # TODO: This does not stay true to multiline strings
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
            if run_key in run_ids_data:
                for run_info in run_ids_data[run_key].values():
                    if isinstance(run_info, dict) and run_info.get("status") == "FAILED":
                        failed_combinations.add(combo)
                        break
        all_combinations = [combo for combo in all_combinations if combo in failed_combinations]

    for values in all_combinations:
        entry = dict(zip(replace_keys, values))
        entrypoint = pattern
        for k, v in entry.items():
            entrypoint = entrypoint.replace(k, v)

        # Validate that all placeholders have been replaced
        remaining_placeholders = re.findall(r"<[^>]+>", entrypoint)
        if remaining_placeholders:
            raise ValueError(
                f"Unresolved placeholders in entrypoint pattern: {remaining_placeholders}\n"
                f"Pattern: {pattern}\n"
                f"After substitution: {entrypoint}"
            )

        job_id = "_".join(values)
        submissions[job_id] = {
            "entrypoint": entrypoint,
            **{
                k: v
                for k, v in group_data.items()
                if k not in ("entrypoint_pattern", *IGNORE_KEYS) and not (k.startswith("<") and k.endswith(">"))
            },
        }
    return submissions


def write_back(group: str, job_id: str, run_id: str | dict[str, str | dict[str, str]], *, file):
    with open(file, "r") as f:
        data = yaml_load(f)
    job_id = job_id.removesuffix(RANDOM_SUFFIX)
    if "entrypoint_pattern" not in data[group]:
        data[group][job_id].setdefault("run_ids", {})
        if isinstance(run_id, dict):
            data[group][job_id]["run_ids"].update(run_id)
        else:
            data[group][job_id]["run_ids"][run_id] = "RUNNING"
    else:
        # Add a list of lists to a run_id list with the replace_keys as keys
        data[group].setdefault("run_ids", {})
        replacement_parts = job_id.removeprefix(group + "_").split("_")
        run_key = "(" + ", ".join(replacement_parts).rstrip(", ") + ")"
        if isinstance(run_id, dict):
            data[group]["run_ids"].setdefault(run_key, {}).update(run_id)
        else:
            data[group]["run_ids"].setdefault(run_key, {})[run_id] = "RUNNING"
    with open(file, "w") as f:
        yaml_dump(data, f)


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
    tmux_command = f'tmux new-session -s "{job_id}" -n "{job_id}" -d "bash -c \'source ../env/bin/activate && {ray_command}; exec bash\'"'
    return tmux_command


if __name__ == "__main__":
    if "RAY_UTILITIES_NO_TQDM" not in os.environ:
        if input("Warning: tqdm is not disabled. exit or continue (c)") != "c":
            sys.exit(0)
    os.environ["RAY_UTILITIES_NO_TQDM"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        help="The address of the Ray cluster.",
        default="http://" + os.environ.get("DASHBOARD_ADDRESS", "localhost:8265"),
    )
    parser.add_argument("group", help="The group key in the yaml file to run.", type=str)
    parser.add_argument(
        "submissions_file", nargs="?", default="experiments/submissions.yaml", help="The submissions yaml file."
    )
    parser.add_argument("--test", action="store_true", help="If set, runs in test mode without submitting jobs.")
    parser.add_argument(
        "--failed-only", action="store_true", help="If set, only submits jobs that have previously failed."
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
    finished_jobs: dict[str, JobStatus] = {}

    submissions = get_submissions(args.group, file=args.submissions_file, failed_only=args.failed_only).items()
    for job_id, settings in submissions:
        if args.test:
            print(
                f"\n-----------------------\nTest mode: would submit job {job_id} with settings:\n{pformat(settings)}"
            )
            continue
        print(f"Submitting job: {job_id}")
        assert CLIENT
        try:
            job_id_out = CLIENT.submit_job(
                entrypoint=settings["entrypoint"],
                submission_id=settings.get("submission_id", args.group + "_" + job_id + "_" + RANDOM_SUFFIX),
                runtime_env=settings.get("runtime_env", {"working_dir": "."}),
                entrypoint_num_cpus=settings.get("entrypoint_num_cpus", 0.66),
                entrypoint_num_gpus=settings.get("entrypoint_num_gpus", 0),
                entrypoint_memory=int(settings.get("entrypoint_memory", 4 * 1000 * 1000 * 1000)),
                entrypoint_resources=settings.get("entrypoint_resources", {"persistent_node": 1}),
                metadata=settings.get("metadata", None),
            )
        except Exception as e:
            print(f"Failed to submit job {job_id}: {e}, settings: {settings}")
            raise
        jobs_tracked[job_id_out] = CLIENT.tail_job_logs(job_id_out)
        print(f"Submitted job {job_id} with job ID: {job_id_out}")
        time.sleep(3)

    if args.test:
        import sys

        sys.exit(0)
    assert CLIENT

    task_run_ids = {}

    async def gather_and_print_job_outputs(jobs_tracked: dict[str, AsyncIterator[str]], interval: float = 5.0):
        assert CLIENT

        async def get_next(aiterator):
            return await aiterator.__anext__()

        last_outputs = {job_id: [] for job_id in jobs_tracked}
        tasks = {job_id: asyncio.create_task(get_next(aiterator)) for job_id, aiterator in jobs_tracked.items()}

        last_tmux_print = time.time()
        while tasks:
            start = time.time()
            collected: dict[str, list[str]] = {job_id: [] for job_id in tasks}
            while time.time() - start < interval and tasks:
                done, _ = await asyncio.wait(
                    tasks.values(), timeout=interval - (time.time() - start), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    for job_id, t in list(tasks.items()):
                        if t == task:
                            try:
                                output = task.result()
                                collected[job_id].append(output)
                                # save last 1000 lines
                                last_outputs[job_id] = [*last_outputs[job_id][-1000:], output]
                                tasks[job_id] = asyncio.create_task(get_next(jobs_tracked[job_id]))
                            except StopAsyncIteration:
                                del tasks[job_id]
                            break
                        job_status = CLIENT.get_job_status(job_id)
                        if job_status in JOB_END_STATES:
                            if run_id := task_run_ids.get(job_id):
                                write_back(
                                    args.group,
                                    job_id,
                                    {run_id: {"status": job_status.name, "submission_id": job_id}},
                                    file=args.submissions_file,
                                )
                            del tasks[job_id]
                            break
            # Print outputs for all jobs after interval
            for job_id, lines in collected.items():
                if lines:
                    if job_id not in task_run_ids:
                        # Check line for Run ID: <run_id>
                        for line in lines:
                            if "Run ID:" in line:
                                # NOTE: Currently the ID always ends with 4 the version number
                                run_id_match = re.search(r"Run ID:\s*([a-zA-Z0-9]+)", line)
                                if run_id_match:
                                    run_id = run_id_match.group(1)
                                    task_run_ids[job_id] = run_id
                                    write_back(
                                        args.group,
                                        job_id,
                                        {run_id: {"status": "RUNNING", "submission_id": job_id}},
                                        file=args.submissions_file,
                                    )
                                    break
                    print(f"\n\n ============= Out: {job_id} =============\n\n")
                    print("".join(lines))
            if last_tmux_print + 240 < time.time():
                print("You can follow all jobs individually in separate tmux sessions using the following commands:")
                tmux_commands = [get_tmux_log_command(job_id) for job_id in jobs_tracked]
                print("\n".join(tmux_commands))
                last_tmux_print = time.time()

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
        asyncio.run(gather_and_print_job_outputs(jobs_tracked))
    except KeyboardInterrupt:
        print("\n\n\n\n########################## Keyboard Interrupt Detected #########################\n\n\n\n")
        choice = input(
            "Stop all runs or exit log streaming only? "
            "Ctrl+C to exit streaming, "
            "Press any key to monitor statuses only. "
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
        print("Performing only monitoring of job statuses...")
        try:
            jobs_tracked_left = jobs_tracked.copy()
            final_states: dict[str, JobStatus] = {}
            while jobs_tracked_left:
                print("-" * 80)
                jobs_to_delete = []
                for job_id in jobs_tracked.keys():
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
                            write_back(
                                args.group,
                                job_id,
                                {run_id: {"status": job_status.name, "submission_id": job_id}},
                                file=args.submissions_file,
                            )
                        jobs_to_delete.append(job_id)
                for job_id in jobs_to_delete:
                    jobs_tracked_left.pop(job_id, None)
                time.sleep(180)
        except KeyboardInterrupt:
            print(
                "\n\n\n\n########################## Second Keyboard Interrupt Detected #########################\n\n\n\n"
            )
            print("Exiting monitoring.")
            sys.exit(1)
