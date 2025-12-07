"""
A script to update the status of Ray jobs in the submissions YAML file.
Reads job IDs from command line arguments or a file, retrieves their current status
and run IDs, then updates the YAML file accordingly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, cast

from ray.job_submission import JobSubmissionClient, JobDetails

# Import shared functionality from ray_submit.py
from ray_submit import extract_run_id_from_logs, write_back, yaml_load

logger = logging.getLogger(__name__)


Matcher = Callable[[JobDetails], bool]


def _match_sympol(job: JobDetails) -> bool:
    return bool((job.submission_id and "sympol" in job.submission_id) or "sympol" in job.entrypoint)


def _match_ppo_with_kl(job: JobDetails) -> bool:
    return bool("--use_kl_loss" in job.entrypoint and "sympol" not in job.entrypoint)


file_matcher: dict[str, Matcher] = {
    "experiments/submissions_sympol.yaml": _match_sympol,
    "experiments/submissions_ppo_with_kl.yaml": _match_ppo_with_kl,
}
default_file = "experiments/submissions_pbt_grouped.yaml"

# Global cache for submission statuses
_submission_status_cache: dict[str, str] = {}


def cache_submission_statuses(submission_files: list[str]) -> dict[str, str]:
    """
    Parse all submission files to cache submission IDs and their statuses.

    Args:
        submission_files: List of submission file paths to parse

    Returns:
        Dictionary mapping submission_id to status
    """
    global _submission_status_cache

    if _submission_status_cache:
        return _submission_status_cache

    logger.info("Building submission status cache from %d files", len(submission_files))

    for file_path in submission_files:
        try:
            if not Path(file_path).exists():
                logger.warning("Submission file not found: %s", file_path)
                continue

            with open(file_path, "r") as f:
                data = yaml_load(f)

            for group_name, group_data in data.items():
                if not isinstance(group_data, dict) or "run_ids" not in group_data:
                    continue

                run_ids_data = group_data.get("run_ids", {})
                for run_key, runs in run_ids_data.items():
                    for run_id, run_info in runs.items():
                        if isinstance(run_info, dict):
                            submission_id = run_info.get("submission_id")
                            try:
                                status = run_info.get("status")
                            except RuntimeError as e:
                                logger.warning("RuntimeError getting status for %s in %s: %r", run_id, file_path, e)
                                continue

                            if submission_id and status:
                                _submission_status_cache[submission_id] = status

        except Exception as e:
            logger.warning("Failed to parse submission file %s: %s", file_path, e)
            continue

    logger.info("Cached %d submission statuses", len(_submission_status_cache))
    return _submission_status_cache


def should_skip_update(submission_id: str) -> bool:
    """
    Check if a submission should be skipped based on cached status.

    Args:
        submission_id: The submission ID to check

    Returns:
        True if the submission should be skipped, False otherwise
    """
    status = _submission_status_cache.get(submission_id)
    if status and status not in ("PENDING", "RUNNING"):
        logger.debug("Skipping submission %s with status: %s", submission_id, status)
        return True
    return False


def parse_job_id(job_id: str) -> tuple[str, str]:
    """
    Parse a job ID to extract group and environment.

    Args:
        job_id: Job ID in format like "sympol_pbt_lr_Swimmer-v5_2025-12-06_23:32:43"

    Returns:
        Tuple of (group, environment)

    Raises:
        ValueError: If job ID format cannot be parsed
    """
    # Pattern: group_environment_timestamp
    # Environment is always of form (\w|\d)+-v\d+
    match = re.match(r"^(.+?)_([A-Za-z0-9]+-v\d+)_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$", job_id)
    if not match:
        raise ValueError(f"Cannot parse job ID format: {job_id}")

    group = match.group(1)
    environment = match.group(2)
    # Fixup group:
    if group.count("pbt") > 1:
        split = group.rsplit("pbt", 2)
        group = split[0] + "pbt" + split[-1]
    return group, environment


def get_submission_file_for_job(
    client: JobSubmissionClient, job_id: str, job_info: JobDetails | None = None
) -> tuple[str, JobDetails]:
    """
    Get the appropriate submissions file for a job by checking it against matchers.

    Args:
        client: Ray JobSubmissionClient
        job_id: Submission ID to get file for

    Returns:
        Path to the appropriate submissions file
    """
    try:
        if not job_info:
            job_info = client.get_job_info(job_id)
        for file_path, matcher in file_matcher.items():
            if matcher(job_info):
                logger.info("Job %s matched to file: %s", job_id, file_path)
                return file_path, job_info
        logger.info("Job %s using default file: %s", job_id, default_file)
    except Exception as e:
        logger.warning("Failed to get job info for %s, using default file: %r", job_id, e)
        raise
    return default_file, job_info


async def get_run_id_from_job(client: JobSubmissionClient, job_id: str) -> str | None:
    """
    Get the run ID from job logs by scanning the first few lines.

    Args:
        client: Ray JobSubmissionClient
        job_id: Submission ID to get logs for

    Returns:
        Run ID if found, None otherwise
    """
    try:
        log_iterator = client.tail_job_logs(job_id)
        log_lines = []

        # Collect first few lines to scan for run ID
        async for line in log_iterator:
            log_lines.append(line)
            if len(log_lines) >= 50:  # Check first 50 lines
                break
        if not log_lines:
            raise RuntimeError(f"No logs found for job {job_id}")
        return extract_run_id_from_logs(log_lines)
    except StopAsyncIteration as e:
        logger.warning("Failed to get logs for job %s: %r", job_id, e)
        raise


async def update_job_status(
    client: JobSubmissionClient, job_id: str, submissions_file: str, job_info: JobDetails | None = None
) -> bool:
    """
    Update the status of a single job in the submissions file.

    Args:
        client: Ray JobSubmissionClient
        job_id: Submission ID to update
        submissions_file: Path to the submissions YAML file

    Returns:
        True if successfully updated, False otherwise
    """
    try:
        # Check cache first to see if we should skip this update
        if should_skip_update(job_id):
            logger.info("Skipping job %s (status already in terminal state)", job_id)
            return True

        # Parse job ID to get group and environment
        group, environment = parse_job_id(job_id)
        # fixup old format
        if "sympol" in submissions_file and "sympol" not in group:
            group = "sympol_" + group
        logger.info("Processing job %s (group=%s, environment=%s)", job_id, group, environment)

        # Get current job status
        if not job_info:
            job_status = client.get_job_status(job_id)
        else:
            job_status = job_info.status
        logger.info("Job %s status: %s", job_id, job_status)

        # Get run ID from logs
        try:
            run_id = await get_run_id_from_job(client, job_id)
        except RuntimeError as e:
            if "No logs found for job" in str(e):
                # Best delete the from cluster, but a longer job we maybe want to keep
                logger.warning(
                    "Job %s %s appears to be deleted or not available, cannot get run ID", job_id, job_status
                )
                return False
        except Exception as e:
            logger.error("Failed to get run ID for job %s: %r", job_id, e)
            breakpoint()
        if not run_id:
            logger.warning("Could not extract run ID for job %s", job_id)
            # Use job_id as fallback for run_id
            run_id = job_id
            raise RuntimeError(f"Could not extract run ID for job {job_id}")

        logger.info("Job %s run ID: %s", job_id, run_id)

        # Reconstruct original job_id for write_back (remove timestamp suffix)
        original_job_id = f"{environment}"

        # Update the submissions file
        write_back(
            group,
            original_job_id,
            {
                run_id: {
                    "status": job_status.name if hasattr(job_status, "name") else str(job_status),
                    "submission_id": job_id,
                }
            },
            file=submissions_file,
        )

        # Update cache with new status
        _submission_status_cache[job_id] = job_status.name if hasattr(job_status, "name") else str(job_status)

        logger.info("Successfully updated job %s in submissions file %s", job_id, submissions_file)
        return True

    except Exception as e:
        logger.error("Failed to update job %s: %r", job_id, e)
    return False


async def main():
    parser = argparse.ArgumentParser(description="Update Ray job statuses in submissions YAML file")
    parser.add_argument("submissions_file", help="Path to the submissions YAML file or 'match' to auto-detect")
    parser.add_argument("job_ids", nargs="+", help="Job IDs to update, or path to file containing job IDs")
    parser.add_argument(
        "--address",
        type=str,
        help="The address of the Ray cluster.",
        default="http://" + os.environ.get("DASHBOARD_ADDRESS", "localhost:8265"),
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create Ray client
    try:
        client = JobSubmissionClient(args.address)
    except Exception as e:
        logger.error("Failed to create JobSubmissionClient: %s", e)
        sys.exit(1)

    jobs = client.list_jobs()
    global ALL_JOBS  # noqa: PLW0603
    ALL_JOBS = {job.submission_id: job for job in jobs}
    ALL_JOBS.pop(None, None)
    ALL_JOBS = cast("dict[str, JobDetails]", ALL_JOBS)

    # Parse job IDs - check if first argument is a file
    job_ids = []
    if len(args.job_ids) == 1:
        if Path(args.job_ids[0]).suffix in {".txt", ".list"}:
            # Read job IDs from file
            file_path = Path(args.job_ids[0])
            if file_path.exists():
                with open(file_path, "r") as f:
                    job_ids = [ln for line in f if (ln := line.strip()) and len(ln) > 3 and ln != "Submission ID"]
            else:
                logger.error("Job ID file not found: %s", file_path)
                sys.exit(1)
        elif args.job_ids == ["all"]:
            # Use all job IDs from cluster
            job_ids = list(ALL_JOBS.keys())
        else:
            job_ids = args.job_ids
    else:
        # Use job IDs from command line
        job_ids = args.job_ids
    if not job_ids:
        logger.error("No job IDs provided")
        sys.exit(1)

    # Build cache of submission files to scan
    submission_files = []
    if args.submissions_file == "match":
        # Include all possible submission files for caching
        submission_files = list(file_matcher.keys()) + [default_file]
    else:
        submission_files = [args.submissions_file]

    # Cache submission statuses from all relevant files
    cache_submission_statuses(submission_files)

    logger.info("Updating %d jobs", len(job_ids))

    # Update each job
    success_count = 0
    skipped_count = 0

    for job_id in job_ids:
        # Check cache first
        if should_skip_update(job_id):
            skipped_count += 1
            success_count += 1
            continue

        # Determine submissions file for this job
        if args.submissions_file == "match":
            job_info = ALL_JOBS.get(job_id)
            if not job_info:
                logger.error("Job ID %s not found in cluster jobs", job_id)
                continue
            assert job_info.submission_id == job_id
            try:
                submissions_file, job_info = get_submission_file_for_job(client, job_id, job_info=job_info)
            except RuntimeError as e:
                # likely deleted job
                logger.error("Failed to get submissions file for job %s: %r", job_id, e)
                continue
        else:
            submissions_file = args.submissions_file

        if await update_job_status(client, job_id, submissions_file, job_info=job_info):
            success_count += 1

    logger.info(
        "Successfully processed %d/%d jobs (skipped %d already in terminal state)",
        success_count,
        len(job_ids),
        skipped_count,
    )

    if success_count < len(job_ids):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
