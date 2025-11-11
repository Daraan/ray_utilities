#!/usr/bin/bash
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import argcomplete
import ray

from ray_utilities.callbacks.wandb import RunNotFound

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
            default="./outputs/experiments",
            help="Path to experiment outputs (default: ./outputs/experiments).",
        )
        subparser.add_argument("--entity", type=str, default=None, help="WandB entity (optional).")
        subparser.add_argument(
            "--experiment_key", type=str, default=None, help="WandB to upload check a single experiment key (optional)."
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
    argcomplete.autocomplete(parser)
    return parser


parser = get_parser()
if __name__ == "__main__":
    args = parser.parse_args()

    logger.info("Parsed arguments: run_id=%s, project=%s, entity=%s", args.run_id, args.project, args.entity)

    # If project is a path split it
    first_arg_path = Path(args.project)
    # fork_file_present
    if first_arg_path.is_dir() and first_arg_path.exists():
        args.project = first_arg_path.name.rsplit("-", 1)[0]
        if args.project == "driver_artifacts":
            args.project = first_arg_path.parent.name.rsplit("-", 1)[0]
            args.run_id = first_arg_path.parent.name.split("-")[-1]
        else:
            args.run_id = first_arg_path.name.split("-")[-1]
        assert args.experiment_path == "./outputs/experiments", (
            "When project is a path, experiment_path must be default."
        )  # noqa: E501
        args.experiment_path = str(first_arg_path.parent)
        fork_file_present = bool(list(first_arg_path.glob("pbt_fork_data*")))
        if fork_file_present:
            args.no_monitor = True
    assert args.run_id is not None, "run_id is required."
    from ray_utilities.callbacks.wandb import WandbUploaderMixin

    uploader = WandbUploaderMixin()
    uploader.project = args.project
    if args.command == "verify":
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
        uploader.upload_paths(wandb_paths=wandb_paths, use_tqdm=True, wait=True, skip_synced=False)
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
