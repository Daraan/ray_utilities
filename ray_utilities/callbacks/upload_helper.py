from __future__ import annotations

import io
import logging
import os
import select
import subprocess
import sys
import time
from enum import IntEnum, auto
from typing import IO, ClassVar, Optional, TypeAlias

from ray_utilities.nice_logger import ImportantLogger

logger = logging.getLogger(__name__)

AnyPopen: TypeAlias = subprocess.Popen[str] | subprocess.Popen[bytes]


class ExitCode(IntEnum):
    SUCCESS = 0
    """Process completed successfully without errors."""

    ERROR = 1
    """Process encountered an error during execution."""

    TIMEOUT = auto()
    """Process was terminated due to a timeout."""

    TERMINATED = auto()
    """Process was manually terminated."""

    WANDB_PARENT_NOT_FOUND = auto()
    """Process failed due to WandB specific error, normally a HTTP 404. Parent needs to be uploaded first."""

    WANDB_BEHIND_STEP = auto()
    """Failed because current step on the server is behind the local step.

    Solution: Either the parent is not uploaded yet or the wandb-run-history is not properly updated.
    Visiting the run page of the parent should solve this issue.
    """

    WANDB_SERVER_ERROR = auto()
    """
    Process failed due to a WandB server error (5xx).

    This could be when a fork is created but the parent's history artifact is not yet available.

    Solution: Visit the run page of the parent to trigger creation of the history artifact. Potentially wait and retry.
    """

    WANDB_SERVER_UNAVAILABLE = auto()
    """
    Process failed because WandB server is currently unavailable (e.g., HTTP 503).

    Solution: Wait and retry later.
    """

    WANDB_FILE_EMPTY = auto()
    """
    Upload process failed because of an empty header: "wandb file is empty".
    This can happen if the files are not fully synced yet or a data loss occurred.

    Attention:
        It is likely that the `_WandbLoggingActor` crashed - this can happen silently.
    """

    WANDB_UNKNOWN_ERROR = auto()
    """Process failed due to an unknown WandB specific error."""

    COMET_ALREADY_UPLOADED = -1
    """The experiment was already uploaded to Comet, but the zip file was not yet moved."""

    NO_PARENT_FOUND = 499
    """No parent found for the current run, but one was expected - this points at a implementation error."""


class UploadHelperMixin:
    error_patterns: ClassVar[set[str]] = {"error", "failed", "exception", "traceback", "critical"}
    """lowercase error patterns to look for in output"""
    _upload_service_name: ClassVar[str] = "upload_service"

    @staticmethod
    def _popen_to_completed_process(
        process: AnyPopen,
        out: Optional[str] = None,
        returncode: Optional[int] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Convert a Popen process to a CompletedProcess by waiting for it to finish."""
        if process.poll() is None:
            logger.warning("Calling _popen_to_completed_process on running process", stacklevel=2)
        if out is not None:
            out_str = out
        else:
            out_str = process.stdout.read() if process.stdout else None
        if isinstance(out_str, bytes):
            out_str = out_str.decode("utf-8")
        err = process.stderr.read() if process.stderr else None
        err_str = err if isinstance(err, str) or err is None else err.decode("utf-8")
        return subprocess.CompletedProcess(
            args=" ".join(map(str, process.args)) if isinstance(process.args, list) else process.args,
            returncode=returncode if returncode is not None else process.returncode,
            stdout=out_str or "",
            stderr=err_str or "",
        )

    @classmethod
    def _failure_aware_wait(
        cls,
        process: AnyPopen,
        timeout: float = 300,
        trial_id: str = "",
        *,
        terminate_on_timeout: bool = True,
        report_upload: bool = True,
        prev_out: Optional[str] = None,
        upload_service_name: Optional[str] = None,
    ) -> int | ExitCode:
        """
        Wait for process to complete and return its exit code, handling exceptions.

        If an error pattern is detected in the process output or if the timeout is reached
        and `terminate_on_timeout` is ``True``, this method forcibly terminates the process.
        """
        start = last_time = time.time()

        if os.name != "posix":
            # On non-POSIX (e.g., Windows), select.select does not work on file objects.
            # Fallback: blocking read with warning (may block if no output).
            ImportantLogger.important_warning(
                logger,
                "Non-POSIX system detected; using blocking readline on process.stdout. "
                "This may block if no output is available. Use Ctrl+C to interrupt if necessary.",
                stacklevel=2,
            )
        stdout_accum = prev_out or ""
        error_occurred = False
        # Define error patterns to look for in output (case-insensitive)
        stdout_type = None
        error_code = ExitCode.SUCCESS
        max_iterations = 10 * timeout  # Hard safeguard to prevent infinite loop
        iteration = 0
        try:
            while True:
                iteration += 1
                if iteration > max_iterations:
                    logger.error(
                        "Maximum iterations (%d) reached in _failure_aware_wait for trial %s. Forcibly terminating process.",
                        max_iterations,
                        trial_id,
                    )
                    if terminate_on_timeout:
                        process.terminate()
                    error_occurred = True
                    error_code = ExitCode.TIMEOUT
                    break
                line = None
                if process.stdout and not process.stdout.closed:
                    try:
                        # Avoid hanging: only read if data is available
                        if os.name == "posix":
                            try:
                                rlist, _, _ = select.select([process.stdout], [], [], 0.2)
                            except ValueError:
                                logger.exception("Error selecting stdout for wandb upload process")
                                rlist = []
                                line = ""
                            else:
                                # somehow this could still raise ValueError: I/O operation on closed file.
                                line = process.stdout.readline()
                        else:
                            line = process.stdout.readline()
                    except (IOError, OSError, ValueError) as e:
                        logger.error("Error reading stdout for wandb upload process: %s", e)
                time_now = time.time()
                # ray.tune.utils.util import warn_if_slow
                # from ray.tune.execution import tune_controller

                if time_now - last_time > 15:
                    logger.info(
                        "Still uploading trial %s (pid %s) to %s after %.1f seconds...",
                        trial_id,
                        process.pid,
                        cls._upload_service_name,
                        time_now - start,
                    )
                    last_time = time_now
                if line:
                    if isinstance(line, bytes):
                        stdout_type = bytes
                        line = line.decode("utf-8")
                    else:
                        stdout_type = str
                    if not line.endswith("\n"):
                        line += "\n"
                    stdout_accum += line
                    # Check for any error pattern in the line (case-insensitive)
                    if any(pattern in line.lower() for pattern in map(str.lower, cls.error_patterns)):
                        error_occurred = True
                        recoverable = False
                        if ("contact support" in line and "500" in line) or "history not materialized yet" in line:
                            # When forking the run-*-history artifact is not yet available
                            # this file is ONLY created when viewing the run on the website
                            # its possible that this error is raised while the file is still built
                            # it *might* be resolved after some wait time.
                            recoverable = True
                            error_code = ExitCode.WANDB_SERVER_ERROR
                        elif "not found (<Response [404]>)" in line:
                            # Low chance that still in progress, but likely parent upload failed
                            recoverable = True
                            error_code = ExitCode.WANDB_PARENT_NOT_FOUND
                        elif "experiment was already uploaded" in line:
                            # we already uploaded to comet, zip not yet moved
                            error_code = ExitCode.COMET_ALREADY_UPLOADED
                        elif "fromStep is greater than the run's last step" in line:
                            # run-*-history artifact needs to be updated to the latest step
                            # need to visit the website to trigger the update
                            recoverable = True
                            error_code = ExitCode.WANDB_BEHIND_STEP
                        elif "wandb file is empty" in line:
                            error_code = ExitCode.WANDB_FILE_EMPTY
                        elif "Response [503]" in line or "currently unavailable" in line:
                            recoverable = True  # but likely not by waiting here
                            error_code = ExitCode.WANDB_SERVER_UNAVAILABLE
                        else:
                            error_code = ExitCode.ERROR
                        if not recoverable:
                            logger.error(
                                "Detected error pattern in %s sync output while uploading trial %s. "
                                "Killing process. Output line: %s",
                                upload_service_name or cls._upload_service_name,
                                trial_id,
                                line.strip(),
                            )
                            process.terminate()
                            time.sleep(2)  # give some time to terminate
                            break
                        ImportantLogger.important_warning(
                            logger,
                            (
                                "Detected error %s in %s sync output while uploading trial %s. "
                                + (
                                    "Waiting but speeding up wait progressively before retrying..."
                                    if timeout > 15
                                    else "Timeout reached, "
                                    + (
                                        "not terminating process anymore but also not tracking it further. "
                                        if not terminate_on_timeout
                                        else "terminating process."
                                    )
                                )
                                + "\n%s"
                            ),
                            error_code.name,
                            cls._upload_service_name,
                            trial_id,
                            line.strip(),
                        )
                        if timeout > 15:
                            # try again recursively with less time. Then continue if still fails.
                            time.sleep(7)
                            ImportantLogger.important_info(
                                logger,
                                "Still uploading trial %s (pid %s) to %s after error max time until timeout %.1f. "
                                "Encountered error: %s retrying...",
                                trial_id,
                                process.pid,
                                cls._upload_service_name,
                                timeout,
                                line.strip(),
                            )
                            return cls._failure_aware_wait(
                                process,
                                timeout=min(max(10, timeout - (time_now - start) - 10), 200),
                                terminate_on_timeout=False,
                                prev_out=stdout_accum,
                                trial_id=trial_id,
                            )
                        error_code = ExitCode.TIMEOUT
                        if terminate_on_timeout:
                            process.terminate()
                        break
                elif process.poll() is not None:
                    error_code = ExitCode.SUCCESS
                    break  # Process finished
                elif time_now - start > timeout:
                    logger.warning(
                        "Timeout reached while uploading trial %s to %s. %s: %s",
                        trial_id,
                        cls._upload_service_name,
                        "Killing process."
                        if terminate_on_timeout
                        else "Not killing process, but not tracking anymore.",
                        process.args,
                    )
                    if terminate_on_timeout:
                        process.terminate()
                    error_occurred = True
                    error_code = ExitCode.TIMEOUT
                    break
                else:
                    time.sleep(0.2)  # Avoid busy waiting
        except KeyboardInterrupt:
            ImportantLogger.important_info(
                logger, "KeyboardInterrupt received. Stopping upload process for %s.", trial_id
            )
            process.terminate()
            error_occurred = True
            error_code = ExitCode.TERMINATED
        except Exception:
            logger.exception(
                "Exception occurred while waiting for upload process of trial %s",
                trial_id,
            )
            process.terminate()
            error_occurred = True
            error_code = ExitCode.ERROR

        # Only attempt to read remaining output if we can do so safely
        if process.stdout is not None:
            # For terminated processes, read all remaining output
            if process.poll() is not None:
                try:
                    rest = process.stdout.read()
                    if rest:
                        if isinstance(rest, bytes):
                            rest = rest.decode("utf-8")
                        stdout_accum += rest
                except (IOError, OSError, ValueError) as e:
                    logger.warning("Could not read remaining output from terminated process.stdout: %s", e)
            # For still-running processes (abandoned due to timeout), try non-blocking read
            elif not terminate_on_timeout and os.name == "posix":
                try:
                    # Use select with 0 timeout to check if data is available without blocking
                    rlist, _, _ = select.select([process.stdout], [], [], 0)
                    if rlist:
                        rest = process.stdout.read()
                        if rest:
                            if isinstance(rest, bytes):
                                rest = rest.decode("utf-8")
                            stdout_accum += rest
                except (IOError, OSError, ValueError) as e:
                    logger.debug("Could not read output from still-running process: %s", e)

        if error_occurred:
            if error_code <= ExitCode.ERROR:
                returncode = process.returncode or ExitCode.ERROR
            else:  # more specific
                returncode = error_code
        elif process.returncode is not None:
            if error_code <= ExitCode.ERROR:
                returncode = process.returncode
            else:  # more specific
                returncode = error_code
        else:
            returncode = ExitCode.SUCCESS
        if report_upload:
            return cls._report_upload(
                cls._popen_to_completed_process(process, returncode=returncode, out=stdout_accum),
                trial_id,
                stacklevel=3,  # one above this function
            )
        logger.debug("Not reported output of trial %s upload process: %s", trial_id, stdout_accum)
        if process.poll() is not None and stdout_accum and stdout_type is not None:
            # regenerate stdout

            fresh_stdout: IO[str] | IO[bytes]
            if stdout_type is bytes:
                fresh_stdout = io.BytesIO(stdout_accum.encode("utf-8"))
            else:
                fresh_stdout = io.StringIO(stdout_accum)
            process.stdout = fresh_stdout  # pyright: ignore[reportAttributeAccessIssue]
        return returncode

    @classmethod
    def _report_upload(
        cls,
        result: subprocess.CompletedProcess[str] | AnyPopen,
        trial_id: Optional[str] = None,
        stacklevel: int = 2,
    ) -> int | ExitCode:
        """Check result return code and log output."""
        if isinstance(result, subprocess.Popen):
            result = cls._popen_to_completed_process(result)
        exit_code = result.returncode
        trial_info = f"for trial {trial_id}" if trial_id else ""
        stdout = result.stdout or ""
        error_code = ExitCode.SUCCESS
        if result.returncode == 0 and (
            not stdout or not any(pattern in stdout.lower() for pattern in map(str.lower, cls.error_patterns))
        ):
            ImportantLogger.important_info(
                logger,
                "Successfully synced offline run %s: %s\n%s",
                result.args[2:] if isinstance(result.args, list) else result.args,
                trial_info,
                stdout,
                stacklevel=stacklevel,
            )
            error_code = ExitCode.SUCCESS
        elif "not found (<Response [404]>)" in stdout:
            logger.error(
                "Could not sync run for %s %s (Is it a forked_run? - The parent needs to be uploaded first): %s",
                trial_info,
                result.args if isinstance(result.args, str) else result.args[2:],
                result.stdout,
                stacklevel=stacklevel,
            )
            error_code = ExitCode.WANDB_PARENT_NOT_FOUND
            exit_code = error_code
        elif "fromStep is greater than the run's last step" in stdout:
            logger.error(
                "Could not sync run %s %s "
                "(Is it a forked_run? - The parents fork step needs to be uploaded first.) "
                "If this error persists it might be a off-by-one error:\n%s",
                trial_info,
                result.args[2:] if isinstance(result.args, list) else result.args,
                result.stdout,
                stacklevel=stacklevel,
            )
            error_code = exit_code = ExitCode.WANDB_BEHIND_STEP
        else:
            logger.error(
                "Error during syncing offline run %s %s:\n%s",
                trial_info,
                result.args[2:] if isinstance(result.args, list) else result.args,
                stdout,
                stacklevel=stacklevel,
            )
            exit_code = result.returncode or 1
            error_code = ExitCode.ERROR
        if result.returncode != 0 or result.stderr:
            logger.error(
                "Failed to sync offline run %s %s (%s):\n%s",
                trial_info,
                " ".join(map(str, result.args[2:])) if isinstance(result.args, list) else result.args,
                error_code,
                result.stderr or stdout or "",
                stacklevel=stacklevel,
            )
        return exit_code
