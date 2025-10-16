from __future__ import annotations

import io
import logging
import subprocess
import time
from typing import IO, ClassVar, Optional, TypeAlias
from enum import IntEnum, auto

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

    WANDB_FILE_EMPTY = auto()
    """
    Upload process failed because of an empty header: "wandb file is empty".
    This can happen if the files are not fully synced yet or a data loss occurred.

    Attention:
        It is likely that the `_WandbLoggingActor` crashed - this can happen silently.
    """

    WANDB_UNKNOWN_ERROR = auto()
    """Process failed due to an unknown WandB specific error."""

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
            args=process.args,
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
    ) -> int | ExitCode:
        """
        Wait for process to complete and return its exit code, handling exceptions.

        If an error pattern is detected in the process output or if the timeout is reached
        and `terminate_on_timeout` is ``True``, this method forcibly terminates the process.
        """
        start = last_time = time.time()

        stdout_accum = ""
        error_occurred = False
        # Define error patterns to look for in output (case-insensitive)
        stdout_type = None
        error_code = ExitCode.SUCCESS
        while True:
            line = process.stdout.readline() if process.stdout else None
            time_now = time.time()
            if time_now - last_time > 10:
                logger.info(
                    "Still uploading trial %s to %s after %.1f seconds...",
                    trial_id,
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
                    logger.error(
                        "Detected error pattern in %s sync output while uploading trial %s. "
                        "Killing process. Output line: %s",
                        cls._upload_service_name,
                        trial_id,
                        line.strip(),
                    )
                    if "contact support" in line and "500" in line:
                        # When forking the run-*-history artifact is not yet available
                        # this file is ONLY created when viewing the run on the website
                        # its possible that this error is raised while the file is still built
                        # it *might* be resolved after some wait time.
                        if timeout > 15:
                            # try again recursively with less time. Then continue if still fails.
                            time.sleep(5)
                            logger.info(
                                "Still uploading trial %s to %s after error max time until timeout %.1f. "
                                "Encountered error: %s retrying...",
                                trial_id,
                                cls._upload_service_name,
                                timeout,
                                line.strip(),
                            )
                            return cls._failure_aware_wait(
                                process, timeout=max(10, timeout - (time_now - start) - 10), terminate_on_timeout=False
                            )
                        error_code = ExitCode.WANDB_SERVER_ERROR
                    elif "not found (<Response [404]>)" in line:
                        error_code = ExitCode.WANDB_PARENT_NOT_FOUND
                    elif "fromStep is greater than the run's last step" in line:
                        error_code = ExitCode.WANDB_BEHIND_STEP
                    elif "wandb file is empty" in line:
                        error_code = ExitCode.WANDB_FILE_EMPTY
                    else:
                        error_code = ExitCode.ERROR
                    process.terminate()
                    time.sleep(2)  # give some time to terminate
                    break
            elif process.poll() is not None:
                error_code = ExitCode.SUCCESS
                break  # Process finished
            elif time_now - start > timeout:
                logger.warning(
                    "Timeout reached while uploading trial %s to %s. %s",
                    trial_id,
                    cls._upload_service_name,
                    "Killing process." if terminate_on_timeout else "Not killing process, but not tracking anymore.",
                )
                if terminate_on_timeout:
                    process.terminate()
                error_occurred = True
                error_code = ExitCode.TIMEOUT
                break
            else:
                time.sleep(0.2)  # Avoid busy waiting
        if process.stdout is not None:
            try:
                rest = process.stdout.read()
                if rest:
                    if isinstance(rest, bytes):
                        rest = rest.decode("utf-8")
                    stdout_accum += rest
            except (IOError, OSError) as e:  # noqa: BLE001
                logger.warning("Could not read remaining output from process.stdout: %s", e)
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
            logger.info(
                "Successfully synced offline run %s: %s\n%s",
                result.args[2:],
                trial_info,
                stdout,
                stacklevel=stacklevel,
            )
            error_code = ExitCode.SUCCESS
        elif "not found (<Response [404]>)" in stdout:
            logger.error(
                "Could not sync run for %s %s (Is it a forked_run? - The parent needs to be uploaded first): %s",
                trial_info,
                result.args[2:],
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
                result.args[2:],
                result.stdout,
                stacklevel=stacklevel,
            )
            error_code = exit_code = ExitCode.WANDB_BEHIND_STEP
        else:
            logger.error(
                "Error during syncing offline run %s %s:\n%s",
                trial_info,
                result.args[2:],
                stdout,
                stacklevel=stacklevel,
            )
            exit_code = result.returncode or 1
            error_code = ExitCode.ERROR
        if result.returncode != 0 or result.stderr:
            logger.error(
                "Failed to sync offline run %s %s (%s):\n%s",
                trial_info,
                result.args[2:],
                result.stderr or "",
                error_code,
                stacklevel=stacklevel,
            )
        return exit_code
