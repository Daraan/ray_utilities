from __future__ import annotations

import io
import subprocess
import time
from typing import AnyStr, ClassVar, Optional
import logging

logger = logging.getLogger(__name__)


class UploadHelperMixin:
    error_patterns: ClassVar[set[str]] = {"error", "failed", "exception", "traceback", "critical"}
    """lowercase error patterns to look for in output"""
    _upload_service_name: ClassVar[str] = "upload_service"

    @staticmethod
    def _popen_to_completed_process(
        process: subprocess.Popen[AnyStr],
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
        process: subprocess.Popen[AnyStr],
        timeout: int = 600,
        trial_id: str = "",
        *,
        terminate_on_timeout: bool = True,
        report_upload: bool = True,
    ) -> int:
        """
        Wait for process to complete and return its exit code, handling exceptions.

        If an error pattern is detected in the process output or if the timeout is reached
        and `terminate_on_timeout` is ``True``, this method forcibly terminates the process.
        """
        start = last_count = time.time()

        stdout_accum = ""
        error_occurred = False
        # Define error patterns to look for in output (case-insensitive)
        while True:
            line = process.stdout.readline() if process.stdout else None
            count = time.time()
            if line:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.endswith("\n"):
                    line += "\n"
                stdout_accum += line
                # Check for any error pattern in the line (case-insensitive)
                if any(pattern in line.lower() for pattern in cls.error_patterns):
                    error_occurred = True
                    logger.error(
                        "Detected error pattern in %s sync output while uploading trial %s. "
                        "Killing process. Output line: %s",
                        cls._upload_service_name,
                        trial_id,
                        line.strip(),
                    )
                    process.terminate()
                    break
            elif process.poll() is not None:
                break  # Process finished
            elif count - start > timeout:
                logger.warning(
                    "Timeout reached while uploading trial %s to %s. %s",
                    trial_id,
                    cls._upload_service_name,
                    "Killing process." if terminate_on_timeout else "Not killing process as.",
                )
                error_occurred = True
                if terminate_on_timeout:
                    process.terminate()
                break
            else:
                time.sleep(0.2)  # Avoid busy waiting
            if count - last_count > 10:
                logger.info(
                    "Still uploading trial %s to %s after %.1f seconds...",
                    trial_id,
                    cls._upload_service_name,
                    count - start,
                )
                last_count = count
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
            returncode = process.returncode or 1
        elif process.returncode is not None:
            returncode = process.returncode
        else:
            returncode = 0
        if report_upload:
            return cls._report_upload(
                cls._popen_to_completed_process(process, returncode=returncode),
                trial_id,
            )
        return returncode

    @classmethod
    def _report_upload(
        cls,
        result: subprocess.CompletedProcess[str] | subprocess.Popen[AnyStr],
        trial_id: Optional[str] = None,
    ):
        """Check result return code and log output."""
        if isinstance(result, subprocess.Popen):
            result = cls._popen_to_completed_process(result)
        exit_code = result.returncode
        trial_info = f"for trial {trial_id}" if trial_id else ""
        stdout = result.stdout or ""
        if result.returncode == 0 and (
            not stdout or not any(pattern in stdout.lower() for pattern in cls.error_patterns)
        ):
            logger.info("Successfully synced offline run %s: %s\n%s", result.args[-1], trial_info, stdout)
        elif "not found (<Response [404]>)" in stdout:
            logger.error(
                "Could not sync run for %s %s (Is it a forked_run? - The parent needs to be uploaded first): %s",
                trial_info,
                result.args[-1],
                result.stdout,
            )
            exit_code = result.returncode or 1
        elif "fromStep is greater than the run's last step" in stdout:
            logger.error(
                "Could not sync run %s %s"
                "(Is it a forked_run? - The parents fork step needs to be uploaded first. )"
                "If this error persists it might be a off-by-one error:\n%s",
                trial_info,
                result.args[-1],
                result.stdout,
            )
            exit_code = result.returncode or 1
        else:
            logger.error("Error during syncing offline run %s %s:\n%s", trial_info, result.args[-1], stdout)
            exit_code = result.returncode or 1
        if result.returncode != 0 or result.stderr:
            logger.error("Failed to sync offline run %s %s:\n%s", trial_info, result.args[-1], result.stderr or "")
        return exit_code
