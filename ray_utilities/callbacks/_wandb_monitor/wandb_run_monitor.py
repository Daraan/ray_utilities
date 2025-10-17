"""
WandB Run Monitor Script

This script provides threaded functionality to:
1. Login to WandB using Selenium
2. Visit specific WandB run pages
3. Monitor artifact availability using WandB API
4. Report when artifacts become available
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import dotenv

import ray
import wandb
import wandb.errors

# multiprocessing version we currently do not use
# from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login_mp import WandBCredentials, WandBSeleniumSession
# Instead we use one in the same process
from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login import WandBCredentials, WandBSeleniumSession

if TYPE_CHECKING:
    from ray.actor import ActorHandle, ActorProxy


logger = logging.getLogger(__name__)


@dataclass
class RunMonitorConfig:
    """Configuration for run monitoring."""

    entity: str
    project: str
    run_id: str
    check_interval: float = 5.0  # seconds between artifact checks
    max_wait_time: float = 300.0  # maximum time to wait for artifact (5 minutes)


@dataclass
class MonitorResult:
    """Result of the monitoring operation."""

    success: bool
    artifact_available: bool
    artifact_name: str
    wait_time: float
    error_message: Optional[str] = None


class WandbRunMonitor:
    """
    Threaded WandB run monitor that combines Selenium login with API artifact checking.

    This class provides a complete solution for:
    - Logging into WandB via Selenium
    - Visiting specific run pages
    - Monitoring artifact availability
    - Reporting results via callbacks
    """

    def __init__(
        self,
        credentials: WandBCredentials | None = None,
        *,
        project: str,
        entity: Optional[str] = None,
        browser: str = "firefox",
        headless: bool = True,
        timeout: int = 30,
        callback: Optional[Callable[[str, Any], None]] = None,
        wandb_api: Optional[wandb.Api] = None,
    ):
        """
        Initialize the WandB run monitor.

        Args:
            credentials: WandB login credentials. If not provided, will use environment variables.
                WANDB_VIEWER_MAIL, WANDB_VIEWER_PW, and WANDB_API_KEY.
            entity: WandB entity name. If not provided falls back to auto-detected entity.
            project: Default WandB project name
            browser: Browser to use ("chrome" or "firefox")
            headless: Whether to run browser in headless mode
            timeout: Default timeout for web elements
            callback: Optional callback function for status updates
        """
        if credentials is None:
            if "WANDB_VIEWER_MAIL" not in os.environ or "WANDB_VIEWER_PW" not in os.environ:
                dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())
                if "WANDB_VIEWER_MAIL" not in os.environ or "WANDB_VIEWER_PW" not in os.environ:
                    raise ValueError(
                        "WandB credentials not provided and environment variables %s or %s not set",
                        "WANDB_VIEWER_MAIL",
                        "WANDB_VIEWER_PW",
                    )
            credentials = WandBCredentials(
                username=os.getenv("WANDB_VIEWER_MAIL", ""),
                password=os.getenv("WANDB_VIEWER_PW", ""),
                api_key=os.getenv("WANDB_API_KEY", None),
            )
        self.credentials = credentials
        self.entity = entity
        self.project = project
        self.browser = browser
        self.headless = headless
        self.timeout = timeout
        self.callback = callback

        self.selenium_session: Optional[WandBSeleniumSession] = None
        self.wandb_api: Optional[wandb.Api] = wandb_api
        self._is_initialized = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @staticmethod
    def _set_wandb_viewer_key() -> bool:
        if "WANDB_VIEWER_MAIL" in os.environ and "WANDB_VIEWER_PW" in os.environ:
            return True
        logger.debug(
            "WANDB_VIEWER_MAIL or WANDB_VIEWER_PW not in environment variables, trying to load from ~/.wandb_viewer.env"
        )
        return dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())

    def _notify(self, status: str, data: Any = None) -> None:
        """Send notification via callback if available."""
        if self.callback:
            try:
                self.callback(status, data)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Callback error: %s", e)

    def is_initialized(self) -> bool:
        """Check if the monitor is initialized."""
        return self._is_initialized

    def initialize(self) -> bool:
        """
        Initialize the monitor by setting up Selenium session and WandB API.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.is_initialized():
                logger.warning(
                    "Monitor already initialized - should not call initialize again. Call refresh/cleanup first."
                )
                return True
            self._notify("initializing_monitor")

            # Initialize Selenium session
            self.selenium_session = WandBSeleniumSession(
                self.credentials,
                browser=self.browser,
                headless=self.headless,
                timeout=self.timeout,
                callback=self._selenium_callback,
            )

            # login which sets up the driver and keeps session open for later use
            if not self.selenium_session.login():
                logger.error("Failed to login via Selenium")
                self._notify("initialization_failed", "Selenium login failed")
                return False

            if self.wandb_api is None:
                self.wandb_api = wandb.Api()
            if self.entity is None:
                self.entity = self.wandb_api.default_entity
                if self.entity is None:
                    logger.error("Failed to auto-detect WandB entity")
                    self._notify("initialization_failed", "Failed to auto-detect entity")
                    return False

            logger.info("Successfully initialized monitor")
            self._is_initialized = True
            self._notify("monitor_initialized")

        except Exception as e:
            logger.error("Failed to initialize monitor: %s", e)
            self._notify("initialization_failed", str(e))
            return False
        else:
            return True

    def _selenium_callback(self, status: str, data: Any = None) -> None:
        """Internal callback to forward Selenium status updates."""
        self._notify(f"selenium_{status}", data)

    def visit_run_page(self, run_id: str, entity: Optional[str] = None, project: Optional[str] = None) -> bool:
        """
        Visit a specific WandB run page.

        Args:
            run_id: WandB run ID
            entity: WandB entity name (optional, uses default from init if not provided)
            project: WandB project name (optional, uses default from init if not provided)

        Returns:
            True if page visited successfully, False otherwise
        """
        if not self.selenium_session or not self._is_initialized:
            logger.error("Monitor not initialized")
            return False

        # Use provided entity/project or fall back to defaults
        entity = entity or self.entity
        project = project or self.project

        if entity is None or project is None:
            logger.error("Entity and project must be set to visit run page")
            return False
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        try:
            self._notify("visiting_run_page", {"url": run_url})

            # For multiprocessing implementation, driver is None (it's in the worker process)
            # Just use the session directly without checking driver
            # success = self.selenium_session.visit_wandb_page(run_url)
            success = self.selenium_session.visit_run_page(entity, project, run_id)

        except Exception as e:
            logger.error("Error visiting run page %s: %s", run_url, e)
            self._notify("run_page_visit_error", {"url": run_url, "error": str(e)})
            return False
        else:
            if success:
                logger.info("Successfully visited run page: %s", run_url)
                self._notify("run_page_visited", {"url": run_url})
                return success

            logger.error("Failed to visit run page: %s", run_url)
            self._notify("run_page_visit_failed", {"url": run_url})
            return False

    def close_run_tab(self, run_id: str, entity: Optional[str] = None, project: Optional[str] = None) -> bool:
        """Close the WandB run tab in the Selenium session if available."""
        if not self.selenium_session or not self._is_initialized:
            logger.error("Monitor not initialized or Selenium session missing")
            return False
        entity = entity or self.entity
        project = project or self.project
        if entity is None or project is None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.error("Entity and project must be set to close run tab")
            return False
        try:
            result = self.selenium_session.close_run_tab(entity, project, run_id)
        except Exception as e:
            logger.exception("Error closing run tab: %s")
            self._notify("close_run_tab_error", str(e))
            return False
        else:
            self._notify("run_tab_closed")
            return result

    def get_history_artifact_name(self, run_id: str, version: Optional[str | int] = "latest") -> str:
        """Get the full name of the history artifact for a given run ID and version."""
        if not self.entity or not self.project:
            raise ValueError("Entity and project must be set to construct artifact name")
        if isinstance(version, int):
            version = "v" + str(version)
        return f"{self.entity}/{self.project}/run-{run_id}-history:{version}"

    def is_artifact_present(self, run_id: str, version: Optional[str | int] = "latest") -> Optional[bool]:
        if not self.wandb_api:
            logger.error("WandB API not initialized")
            return None
        history_artifact_name = self.get_history_artifact_name(run_id, version)
        return self.wandb_api.artifact_exists(history_artifact_name)

    def get_artifact_for_run(self, run_id: str, version: Optional[str | int] = "latest") -> Optional[wandb.Artifact]:
        """Retrieve the history artifact for a given run ID and version."""
        if not self.wandb_api:
            logger.error("WandB API not initialized")
            return None
        history_artifact_name = self.get_history_artifact_name(run_id, version)
        try:
            artifact = self.wandb_api.artifact(history_artifact_name)
        except wandb.errors.CommError as e:
            logger.error("Error retrieving artifact %s: %s", history_artifact_name, e)
            return None
        else:
            return artifact

    def latest_artifact_version(self, artifact: wandb.Artifact) -> Optional[int]:
        """Get the latest version number of an artifact."""
        if not artifact or not artifact.version:
            return None
        try:
            if artifact.version.startswith("v"):
                return int(artifact.version[1:])
            return int(artifact.version)
        except ValueError:
            logger.warning("Artifact version %s is not an integer", artifact.version)
            return None

    def monitor_artifact(
        self,
        run_id: str,
        version: Optional[str | int] = "latest",
        *,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        check_interval: float = 5.0,
        max_wait_time: float = 300.0,
    ) -> MonitorResult:
        """
        Monitor artifact availability for a specific run.

        Args:
            run_id: WandB run ID
            entity: WandB entity name (optional, uses default from init if not provided)
            project: WandB project name (optional, uses default from init if not provided)
            check_interval: Seconds between artifact checks
            max_wait_time: Maximum time to wait for artifact (seconds)

        Returns:
            MonitorResult containing the outcome of the monitoring
        """
        if not self.wandb_api:
            error_msg = "WandB API not initialized"
            logger.error(error_msg)
            return MonitorResult(
                success=False,
                artifact_available=False,
                artifact_name="",
                wait_time=0.0,
                error_message=error_msg,
            )

        # Use provided entity/project or fall back to defaults
        entity = entity or self.entity
        project = project or self.project

        # Construct artifact name
        if isinstance(version, int):
            version = "v" + str(version)

        history_artifact_name = self.get_history_artifact_name(run_id, version)

        self._notify(
            "starting_artifact_monitor",
            {
                "artifact_name": history_artifact_name,
                "check_interval": check_interval,
                "max_wait_time": max_wait_time,
            },
        )

        start_time = time.time()
        total_wait_time = 0.0

        try:
            while total_wait_time < max_wait_time and not self._stop_event.is_set():
                # Check if artifact exists
                try:
                    artifact_available = self.wandb_api.artifact_exists(history_artifact_name)
                except Exception as e:  # noqa: BLE001
                    # Handle WandB communication errors (network issues, etc.)
                    logger.warning("WandB API error while checking artifact: %s", e)
                    self._notify(
                        "artifact_check_error",
                        {
                            "artifact_name": history_artifact_name,
                            "error": str(e),
                            "wait_time": total_wait_time,
                        },
                    )
                    artifact_available = False

                if artifact_available:
                    elapsed_time = time.time() - start_time
                    logger.info("Artifact %s is now available after %.2f seconds", history_artifact_name, elapsed_time)

                    self._notify(
                        "artifact_available",
                        {
                            "artifact_name": history_artifact_name,
                            "wait_time": elapsed_time,
                        },
                    )

                    return MonitorResult(
                        success=True,
                        artifact_available=True,
                        artifact_name=history_artifact_name,
                        wait_time=elapsed_time,
                    )

                # Artifact not available yet
                self._notify(
                    "artifact_not_available",
                    {
                        "artifact_name": history_artifact_name,
                        "wait_time": total_wait_time,
                    },
                )

                logger.debug(
                    "Artifact %s not available after %.2f seconds. Waiting...",
                    history_artifact_name,
                    total_wait_time,
                )

                # Wait for next check
                if self._stop_event.is_set():
                    break
                time.sleep(check_interval)
                total_wait_time = time.time() - start_time

            # Timeout or stopped
            elapsed_time = time.time() - start_time
            if self._stop_event.is_set():
                error_msg = "Monitoring stopped by user"
            else:
                error_msg = f"Timeout after {max_wait_time} seconds"

            logger.warning("Artifact monitoring ended: %s", error_msg)
            self._notify(
                "artifact_monitor_timeout",
                {
                    "artifact_name": history_artifact_name,
                    "wait_time": elapsed_time,
                    "reason": error_msg,
                },
            )

            return MonitorResult(
                success=False,
                artifact_available=False,
                artifact_name=history_artifact_name,
                wait_time=elapsed_time,
                error_message=error_msg,
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Unexpected error during artifact monitoring: {e}"
            logger.error(error_msg)

            self._notify(
                "artifact_monitor_error",
                {
                    "artifact_name": history_artifact_name,
                    "error": str(e),
                    "wait_time": elapsed_time,
                },
            )

            return MonitorResult(
                success=False,
                artifact_available=False,
                artifact_name=history_artifact_name,
                wait_time=elapsed_time,
                error_message=error_msg,
            )

    def monitor_run_threaded(
        self,
        run_id: str,
        version: Optional[str | int] = "latest",
        *,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        check_interval: float = 5.0,
        max_wait_time: float = 300.0,
    ) -> threading.Thread:
        """
        Start monitoring a run in a separate thread.

        Args:
            run_id: WandB run ID
            entity: WandB entity name (optional, uses default from init if not provided)
            project: WandB project name (optional, uses default from init if not provided)
            check_interval: Seconds between artifact checks
            max_wait_time: Maximum time to wait for artifact (seconds)

        Returns:
            Thread object for the monitoring process

        Attention:
            Do not call ray.get on this method as the returning thread object is not serializable.
        """
        if self.monitor_thread_running():
            logger.warning("Monitor thread already running")
            return cast("threading.Thread", self.monitor_thread)

        self.monitor_thread = threading.Thread(
            target=self.monitor_run,
            args=(run_id,),
            kwargs={
                "entity": entity,
                "project": project,
                "check_interval": check_interval,
                "max_wait_time": max_wait_time,
                "version": version,
            },
            name=f"wandb-monitor-{run_id}-v{version}",
            daemon=True,
        )
        self.monitor_thread.start()

        logger.info("Started WandB run monitor thread for run %s version %s", run_id, version)
        self._notify("monitor_thread_started", {"run_id": run_id, "version": version})

        return self.monitor_thread

    def monitor_run(
        self,
        run_id: str,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str | int] = "latest",
        check_interval: float = 5.0,
        max_wait_time: float = 300.0,
        monitor_artifact: bool = True,
    ) -> None:
        """
        Monitor a WandB run for artifact availability.

        Can be run via threading or directly, compatible with `ray.remote`.
        """
        try:
            # Initialize if not already done
            if not self._is_initialized:
                if not self.initialize():
                    self._notify("threaded_monitor_failed", "Initialization failed")
                    return

            # Visit the run page
            page_visited = self.visit_run_page(run_id, entity=entity, project=project)
            if not page_visited:
                logger.warning("Failed to visit run page, but continuing with artifact monitoring.")

            self._notify(
                "monitor_visited_run_page",
                {
                    "run_id": run_id,
                    "entity": entity or self.entity,
                    "project": project or self.project,
                    "success": page_visited,
                },
            )

            # Monitor the artifact
            if monitor_artifact:
                result = self.monitor_artifact(
                    run_id,
                    version=version,
                    entity=entity,
                    project=project,
                    check_interval=check_interval,
                    max_wait_time=max_wait_time,
                )

                self._notify(
                    "threaded_monitor_complete",
                    {
                        "run_id": run_id,
                        "entity": entity or self.entity,
                        "project": project or self.project,
                        "result": result,
                    },
                )

        except Exception as e:
            logger.error("Thread execution error: %s", e)
            self._notify("threaded_monitor_error", str(e))

    def monitor_thread_running(self) -> bool:
        """Check if monitoring thread is active."""
        if self.monitor_thread is None:
            return False
        if self.monitor_thread.is_alive():
            return True
        return False

    def stop_monitoring(self, timeout: float = 10) -> bool:
        """Stop any active monitoring. Returns True if no active activity remains."""
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=timeout)
            logger.debug("Stopped monitoring thread")
            if not self.monitor_thread.is_alive():
                self.monitor_thread = None
                logger.debug("Monitoring thread has been joined and cleared")
                return True
            return False
        return True

    def refresh(self, timeout=30):
        done = self.stop_monitoring()
        logger.debug("Stopped monitoring: %s", done)
        if done:
            self._stop_event.clear()
            self.monitor_thread = None
            return
        logger.info("Waiting for monitoring thread to stop...")
        wait = 0
        while self.monitor_thread_running() and wait < timeout:
            time.sleep(1)
            wait += 1
        if wait >= timeout:
            logger.warning("Timeout waiting for monitoring thread to stop")
        self._stop_event.clear()
        self.monitor_thread = None

    def cleanup(self) -> None:
        """Clean up resources safely and thoroughly."""
        # Stop monitoring first to prevent race conditions
        try:
            self.stop_monitoring(timeout=0.2)
        except Exception as e:  # ruff: noqa: BLE001
            logger.warning("Error stopping monitoring: %s", e)

        # Clean up selenium session
        if self.selenium_session:
            try:
                self.selenium_session.cleanup()
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Error cleaning up selenium session: %s", e)
            finally:
                self.selenium_session = None

        # Clean up WandB API
        self.wandb_api = None
        self._is_initialized = False

        # Notify cleanup completion
        try:
            self._notify("monitor_cleanup_complete")
        except Exception as e:  # ruff: noqa: BLE001
            logger.warning("Error in cleanup notification: %s", e)

        logger.info("Cleaned up WandB run monitor")

    def __del__(self) -> None:
        """
        Destructor to ensure cleanup happens even if explicitly not called.

        This is a safety net for resource cleanup in case the object is
        garbage collected without proper cleanup being called.
        """
        try:
            # Only log if we actually have resources to clean up
            if logger is not None:  # pyright: ignore[reportUnnecessaryComparison] check if gc cleaned logger
                logger.debug("__del__ called, performing safety cleanup")
            if self.selenium_session is not None or self.monitor_thread is not None or self._is_initialized:
                self.cleanup()
                logger.debug("__del__ cleanup done")
        except KeyboardInterrupt:
            self.__del__()  # important that we run cleanup here
        except Exception:  # noqa: BLE001
            # Suppress all exceptions in __del__ to avoid issues during interpreter shutdown
            # This follows Python best practices for destructors
            pass

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    @classmethod
    def get_remote_monitor(
        cls,
        credentials: WandBCredentials | None = None,
        *,
        project: str,
        entity: Optional[str] = None,
        browser: str = "firefox",
        headless: bool = True,
        timeout: int = 30,
        callback: Optional[Callable[[str, Any], None]] = None,
        wandb_api: Optional[wandb.Api] = None,
        # actor options
        num_cpus: int = 1,
        name: str = "remote_wandb_run_monitor",
        actor_options: Optional[dict[str, Any]] = None,
    ) -> "ActorHandle[WandbRunMonitor] | ActorProxy[WandbRunMonitor]":
        """Create a remote WandbRunMonitor actor."""
        if actor_options is None:
            actor_options = {
                "num_cpus": num_cpus,
                "max_restarts": -1,
                "name": name,
            }
        else:
            actor_options = dict(actor_options)
            actor_options.setdefault("num_cpus", num_cpus)
            actor_options.setdefault("max_restarts", -1)
            actor_options.setdefault("name", name)

        remote_actor = (
            ray.remote(cls)
            .options(max_concurrency=1, get_if_exists=True, **actor_options)
            .remote(
                credentials,
                project=project,
                entity=entity,
                browser=browser,
                headless=headless,
                timeout=timeout,
                callback=callback,
                wandb_api=wandb_api,
            )
        )
        return remote_actor


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print("🔍 WandB Run Monitor")
    print("=" * 40)
    print("This module provides the WandBRunMonitor class for monitoring WandB runs.")
    print("For testing and example usage, please use: python final_test_monitor.py")
    print("\nUsage examples:")
    print("  python final_test_monitor.py                    # Run all tests")
    print("  python final_test_monitor.py --run-id abc123    # Test specific run")
    print("  python final_test_monitor.py threaded           # Run threaded test only")
    print("  python final_test_monitor.py quick              # Run quick status check")
