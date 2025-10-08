"""
WandB Run Monitor Script

This script provides threaded functionality to:
1. Login to WandB using Selenium
2. Visit specific WandB run pages
3. Monitor artifact availability using WandB API
4. Report when artifacts become available
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import dotenv
import wandb
from selenium.common.exceptions import TimeoutException, WebDriverException

from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())
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


class WandBRunMonitor:
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
        credentials: WandBCredentials,
        *,
        browser: str = "firefox",
        headless: bool = True,
        timeout: int = 30,
        callback: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        Initialize the WandB run monitor.

        Args:
            credentials: WandB login credentials
            browser: Browser to use ("chrome" or "firefox")
            headless: Whether to run browser in headless mode
            timeout: Default timeout for web elements
            callback: Optional callback function for status updates
        """
        self.credentials = credentials
        self.browser = browser
        self.headless = headless
        self.timeout = timeout
        self.callback = callback

        self.selenium_session: Optional[WandBSeleniumSession] = None
        self.wandb_api: Optional[wandb.Api] = None
        self.is_initialized = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _notify(self, status: str, data: Any = None) -> None:
        """Send notification via callback if available."""
        if self.callback:
            try:
                self.callback(status, data)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Callback error: %s", e)

    def initialize(self) -> bool:
        """
        Initialize the monitor by setting up Selenium session and WandB API.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._notify("initializing_monitor")

            # Initialize Selenium session
            self.selenium_session = WandBSeleniumSession(
                self.credentials,
                browser=self.browser,
                headless=self.headless,
                timeout=self.timeout,
                callback=self._selenium_callback,
            )

            # Setup driver and login
            with self.selenium_session as session:
                if not session.login():
                    logger.error("Failed to login via Selenium")
                    self._notify("initialization_failed", "Selenium login failed")
                    return False

                # Initialize WandB API
                if session.credentials.api_key:
                    self.wandb_api = wandb.Api(api_key=session.credentials.api_key)
                else:
                    # Try to initialize API without explicit key (using environment/config)
                    self.wandb_api = wandb.Api()

                logger.info("Successfully initialized monitor")
                self.is_initialized = True
                self._notify("monitor_initialized")
                return True

        except Exception as e:
            logger.error("Failed to initialize monitor: %s", e)
            self._notify("initialization_failed", str(e))
            return False

    def _selenium_callback(self, status: str, data: Any = None) -> None:
        """Internal callback to forward Selenium status updates."""
        self._notify(f"selenium_{status}", data)

    def visit_run_page(self, entity: str, project: str, run_id: str) -> bool:
        """
        Visit a specific WandB run page.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID

        Returns:
            True if page visited successfully, False otherwise
        """
        if not self.selenium_session or not self.is_initialized:
            logger.error("Monitor not initialized")
            return False

        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        try:
            self._notify("visiting_run_page", {"url": run_url})

            # Use existing session if available, otherwise create new context
            if self.selenium_session.driver:
                success = self.selenium_session.visit_wandb_page(run_url)
            else:
                with self.selenium_session as session:
                    success = session.visit_wandb_page(run_url)

            if success:
                logger.info("Successfully visited run page: %s", run_url)
                self._notify("run_page_visited", {"url": run_url})
            else:
                logger.error("Failed to visit run page: %s", run_url)
                self._notify("run_page_visit_failed", {"url": run_url})

            return success

        except Exception as e:
            logger.error("Error visiting run page %s: %s", run_url, e)
            self._notify("run_page_visit_error", {"url": run_url, "error": str(e)})
            return False

    def monitor_artifact(self, config: RunMonitorConfig) -> MonitorResult:
        """
        Monitor artifact availability for a specific run.

        Args:
            config: Configuration for the monitoring operation

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

        # Construct artifact name
        history_artifact_name = f"{config.entity}/{config.project}/run-{config.run_id}-history:latest"

        self._notify(
            "starting_artifact_monitor",
            {
                "artifact_name": history_artifact_name,
                "check_interval": config.check_interval,
                "max_wait_time": config.max_wait_time,
            },
        )

        start_time = time.time()
        total_wait_time = 0.0

        try:
            while total_wait_time < config.max_wait_time and not self._stop_event.is_set():
                try:
                    # Check if artifact exists
                    if self.wandb_api.artifact_exists(history_artifact_name):
                        elapsed_time = time.time() - start_time
                        logger.info(
                            "Artifact %s is now available after %.2f seconds", history_artifact_name, elapsed_time
                        )

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
                    time.sleep(config.check_interval)
                    total_wait_time = time.time() - start_time

                except wandb.CommError as e:
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

                    # Continue checking after a brief delay
                    time.sleep(config.check_interval)
                    total_wait_time = time.time() - start_time
                    continue

            # Timeout or stopped
            elapsed_time = time.time() - start_time
            if self._stop_event.is_set():
                error_msg = "Monitoring stopped by user"
            else:
                error_msg = f"Timeout after {config.max_wait_time} seconds"

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

    def monitor_run_threaded(self, config: RunMonitorConfig) -> threading.Thread:
        """
        Start monitoring a run in a separate thread.

        Args:
            config: Configuration for the monitoring operation

        Returns:
            Thread object for the monitoring process
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitor thread already running")
            return self.monitor_thread

        self.monitor_thread = threading.Thread(
            target=self._threaded_monitor_run,
            args=(config,),
            name=f"wandb-monitor-{config.run_id}",
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("Started WandB run monitor thread for run %s", config.run_id)
        self._notify("monitor_thread_started", {"run_id": config.run_id})

        return self.monitor_thread

    def _threaded_monitor_run(self, config: RunMonitorConfig) -> None:
        """Internal method for threaded run monitoring."""
        try:
            # Initialize if not already done
            if not self.is_initialized:
                if not self.initialize():
                    self._notify("threaded_monitor_failed", "Initialization failed")
                    return

            # Visit the run page
            page_visited = self.visit_run_page(config.entity, config.project, config.run_id)
            if not page_visited:
                logger.warning("Failed to visit run page, but continuing with artifact monitoring")

            # Monitor the artifact
            result = self.monitor_artifact(config)

            self._notify(
                "threaded_monitor_complete",
                {
                    "config": config,
                    "result": result,
                },
            )

        except Exception as e:
            logger.error("Thread execution error: %s", e)
            self._notify("threaded_monitor_error", str(e))

    def stop_monitoring(self) -> None:
        """Stop any active monitoring."""
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
            logger.info("Stopped monitoring thread")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_monitoring()

        if self.selenium_session:
            self.selenium_session.cleanup()
            self.selenium_session = None

        self.wandb_api = None
        self.is_initialized = False
        self._notify("monitor_cleanup_complete")
        logger.info("Cleaned up WandB run monitor")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def example_usage():
    """Example usage of the WandB run monitor."""

    def status_callback(status: str, data: Any = None):
        """Example callback for status updates."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {status}: {data}")

    # Setup credentials
    credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])

    # Configuration for monitoring
    config = RunMonitorConfig(
        entity="daraan",
        project="dev-workspace",
        run_id="your-run-id-here",  # Replace with actual run ID
        check_interval=5.0,
        max_wait_time=300.0,
    )

    print("🔍 Starting WandB Run Monitor Example")
    print("=" * 50)

    try:
        with WandBRunMonitor(
            credentials,
            browser="firefox",
            headless=True,
            callback=status_callback,
        ) as monitor:
            # Example 1: Synchronous monitoring
            print("\n📋 Example 1: Synchronous monitoring")
            if monitor.initialize():
                result = monitor.monitor_artifact(config)
                print(f"📊 Monitoring result: {result}")

            # Example 2: Threaded monitoring
            print("\n🧵 Example 2: Threaded monitoring")
            thread = monitor.monitor_run_threaded(config)

            # Do other work while monitoring happens
            print("💼 Doing other work while monitoring runs in background...")
            for i in range(10):
                print(f"   Working... {i + 1}/10")
                time.sleep(2)

            # Wait for monitoring to complete
            print("⏳ Waiting for monitoring to complete...")
            thread.join(timeout=30)

            if thread.is_alive():
                print("⚠️  Monitoring still running, stopping...")
                monitor.stop_monitoring()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ Example completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run example
    example_usage()
