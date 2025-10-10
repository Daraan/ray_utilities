#!/usr/bin/env python3
"""
Simple WandB Run Monitor Script

This script demonstrates the exact features requested:
1. Login to WandB successfully
2. Interface function to visit WandB runs
3. Monitor artifact availability via WandB API
4. Report when artifacts are available (threaded operation)
"""

import logging
import os
import threading
import time
from pathlib import Path

import dotenv
import wandb

from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class SimpleWandBMonitor:
    """
    Simple WandB monitor that implements the exact requested features:
    1. Login to WandB successfully
    2. Visit WandB run pages
    3. Monitor artifact availability
    4. Report when artifacts are available (threaded)
    """

    def __init__(self, credentials: WandBCredentials, headless: bool = True):
        """Initialize the monitor with credentials."""
        self.credentials = credentials
        self.headless = headless
        self.selenium_session = None
        self.wandb_api = None
        self.is_logged_in = False
        self._stop_event = threading.Event()
        self._persistent_session = None  # For keeping page open

    def login(self) -> bool:
        """
        Feature 1: Login to WandB successfully.

        Returns:
            True if login successful, False otherwise
        """
        try:
            print("🔐 Logging into WandB...")

            # Create selenium session
            self.selenium_session = WandBSeleniumSession(
                self.credentials, browser="firefox", headless=self.headless, timeout=30
            )

            # Login via selenium
            with self.selenium_session as session:
                if session.login():
                    self.is_logged_in = True
                    print("✅ Successfully logged into WandB")

                    # Initialize WandB API
                    try:
                        self.wandb_api = wandb.Api()
                        print("✅ WandB API initialized")
                        return True
                    except Exception as e:
                        print(f"⚠️  WandB API initialization failed: {e}")
                        print("   Continuing with basic functionality...")
                        return True
                else:
                    print("❌ Failed to login to WandB")
                    return False

        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    def check_artifact_only(self, entity: str, project: str, run_id: str) -> bool:
        """
        Simple artifact availability check using only WandB API (no Selenium).

        This method only uses the WandB API to check if an artifact exists,
        without any browser automation or page visits.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID

        Returns:
            True if artifact exists, False otherwise
        """
        try:
            # Initialize WandB API if not already done
            if not self.wandb_api:
                self.wandb_api = wandb.Api()
                print("✅ WandB API initialized")

            # Construct artifact name
            history_artifact_name = f"{entity}/{project}/run-{run_id}-history:latest"
            print(f"🔍 Checking artifact: {history_artifact_name}")

            # Check if artifact exists
            exists = self.wandb_api.artifact_exists(history_artifact_name)

            if exists:
                print(f"✅ Artifact exists: {history_artifact_name}")
            else:
                print(f"❌ Artifact does not exist: {history_artifact_name}")

            return exists

        except wandb.CommError as e:
            print(f"⚠️  WandB API communication error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error checking artifact: {e}")
            return False

    def monitor_artifact_only(
        self, entity: str, project: str, run_id: str, check_interval: float = 5.0, max_wait_time: float = 300.0
    ) -> bool:
        """
        Monitor artifact availability using only WandB API (no Selenium).

        This method continuously checks for artifact availability using only
        the WandB API, without any browser automation.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID
            check_interval: Seconds between checks
            max_wait_time: Maximum time to wait

        Returns:
            True if artifact becomes available, False if timeout or error
        """
        try:
            # Initialize WandB API if not already done
            if not self.wandb_api:
                self.wandb_api = wandb.Api()
                print("✅ WandB API initialized")

            # Construct artifact name
            history_artifact_name = f"{entity}/{project}/run-{run_id}-history:latest"
            print(f"🔍 Monitoring artifact (API only): {history_artifact_name}")
            print(f"⚙️  Check interval: {check_interval}s, Max wait: {max_wait_time}s")

            start_time = time.time()
            total_wait_time = 0.0

            while total_wait_time < max_wait_time and not self._stop_event.is_set():
                try:
                    # Check if artifact exists
                    if self.wandb_api.artifact_exists(history_artifact_name):
                        elapsed_time = time.time() - start_time
                        print(f"🎉 Artifact found after {elapsed_time:.1f} seconds!")
                        print(f"   Artifact: {history_artifact_name}")
                        return True

                    # Artifact not available yet
                    print(f"⏳ Artifact not found (waited {total_wait_time:.1f}s)...")

                    # Wait before next check
                    time.sleep(check_interval)
                    total_wait_time = time.time() - start_time

                except wandb.CommError as e:
                    print(f"⚠️  WandB API error: {e} (retrying...)")
                    time.sleep(check_interval)
                    total_wait_time = time.time() - start_time
                    continue

            # Timeout reached
            elapsed_time = time.time() - start_time
            print(f"⏰ Timeout after {max_wait_time}s. Artifact not found.")
            print(f"   Total time waited: {elapsed_time:.1f}s")
            return False

        except Exception as e:
            print(f"❌ Error during API monitoring: {e}")
            return False

    def visit_run_page_and_monitor(
        self, entity: str, project: str, run_id: str, check_interval: float = 5.0, max_wait_time: float = 300.0
    ) -> bool:
        """
        Visit run page and monitor artifact while keeping the page open.

        This method combines visiting the run page with artifact monitoring,
        keeping the browser session open throughout the monitoring process.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID
            check_interval: Seconds between artifact checks
            max_wait_time: Maximum time to wait for artifact

        Returns:
            True if artifact becomes available, False if timeout or error
        """
        if not self.is_logged_in:
            print("❌ Not logged in. Please call login() first.")
            return False

        # Construct the run URL and artifact name
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        history_artifact_name = f"{entity}/{project}/run-{run_id}-history:latest"

        print(f"🌐 Visiting and staying on run page: {run_url}")
        print(f"🔍 Monitoring artifact: {history_artifact_name}")

        try:
            # Create a persistent selenium session
            self._persistent_session = WandBSeleniumSession(
                self.credentials, browser="firefox", headless=self.headless, timeout=30
            )

            # Use the session as a context manager but keep it open
            with self._persistent_session as session:
                # Login first
                if not session.login():
                    print("❌ Failed to login for page visit")
                    return False

                # Visit the run page
                if not session.visit_wandb_page(run_url):
                    print("❌ Failed to visit run page")
                    return False

                print("✅ Successfully visited run page - staying on page while monitoring...")
                print(f"📍 Browser is now on: {session.driver.current_url}")

                # Now monitor the artifact while staying on the page
                start_time = time.time()
                total_wait_time = 0.0

                while total_wait_time < max_wait_time and not self._stop_event.is_set():
                    try:
                        # Check if artifact exists via API
                        if self.wandb_api and self.wandb_api.artifact_exists(history_artifact_name):
                            elapsed_time = time.time() - start_time
                            print(f"🎉 Artifact is now available after {elapsed_time:.1f} seconds!")
                            print(f"   Artifact: {history_artifact_name}")
                            print(f"📍 Browser stayed on: {session.driver.current_url}")
                            return True

                        # Artifact not available yet
                        print(f"⏳ Artifact not available yet (waited {total_wait_time:.1f}s)...")
                        print(f"📍 Still on page: {session.driver.current_url}")

                        # Wait before next check
                        time.sleep(check_interval)
                        total_wait_time = time.time() - start_time

                    except wandb.CommError as e:
                        print(f"⚠️  WandB API error: {e}")
                        time.sleep(check_interval)
                        total_wait_time = time.time() - start_time
                        continue

                # Timeout reached
                elapsed_time = time.time() - start_time
                print(f"⏰ Timeout reached after {max_wait_time}s. Artifact not available.")
                print(f"📍 Browser was on: {session.driver.current_url} throughout monitoring")
                return False

        except Exception as e:
            print(f"❌ Error during page visit and monitoring: {e}")
            return False
        finally:
            # Clean up the persistent session
            if self._persistent_session:
                print("🧹 Closing browser session...")
                self._persistent_session.cleanup()
                self._persistent_session = None

    def visit_run_page(self, entity: str, project: str, run_id: str) -> bool:
        """
        Feature 2: Interface function to visit WandB runs.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID

        Returns:
            True if page visited successfully, False otherwise
        """
        if not self.is_logged_in:
            print("❌ Not logged in. Please call login() first.")
            return False

        # Construct the run URL
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        try:
            print(f"🌐 Visiting run page: {run_url}")

            # Create a new selenium session for page visit
            with WandBSeleniumSession(
                self.credentials, browser="firefox", headless=self.headless, timeout=30
            ) as session:
                # Login first
                if not session.login():
                    print("❌ Failed to login for page visit")
                    return False

                # Visit the run page
                if session.visit_wandb_page(run_url):
                    print(f"✅ Successfully visited run page")
                    return True
                else:
                    print(f"❌ Failed to visit run page")
                    return False

        except Exception as e:
            print(f"❌ Error visiting run page: {e}")
            return False

    def monitor_artifact_availability(
        self, entity: str, project: str, run_id: str, check_interval: float = 5.0, max_wait_time: float = 300.0
    ) -> bool:
        """
        Feature 3: Monitor artifact availability via WandB API.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID
            check_interval: Seconds between checks
            max_wait_time: Maximum time to wait

        Returns:
            True if artifact becomes available, False if timeout or error
        """
        if not self.wandb_api:
            print("❌ WandB API not available. Please ensure login() succeeded.")
            return False

        # Construct artifact name as specified
        history_artifact_name = f"{entity}/{project}/run-{run_id}-history:latest"
        print(f"🔍 Monitoring artifact: {history_artifact_name}")

        start_time = time.time()
        total_wait_time = 0.0

        try:
            while total_wait_time < max_wait_time and not self._stop_event.is_set():
                try:
                    # Check artifact existence as specified in the request
                    if self.wandb_api.artifact_exists(history_artifact_name):
                        elapsed_time = time.time() - start_time
                        print(f"🎉 Artifact is now available after {elapsed_time:.1f} seconds!")
                        print(f"   Artifact: {history_artifact_name}")
                        return True

                    # Artifact not available yet
                    print(f"⏳ Artifact not available yet (waited {total_wait_time:.1f}s)...")

                    # Wait before next check
                    time.sleep(check_interval)
                    total_wait_time = time.time() - start_time

                except wandb.CommError as e:
                    print(f"⚠️  WandB API error: {e}")
                    time.sleep(check_interval)
                    total_wait_time = time.time() - start_time
                    continue

            # Timeout reached
            print(f"⏰ Timeout reached after {max_wait_time}s. Artifact not available.")
            return False

        except Exception as e:
            print(f"❌ Error monitoring artifact: {e}")
            return False

    def run_threaded_monitor(
        self, entity: str, project: str, run_id: str, check_interval: float = 5.0, max_wait_time: float = 300.0
    ) -> threading.Thread:
        """
        Feature 4: Run the monitoring in a thread and report when artifacts are available.

        This combines features 2 and 3 in a threaded operation.

        Args:
            entity: WandB entity name
            project: WandB project name
            run_id: WandB run ID
            check_interval: Seconds between checks
            max_wait_time: Maximum time to wait

        Returns:
            Thread object for the monitoring operation
        """

        def threaded_monitor():
            """Internal function that runs in the thread."""
            try:
                print(f"🧵 Starting threaded monitoring for run {run_id}")

                # Step 1: Visit the run page
                print("📄 Step 1: Visiting run page...")
                page_success = self.visit_run_page(entity, project, run_id)
                if not page_success:
                    print("⚠️  Failed to visit run page, but continuing with artifact monitoring...")

                # Step 2: Monitor artifact availability
                print("🔍 Step 2: Starting artifact monitoring...")
                artifact_success = self.monitor_artifact_availability(
                    entity, project, run_id, check_interval, max_wait_time
                )

                # Step 3: Report final result
                if artifact_success:
                    print(f"✅ THREAD COMPLETE: Artifact for run {run_id} is now available!")
                else:
                    print(f"❌ THREAD COMPLETE: Artifact for run {run_id} not available within timeout")

            except Exception as e:
                print(f"💥 Thread error: {e}")

        # Create and start the thread
        thread = threading.Thread(target=threaded_monitor, name=f"wandb-monitor-{run_id}", daemon=True)
        thread.start()

        print(f"🚀 Started monitoring thread for run {run_id}")
        return thread

    def stop_monitoring(self):
        """Stop any active monitoring."""
        self._stop_event.set()
        print("🛑 Stop signal sent to monitoring threads")

    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        if self.selenium_session:
            self.selenium_session.cleanup()
        if self._persistent_session:
            self._persistent_session.cleanup()
            self._persistent_session = None
        print("🧹 Cleanup completed")


def main():
    """Demonstrate the requested features."""
    print("🎯 WandB Simple Monitor Demo")
    print("=" * 50)

    # Setup credentials from environment
    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        print("Please set WANDB_VIEWER_MAIL and WANDB_VIEWER_PW")
        return

    # Create monitor
    monitor = SimpleWandBMonitor(credentials, headless=True)

    try:
        # Feature 1: Login
        print("\n🔐 Testing Feature 1: Login")
        if not monitor.login():
            print("❌ Login failed, cannot continue")
            return

        # Example run parameters (replace with real values for testing)
        entity = "daraan"
        project = "dev-workspace"
        run_id = "your-run-id-here"  # Replace with actual run ID

        # Feature 2: Visit run page (synchronous)
        print(f"\n🌐 Testing Feature 2: Visit run page")
        monitor.visit_run_page(entity, project, run_id)

        # Feature 3: Monitor artifact (synchronous)
        print(f"\n🔍 Testing Feature 3: Monitor artifact availability")
        monitor.monitor_artifact_availability(entity, project, run_id, check_interval=2.0, max_wait_time=20.0)

        # Feature 4: Threaded monitoring
        print(f"\n🧵 Testing Feature 4: Threaded monitoring")
        thread = monitor.run_threaded_monitor(entity, project, run_id, check_interval=3.0, max_wait_time=30.0)

        # Do other work while monitoring runs
        print("💼 Doing other work while monitoring runs in background...")
        for i in range(10):
            print(f"   Working... {i + 1}/10")
            time.sleep(2)

        # Wait for thread to complete
        print("⏳ Waiting for monitoring thread to complete...")
        thread.join(timeout=60)

        if thread.is_alive():
            print("⚠️  Thread still running, stopping...")
            monitor.stop_monitoring()

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        monitor.cleanup()

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()
