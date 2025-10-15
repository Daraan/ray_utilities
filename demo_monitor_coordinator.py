#!/usr/bin/env python3
"""
Example script that uses the WandB Run Monitor in a threaded manner.

This script demonstrates how to start monitoring from another script
and handle the results asynchronously.
"""

import logging
import os
import sys
import time
from pathlib import Path

import dotenv

from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login_mp import WandBCredentials
from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor, RunMonitorConfig


dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class MonitorCoordinator:
    """
    Coordinates multiple run monitoring operations.

    This demonstrates how to use the WandB run monitor from another script
    in a production-like scenario.
    """

    def __init__(self):
        self.active_monitors = {}
        self.results = {}

    def status_callback(self, monitor_id: str):
        """Create a status callback for a specific monitor."""

        def callback(status: str, data=None):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Monitor {monitor_id}: {status}")

            if data:
                if status == "artifact_available":
                    artifact_name = data.get("artifact_name", "unknown")
                    wait_time = data.get("wait_time", 0)
                    print(f"  🎉 Artifact {artifact_name} is now available after {wait_time:.1f}s!")

                elif status == "artifact_not_available":
                    wait_time = data.get("wait_time", 0)
                    print(f"  ⏳ Still waiting... ({wait_time:.1f}s elapsed)")

                elif status == "threaded_monitor_complete":
                    result = data.get("result")
                    if result:
                        self.results[monitor_id] = result
                        if result.artifact_available:
                            print(f"  ✅ Monitor {monitor_id} completed successfully!")
                        else:
                            print(f"  ❌ Monitor {monitor_id} completed with issues: {result.error_message}")

                elif status == "artifact_monitor_timeout":
                    print(f"  ⏰ Monitor {monitor_id} timed out")

                elif status == "threaded_monitor_error":
                    print(f"  💥 Monitor {monitor_id} encountered an error: {data}")

        return callback

    def start_monitoring(self, monitor_id: str, config: RunMonitorConfig) -> bool:
        """
        Start monitoring a specific run.

        Args:
            monitor_id: Unique identifier for this monitoring operation
            config: Configuration for the monitoring

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if monitor_id in self.active_monitors:
            logger.warning("Monitor %s is already active", monitor_id)
            return False

        try:
            # Setup credentials
            credentials = WandBCredentials(
                username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"]
            )

            # Create monitor with specific callback
            monitor = WandbRunMonitor(
                credentials,
                project="dev-workspace",
                browser="firefox",
                headless=True,
                timeout=30,
                callback=self.status_callback(monitor_id),
            )

            # Start threaded monitoring
            thread = monitor.monitor_run_threaded(config)

            # Store the monitor for later reference
            self.active_monitors[monitor_id] = {
                "monitor": monitor,
                "thread": thread,
                "config": config,
                "start_time": time.time(),
            }

            logger.info("Started monitoring for %s (run: %s)", monitor_id, config.run_id)
            return True

        except Exception as e:
            logger.error("Failed to start monitoring for %s: %s", monitor_id, e)
            return False

    def check_monitor_status(self, monitor_id: str) -> dict:
        """Check the status of a specific monitor."""
        if monitor_id not in self.active_monitors:
            return {"status": "not_found"}

        monitor_info = self.active_monitors[monitor_id]
        thread = monitor_info["thread"]

        status = {
            "monitor_id": monitor_id,
            "is_alive": thread.is_alive(),
            "start_time": monitor_info["start_time"],
            "elapsed_time": time.time() - monitor_info["start_time"],
            "config": monitor_info["config"],
        }

        if monitor_id in self.results:
            status["result"] = self.results[monitor_id]

        return status

    def stop_monitor(self, monitor_id: str) -> bool:
        """Stop a specific monitor."""
        if monitor_id not in self.active_monitors:
            logger.warning("Monitor %s not found", monitor_id)
            return False

        try:
            monitor_info = self.active_monitors[monitor_id]
            monitor = monitor_info["monitor"]

            # Stop the monitor
            monitor.stop_monitoring()

            # Clean up
            monitor.cleanup()

            # Remove from active monitors
            del self.active_monitors[monitor_id]

            logger.info("Stopped monitor %s", monitor_id)
            return True

        except Exception as e:
            logger.error("Error stopping monitor %s: %s", monitor_id, e)
            return False

    def stop_all_monitors(self):
        """Stop all active monitors."""
        monitor_ids = list(self.active_monitors.keys())
        for monitor_id in monitor_ids:
            self.stop_monitor(monitor_id)

    def get_active_monitors(self) -> list:
        """Get list of active monitor IDs."""
        return list(self.active_monitors.keys())

    def wait_for_completion(self, timeout: float = 300.0):
        """Wait for all monitors to complete."""
        start_time = time.time()

        while self.active_monitors and (time.time() - start_time) < timeout:
            # Check each monitor
            completed_monitors = []

            for monitor_id, monitor_info in self.active_monitors.items():
                if not monitor_info["thread"].is_alive():
                    completed_monitors.append(monitor_id)

            # Clean up completed monitors
            for monitor_id in completed_monitors:
                monitor_info = self.active_monitors[monitor_id]
                monitor_info["monitor"].cleanup()
                del self.active_monitors[monitor_id]
                logger.info("Monitor %s completed", monitor_id)

            time.sleep(1)

        # Stop any remaining monitors
        if self.active_monitors:
            logger.warning("Timeout reached, stopping remaining monitors")
            self.stop_all_monitors()


def demo_single_run_monitoring():
    """Demonstrate monitoring a single run."""
    print("🎯 Demo: Single Run Monitoring")
    print("=" * 50)

    coordinator = MonitorCoordinator()

    # Example configuration - replace with actual values
    config = RunMonitorConfig(
        entity="daraan",
        project="dev-workspace",
        run_id="your-run-id-here",  # Replace with actual run ID
        check_interval=5.0,
        max_wait_time=60.0,  # Shorter timeout for demo
    )

    try:
        # Start monitoring
        success = coordinator.start_monitoring("demo-run", config)

        if success:
            print("✅ Monitoring started successfully")

            # Wait for completion
            print("⏳ Waiting for monitoring to complete...")
            coordinator.wait_for_completion(timeout=120.0)

            # Check final results
            status = coordinator.check_monitor_status("demo-run")
            print(f"📊 Final status: {status}")

        else:
            print("❌ Failed to start monitoring")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        coordinator.stop_all_monitors()
        print("🧹 Cleanup completed")


def demo_multiple_run_monitoring():
    """Demonstrate monitoring multiple runs simultaneously."""
    print("🎯 Demo: Multiple Run Monitoring")
    print("=" * 50)

    coordinator = MonitorCoordinator()

    # Example configurations - replace with actual values
    configs = [
        RunMonitorConfig(
            entity="daraan",
            project="dev-workspace",
            run_id="run-1",  # Replace with actual run ID
            check_interval=5.0,
            max_wait_time=60.0,
        ),
        RunMonitorConfig(
            entity="daraan",
            project="dev-workspace",
            run_id="run-2",  # Replace with actual run ID
            check_interval=5.0,
            max_wait_time=60.0,
        ),
    ]

    try:
        # Start monitoring multiple runs
        for i, config in enumerate(configs):
            monitor_id = f"run-{i + 1}"
            success = coordinator.start_monitoring(monitor_id, config)
            if success:
                print(f"✅ Started monitoring {monitor_id}")
            else:
                print(f"❌ Failed to start monitoring {monitor_id}")

        # Monitor progress
        print("⏳ Monitoring multiple runs...")

        # Check status periodically
        for _ in range(12):  # Check for 1 minute
            active = coordinator.get_active_monitors()
            if not active:
                break

            print(f"   Active monitors: {len(active)} ({', '.join(active)})")
            time.sleep(5)

        # Wait for completion
        coordinator.wait_for_completion(timeout=120.0)

        # Show final results
        print("\n📊 Final Results:")
        for monitor_id in ["run-1", "run-2"]:
            if monitor_id in coordinator.results:
                result = coordinator.results[monitor_id]
                print(
                    f"  {monitor_id}: {'✅' if result.artifact_available else '❌'} "
                    f"(wait time: {result.wait_time:.1f}s)"
                )

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        coordinator.stop_all_monitors()
        print("🧹 Cleanup completed")


def main():
    """Main function to run the demo."""
    print("🚀 WandB Run Monitor Demo")
    print("=" * 80)

    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        demo_multiple_run_monitoring()
    else:
        demo_single_run_monitoring()

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()
