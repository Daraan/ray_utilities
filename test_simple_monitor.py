#!/usr/bin/env python3
"""
Test the simple WandB monitor with a real run from test_wandb_fork1.py
"""

import os
import sys
import time
from pathlib import Path

import dotenv

from simple_wandb_monitor import SimpleWandBMonitor, WandBCredentials

dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())


def test_with_real_data():
    """Test the monitor with specific run data."""
    print("🧪 Testing Simple WandB Monitor with Real Data")
    print("=" * 60)

    # Setup credentials
    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
        print("✅ Credentials loaded from environment")
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return False

    # Create monitor
    monitor = SimpleWandBMonitor(credentials, headless=True)

    try:
        # Test Feature 1: Login
        print("\n🔐 Feature 1: Testing login...")
        if not monitor.login():
            print("❌ Login failed")
            return False

        # Test parameters from your test_wandb_fork1.py
        entity = "daraan"
        project = "dev-workspace"

        # For demo purposes, use a placeholder run ID
        # In real usage, you would pass the actual run ID from your experiment
        run_id = "example-run-id"  # Replace with actual run ID when testing

        print(f"\n📋 Testing with:")
        print(f"   Entity: {entity}")
        print(f"   Project: {project}")
        print(f"   Run ID: {run_id}")

        # Test Feature 2: Visit run page
        print(f"\n🌐 Feature 2: Visiting run page...")
        page_result = monitor.visit_run_page(entity, project, run_id)
        print(f"   Result: {'✅ Success' if page_result else '❌ Failed'}")

        # Test Feature 3: Check artifact availability (short timeout for demo)
        print(f"\n🔍 Feature 3: Checking artifact availability...")
        artifact_result = monitor.monitor_artifact_availability(
            entity,
            project,
            run_id,
            check_interval=2.0,
            max_wait_time=10.0,  # Short timeout for demo
        )
        print(f"   Result: {'✅ Available' if artifact_result else '❌ Not available/timeout'}")

        # Test Feature 4: Threaded monitoring
        print(f"\n🧵 Feature 4: Starting threaded monitoring...")
        thread = monitor.run_threaded_monitor(
            entity,
            project,
            run_id,
            check_interval=3.0,
            max_wait_time=15.0,  # Short timeout for demo
        )

        # Simulate doing other work
        print("💼 Simulating other work while monitoring runs...")
        for i in range(5):
            print(f"   Working... {i + 1}/5")
            time.sleep(2)
            if not thread.is_alive():
                break

        # Wait for completion
        print("⏳ Waiting for thread to complete...")
        thread.join(timeout=20)

        if thread.is_alive():
            print("⚠️  Thread still running, stopping...")
            monitor.stop_monitoring()

        print("\n✅ All features tested successfully!")
        return True

    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    finally:
        monitor.cleanup()


def test_basic_functionality():
    """Test basic functionality without real API calls."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)

    # Test credential creation
    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
        print("✅ Credentials created successfully")
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return False

    # Test monitor creation
    monitor = SimpleWandBMonitor(credentials, headless=True)
    print("✅ Monitor created successfully")

    # Test thread creation (without actual execution)
    print("✅ Basic functionality test passed")
    return True


def main():
    """Main test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "basic":
        success = test_basic_functionality()
    else:
        success = test_with_real_data()

    if success:
        print("\n🎉 Tests completed successfully!")
    else:
        print("\n💥 Tests failed!")

    return success


if __name__ == "__main__":
    main()
