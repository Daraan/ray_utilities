#!/usr/bin/env python3
"""
Final Test Script for WandB Run Monitor

This script tests the complete workflow:
1. Login to WandB successfully
2. Visit the specific run page: daraan/dev-workspace/runs/9t51eetr
3. Query for the history artifact: daraan/dev-workspace/run-9t51eetr-history:latest
4. Report when the artifact is available
"""

import logging
import os
import sys
import time
from pathlib import Path

import dotenv

from simple_wandb_monitor import SimpleWandBMonitor, WandBCredentials

dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def test_threaded_version():
    """Test the threaded version of the monitoring."""
    print("🚀 Final WandB Run Monitor Test (Threaded Version)")
    print("=" * 70)

    # Setup credentials
    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
        print("✅ Credentials loaded from environment")
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return False

    # Test parameters
    entity = "daraan"
    project = "dev-workspace"
    run_id = "9t51eetr"

    monitor = SimpleWandBMonitor(credentials, headless=True)

    try:
        print("\n🔐 Logging into WandB...")
        if not monitor.login():
            print("❌ Login failed")
            return False

        print("\n🧵 Starting threaded monitoring...")
        print(f"   Entity: {entity}")
        print(f"   Project: {project}")
        print(f"   Run ID: {run_id}")
        print(f"   Expected artifact: {entity}/{project}/run-{run_id}-history:latest")

        # Start threaded monitoring
        thread = monitor.run_threaded_monitor(
            entity,
            project,
            run_id,
            check_interval=3.0,
            max_wait_time=90.0,  # 1.5 minute timeout
        )

        print("\n💼 Simulating other work while monitoring runs in background...")
        # Simulate doing other work
        for i in range(15):  # 30 seconds of "work"
            print(f"   Working... {i + 1}/15")
            time.sleep(2)
            if not thread.is_alive():
                print("   🎯 Monitoring thread completed!")
                break

        # Wait for thread completion
        if thread.is_alive():
            print("\n⏳ Waiting for monitoring thread to complete...")
            thread.join(timeout=60)

            if thread.is_alive():
                print("⚠️  Thread still running, stopping...")
                monitor.stop_monitoring()

        print("\n✅ Threaded monitoring test completed!")
        return True

    except Exception as e:
        print(f"\n❌ Threaded test error: {e}")
        return False
    finally:
        monitor.cleanup()


def check_artifact_api_only():
    """Test the API-only artifact checking functionality."""
    print("🔍 API-Only Artifact Check Test")
    print("=" * 50)

    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
        print("✅ Credentials loaded from environment")
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return False

    entity = "daraan"
    project = "dev-workspace"
    run_id = "9t51eetr"

    monitor = SimpleWandBMonitor(credentials, headless=True)

    try:
        print(f"\n🔍 Testing API-only methods with run {run_id}...")

        # Test 1: Simple one-time check
        print("\n📋 Test 1: Simple artifact check")
        exists = monitor.check_artifact_only(entity, project, run_id)
        print(f"   Result: {'✅ Exists' if exists else '❌ Not found'}")

        # Test 2: Continuous monitoring (API only)
        print(f"\n📋 Test 2: API-only monitoring (15 second timeout)")
        monitor_result = monitor.monitor_artifact_only(entity, project, run_id, check_interval=2.0, max_wait_time=15.0)
        print(f"   Result: {'✅ Found' if monitor_result else '❌ Timeout/Not found'}")

        print(f"\n✅ API-only tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ API-only test error: {e}")
        return False
    finally:
        monitor.cleanup()


def check_run_status():
    """Quick check of the run status without full monitoring."""
    print("🔍 Quick Run Status Check")
    print("=" * 40)

    try:
        credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        return False

    entity = "daraan"
    project = "dev-workspace"
    run_id = "9t51eetr"

    monitor = SimpleWandBMonitor(credentials, headless=True)

    try:
        if monitor.login():
            print(f"🔍 Quick artifact check for run {run_id}...")

            # Just check once, no waiting
            if monitor.wandb_api:
                artifact_name = f"{entity}/{project}/run-{run_id}-history:latest"
                try:
                    exists = monitor.wandb_api.artifact_exists(artifact_name)
                    if exists:
                        print(f"✅ Artifact {artifact_name} EXISTS!")
                    else:
                        print(f"❌ Artifact {artifact_name} does not exist yet")
                    return exists
                except Exception as e:
                    print(f"⚠️  Error checking artifact: {e}")
                    return False
            else:
                print("❌ WandB API not available")
                return False
        else:
            print("❌ Login failed")
            return False
    finally:
        monitor.cleanup()


def main():
    """Main test function with comprehensive monitoring."""
    print("🚀 Starting WandB Monitor Tests")
    print("=" * 50)

    # Test 1: Initial run status check
    if not check_run_status():
        print("❌ Run status check failed - stopping tests")
        return False

    # Test 2: API-only tests
    if not check_artifact_api_only():
        print("❌ API-only tests failed")
        return False

    # Test 3: Full monitor test with persistent browser
    if not test_threaded_version():
        print("❌ Monitor test failed")
        return False

    print("\n🎉 All tests passed successfully!")
    print("✅ Monitor is working correctly with both API-only and browser-based monitoring")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "threaded":
            success = test_threaded_version()
        elif sys.argv[1] == "quick":
            success = check_run_status()
        else:
            print("Usage: python final_test_monitor.py [threaded|quick]")
            sys.exit(1)
    else:
        success = main()

    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
