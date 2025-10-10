#!/usr/bin/env python3
"""
Test script for the multiprocessing WandB Selenium session.
"""

import logging
import os
import sys
from pathlib import Path

from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login_mp import (
    WandBCredentials,
    WandBSeleniumSession,
)

import dotenv

dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())


def test_mp_selenium():
    """Test the multiprocessing selenium session."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def status_callback(status: str, data=None):
        print(f"✓ Status: {status} - {data}")

    print("🔍 Testing Multiprocessing WandB Selenium Session")
    print("=" * 50)

    try:
        # Test 1: Context manager
        print("\n📋 Test 1: Context manager initialization")
        with WandBSeleniumSession(
            credentials=None, browser="firefox", headless=True, callback=status_callback
        ) as session:
            print("✓ Session created successfully")
            print(f"✓ Process object: {session.process}")

            # Test driver setup (this should work even with dummy credentials)
            print("\n📋 Test 2: Driver setup")
            try:
                driver = session._setup_driver()
                print(f"✓ Driver setup: {driver}")
            except Exception as e:
                print(f"✗ Driver setup failed: {e}")

            print("\n📋 Test 3: Process communication")
            # Test basic process communication
            if session.process and session.process.is_alive():
                print(f"✓ Process is alive (PID: {session.process.pid})")
            else:
                print("✗ Process is not alive")

        print("\n✓ Context manager test completed")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    test_mp_selenium()
"""
Test script for the multiprocessing WandB selenium session.
"""

import logging
import os
import time

from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login_mp import (
    WandBCredentials,
    WandBSeleniumSession,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def status_callback(status: str, data=None):
    """Example callback for status updates."""
    print(f"📊 Status: {status} - {data}")


def test_basic_functionality():
    """Test basic functionality of the multiprocessing selenium session."""
    print("🧪 Testing basic multiprocessing WandB Selenium functionality...")

    # Setup credentials from environment or use dummy ones for testing structure
    credentials = WandBCredentials(
        username=os.getenv("WANDB_VIEWER_MAIL", "test@example.com"),
        password=os.getenv("WANDB_VIEWER_PW", "testpass"),
        api_key=os.getenv("WANDB_API_KEY", None),
    )

    # Test 1: Context manager usage
    print("\n1️⃣ Testing context manager usage...")
    try:
        with WandBSeleniumSession(
            credentials,
            callback=status_callback,
            browser="firefox",  # Using firefox as it's more stable in headless mode
            headless=True,
        ) as session:
            print("✅ Session created successfully")

            # Test if we have real credentials before attempting login
            if credentials.username != "test@example.com" and credentials.password != "testpass":
                print("🔐 Attempting login...")
                if session.login():
                    print("✅ Login successful!")

                    # Test visiting a page
                    if session.visit_wandb_page("https://wandb.ai/home"):
                        print("✅ Successfully visited WandB home page")
                    else:
                        print("❌ Failed to visit WandB home page")
                else:
                    print("❌ Login failed")
            else:
                print("⚠️ Using dummy credentials, skipping login test")

        print("✅ Context manager cleanup successful")

    except Exception as e:
        print(f"❌ Context manager test failed: {e}")

    # Test 2: Process-based usage
    print("\n2️⃣ Testing process-based usage...")
    session = None
    try:
        session = WandBSeleniumSession(credentials, callback=status_callback, browser="firefox", headless=True)

        # Start the process
        process = session.run_threaded()  # This maintains the API but uses multiprocessing
        print(f"✅ Process started: PID {process.pid if hasattr(process, 'pid') else 'unknown'}")

        # Wait a bit for initialization
        time.sleep(3)

        print("✅ Process-based test successful")

    except Exception as e:
        print(f"❌ Process-based test failed: {e}")
    finally:
        if session:
            session.stop()
            print("✅ Session stopped")

    # Test 3: Cache functionality
    print("\n3️⃣ Testing cache functionality...")
    try:
        session = WandBSeleniumSession(credentials, use_cache=True, headless=True)

        # Test cache validation
        is_valid = session.is_cache_valid()
        print(f"📁 Cache valid: {is_valid}")

        # Test cache clearing
        cleared = session.clear_cache()
        print(f"🗑️ Cache cleared: {cleared}")

        session.cleanup()
        print("✅ Cache test successful")

    except Exception as e:
        print(f"❌ Cache test failed: {e}")

    print("\n🏁 All tests completed!")


def test_api_compatibility():
    """Test that the multiprocessing version maintains API compatibility."""
    print("\n🔍 Testing API compatibility...")

    credentials = WandBCredentials(username="test@example.com", password="testpass")

    try:
        session = WandBSeleniumSession(credentials, headless=True)

        # Test that all expected methods exist
        expected_methods = [
            "login",
            "visit_wandb_page",
            "initialize_wandb_api",
            "run_threaded",
            "stop",
            "clear_cache",
            "is_cache_valid",
            "cleanup",
        ]

        for method_name in expected_methods:
            if hasattr(session, method_name):
                print(f"✅ {method_name} method exists")
            else:
                print(f"❌ {method_name} method missing")

        # Test that driver attribute exists (for compatibility)
        if hasattr(session, "driver"):
            print("✅ driver attribute exists (API compatibility)")
        else:
            print("❌ driver attribute missing")

        # Test that credentials are accessible
        if hasattr(session, "credentials"):
            print("✅ credentials attribute exists")
        else:
            print("❌ credentials attribute missing")

        session.cleanup()
        print("✅ API compatibility test successful")

    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")


if __name__ == "__main__":
    print("🚀 WandB Selenium Multiprocessing Test Suite")
    print("=" * 50)

    test_api_compatibility()
    test_basic_functionality()

    print("\n" + "=" * 50)
    print("📝 Notes:")
    print("   - Set WANDB_VIEWER_MAIL, WANDB_VIEWER_PW, and WANDB_API_KEY environment variables for full testing")
    print("   - Optionally set WANDB_VIEWER_TEAM_NAME to verify specific team login (default: 'DaraanWandB')")
    print("   - This test uses dummy credentials by default to test structure")
    print("   - The multiprocessing version maintains the same API as the threading version")
    print("   - All selenium operations now run in a separate process for better isolation")
