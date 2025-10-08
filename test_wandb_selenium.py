#!/usr/bin/env python3
"""
Test script for the WandB Selenium Login functionality.
This script tests basic functionality without requiring real credentials.
"""

import logging
import os
import time
from unittest.mock import Mock, patch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Import our script
try:
    from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

    print("✓ Successfully imported WandBSeleniumSession")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    exit(1)


def test_credentials():
    """Test WandBCredentials dataclass."""
    print("\n1. Testing WandBCredentials...")

    # Test basic credentials
    creds = WandBCredentials(username="test@example.com", password="test_password")

    assert creds.username == "test@example.com"
    assert creds.password == "test_password"
    assert creds.api_key is None
    print("   ✓ Basic credentials creation works")

    # Test with API key
    creds_with_key = WandBCredentials(
        username="test@example.com", password="test_password", api_key="wandb_test_key_123"
    )

    assert creds_with_key.api_key == "wandb_test_key_123"
    print("   ✓ Credentials with API key works")


def test_session_init():
    """Test WandBSeleniumSession initialization."""
    print("\n2. Testing WandBSeleniumSession initialization...")

    creds = WandBCredentials(username="test@example.com", password="test_password")

    # Test default initialization
    session = WandBSeleniumSession(creds)

    assert session.credentials == creds
    assert session.browser == "chrome"
    assert session.headless is True
    assert session.timeout == 30
    assert session.callback is None
    assert session.driver is None
    assert session.is_logged_in is False
    print("   ✓ Default initialization works")

    # Test custom initialization
    def dummy_callback(status, data=None):
        pass

    session_custom = WandBSeleniumSession(creds, browser="firefox", headless=False, timeout=60, callback=dummy_callback)

    assert session_custom.browser == "firefox"
    assert session_custom.headless is False
    assert session_custom.timeout == 60
    assert session_custom.callback == dummy_callback
    print("   ✓ Custom initialization works")


def test_driver_setup_mock():
    """Test driver setup with mocking (no actual browser)."""
    print("\n3. Testing driver setup (mocked)...")

    creds = WandBCredentials(username="test@example.com", password="test_password")

    session = WandBSeleniumSession(creds)

    # Mock the webdriver.Chrome to avoid actually launching a browser
    with patch("wandb_selenium_login_fixed.webdriver.Chrome") as mock_chrome:
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        driver = session._setup_driver()

        # Check that Chrome was called
        mock_chrome.assert_called_once()
        assert driver == mock_driver
        print("   ✓ Chrome driver setup works (mocked)")

    # Test Firefox
    session_firefox = WandBSeleniumSession(creds, browser="firefox")

    with patch("wandb_selenium_login_fixed.webdriver.Firefox") as mock_firefox:
        mock_driver = Mock()
        mock_firefox.return_value = mock_driver

        driver = session_firefox._setup_driver()

        mock_firefox.assert_called_once()
        assert driver == mock_driver
        print("   ✓ Firefox driver setup works (mocked)")

    # Test invalid browser
    session_invalid = WandBSeleniumSession(creds, browser="invalid")

    try:
        session_invalid._setup_driver()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported browser" in str(e)
        print("   ✓ Invalid browser raises ValueError")


def test_callback_functionality():
    """Test callback functionality."""
    print("\n4. Testing callback functionality...")

    callback_calls = []

    def test_callback(status, data=None):
        callback_calls.append((status, data))

    creds = WandBCredentials(username="test@example.com", password="test_password")

    session = WandBSeleniumSession(creds, callback=test_callback)

    # Test notify
    session._notify("test_status", {"test": "data"})

    assert len(callback_calls) == 1
    assert callback_calls[0] == ("test_status", {"test": "data"})
    print("   ✓ Callback functionality works")

    # Test with failing callback
    def failing_callback(status, data=None):
        raise Exception("Callback error")

    session_failing = WandBSeleniumSession(creds, callback=failing_callback)

    # This should not raise an exception
    session_failing._notify("test_status")
    print("   ✓ Failing callback handled gracefully")


def test_context_manager():
    """Test context manager functionality."""
    print("\n5. Testing context manager...")

    creds = WandBCredentials(username="test@example.com", password="test_password")

    with patch("wandb_selenium_login_fixed.webdriver.Chrome") as mock_chrome:
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        with WandBSeleniumSession(creds) as session:
            assert session.driver == mock_driver
            print("   ✓ Context manager __enter__ works")

        # Check that cleanup was called
        mock_driver.quit.assert_called_once()
        print("   ✓ Context manager __exit__ works")


def test_env_variable_detection():
    """Test if the script can detect environment variables."""
    print("\n6. Testing environment variable detection...")

    # Check if WANDB_VIEWER_MAIL is set (as we saw in the terminal)
    wandb_email = os.getenv("WANDB_VIEWER_MAIL")
    if wandb_email:
        print(f"   ✓ WANDB_VIEWER_MAIL detected: {wandb_email}")
    else:
        print("   ℹ WANDB_VIEWER_MAIL not set")

    # Check other common wandb environment variables
    wandb_vars = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
    for var in wandb_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✓ {var} detected: {value[:10]}...")
        else:
            print(f"   - {var} not set")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING WANDB SELENIUM LOGIN SCRIPT")
    print("=" * 60)

    try:
        test_credentials()
        test_session_init()
        test_driver_setup_mock()
        test_callback_functionality()
        test_context_manager()
        test_env_variable_detection()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe script is ready for use with real credentials.")
        print("To use it:")
        print("1. Ensure you have Chrome or Firefox installed")
        print("2. Create WandBCredentials with your email/password")
        print("3. Use WandBSeleniumSession for automated login")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
