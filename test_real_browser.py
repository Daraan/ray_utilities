#!/usr/bin/env python3
"""
Real browser test for WandB Selenium script.
This test launches an actual browser to verify selenium works.
"""

import logging
import sys
from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_real_browser():
    """Test that we can actually launch a browser."""
    print("Testing real browser launch...")

    # Create dummy credentials
    creds = WandBCredentials(username="test@example.com", password="test_password")

    try:
        # Test with headless Firefox (since Chrome is not available)
        session = WandBSeleniumSession(creds, browser="firefox", headless=True)

        with session:
            print("✓ Successfully launched headless Firefox")

            # Just navigate to a simple page to test driver works
            if session.driver:
                session.driver.get("https://httpbin.org/get")
                print("✓ Successfully navigated to test URL")

                # Get page title to verify it loaded
                title = session.driver.title
                print(f"✓ Page title: {title}")

                # Test basic interaction
                current_url = session.driver.current_url
                print(f"✓ Current URL: {current_url}")
            else:
                print("✗ Driver not initialized")
                return False

        print("✓ Browser cleanup successful")
        return True

    except Exception as e:
        print(f"✗ Browser test failed: {e}")
        return False


def main():
    """Run browser test."""
    print("=" * 60)
    print("REAL BROWSER TEST FOR WANDB SELENIUM SCRIPT")
    print("=" * 60)

    try:
        success = test_real_browser()

        if success:
            print("\n" + "=" * 60)
            print("✓ BROWSER TEST PASSED!")
            print("=" * 60)
            print("\nSelenium is working correctly.")
            print("You can now use the script with real WandB credentials.")
        else:
            print("\n" + "=" * 60)
            print("✗ BROWSER TEST FAILED!")
            print("=" * 60)
            print("\nCheck that you have Firefox installed and accessible.")
            print("You may need to install geckodriver separately.")
            print("For Chrome support, install Chrome and chromedriver.")
            return False

    except KeyboardInterrupt:
        print("\n✗ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
