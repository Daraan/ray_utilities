#!/usr/bin/env python3
"""
Quick test of the updated WandB Auth0 login flow.
This script tests the updated selectors without requiring credentials.
"""

import logging
import os
import sys
import time
from pathlib import Path

import dotenv

from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

dotenv.load_dotenv(Path("~/.comet_api_key.env").expanduser())


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def test_auth0_navigation():
    """Test navigation to Auth0 and element detection."""
    print("🧪 Testing Auth0 Navigation and Element Detection")
    print("=" * 60)

    # Create dummy credentials for testing
    credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])

    def status_callback(status: str, data=None):
        """Callback to show progress."""
        print(f"  📢 {status}: {data}")

    try:
        with WandBSeleniumSession(
            credentials,
            browser="firefox",
            headless=True,  # Set to True to avoid display issues
            timeout=15,
            callback=status_callback,
        ) as session:
            print("🌐 Navigating to WandB login...")

            # Navigate to the login URL
            login_url = "https://app.wandb.ai/login?_gl=1*1njlh40*_ga*MjAzMDY0NTMxOC4xNjg2NzMzODEw*_ga_JH1SJHJQXJ*MTY5MDk5MDgxNS4xMjEuMS4xNjkwOTkxMDExLjYwLjAuMA.."
            session.driver.get(login_url)

            # Wait for redirect
            time.sleep(3)

            current_url = session.driver.current_url
            print(f"📍 Current URL: {current_url}")

            # Check if we're on Auth0
            if "auth0.com" in current_url:
                print("✅ Successfully redirected to Auth0")
            else:
                print(f"⚠️  Expected Auth0 redirect, but got: {current_url}")

            # Test element detection
            print("\n🔍 Testing element detection...")

            # Test email field detection
            email_selectors = [
                "input[id='1-email']",
                "input.auth0-lock-input[type='email']",
                "input[name='email'].auth0-lock-input",
                "input[name='email']",
                "input[type='email']",
            ]

            email_found = False
            for selector in email_selectors:
                try:
                    elements = session.driver.find_elements("css selector", selector)
                    if elements:
                        print(f"  ✅ Email field found with: {selector}")
                        print(
                            f"     Element: {elements[0].tag_name} with id='{elements[0].get_attribute('id')}' class='{elements[0].get_attribute('class')}'"
                        )
                        email_found = True
                        break
                except Exception as e:
                    continue

            if not email_found:
                print("  ❌ No email field found")

            # Test password field detection
            password_selectors = [
                "input[id='1-password']",
                "input.auth0-lock-input[type='password']",
                "input[name='password'].auth0-lock-input",
                "input[name='password']",
                "input[type='password']",
            ]

            password_found = False
            for selector in password_selectors:
                try:
                    elements = session.driver.find_elements("css selector", selector)
                    if elements:
                        print(f"  ✅ Password field found with: {selector}")
                        print(
                            f"     Element: {elements[0].tag_name} with id='{elements[0].get_attribute('id')}' class='{elements[0].get_attribute('class')}'"
                        )
                        password_found = True
                        break
                except Exception as e:
                    continue

            if not password_found:
                print("  ❌ No password field found")

            # Test submit button detection
            button_selectors = [
                "button[type='submit']",
                "button.auth0-lock-submit",
                ".auth0-lock-submit",
                "input[type='submit']",
            ]

            button_found = False
            for selector in button_selectors:
                try:
                    elements = session.driver.find_elements("css selector", selector)
                    if elements:
                        print(f"  ✅ Submit button found with: {selector}")
                        print(
                            f"     Element: {elements[0].tag_name} with class='{elements[0].get_attribute('class')}' text='{elements[0].text}'"
                        )
                        button_found = True
                        break
                except Exception as e:
                    continue

            if not button_found:
                print("  ❌ No submit button found")

            # Check page source for debugging
            print("\n🔍 Page analysis...")
            page_source = session.driver.page_source

            if "auth0-lock-input" in page_source:
                print("  ✅ Auth0 lock elements detected in page source")
            else:
                print("  ❌ No Auth0 lock elements found in page source")

            if "password" in page_source.lower():
                print("  ✅ Password-related content found")
            else:
                print("  ❌ No password-related content found")

            # Keep browser open for manual inspection
            print(f"\n⏸️  Browser kept open for 10 seconds for inspection...")
            print(f"   Current URL: {session.driver.current_url}")
            time.sleep(10)

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the Auth0 test."""
    print("🎭 WandB Auth0 Login Flow Test")
    print("=" * 80)

    success = test_auth0_navigation()

    if success:
        print("\n🎉 Auth0 navigation test completed!")
        print("Check the browser output above to see if elements were detected correctly.")
    else:
        print("\n💥 Test failed. Check the error output above.")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
