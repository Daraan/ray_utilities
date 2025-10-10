#!/usr/bin/env python3
"""
Demonstration script for WandB Selenium Login.
This shows how to use the script for actual WandB login and API usage.
"""

import logging
import os
import sys
import time
from typing import Optional

from wandb_selenium_login_fixed import WandBCredentials, WandBSeleniumSession

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def get_credentials_from_env() -> Optional[WandBCredentials]:
    """Get WandB credentials from environment variables."""
    email = os.getenv("WANDB_VIEWER_MAIL") or os.getenv("WANDB_VIEW_EMAIL")
    password = os.getenv("WANDB_VIEWER_PW")
    api_key = os.getenv("WANDB_API_KEY")

    if email and password:
        return WandBCredentials(username=email, password=password, api_key=api_key)
    return None


def demo_basic_login():
    """Demonstrate basic login functionality."""
    print("=" * 60)
    print("DEMO: Basic WandB Login")
    print("=" * 60)

    credentials = get_credentials_from_env()
    if not credentials:
        print("⚠️  No credentials found in environment variables.")
        print("Set WANDB_VIEWER_MAIL and WANDB_VIEWER_PW environment variables to test login.")
        print("Example:")
        print("  export WANDB_VIEWER_MAIL='your_email@example.com'")
        print("  export WANDB_VIEWER_PW='your_password'")
        return False

    def status_callback(status: str, data=None):
        """Callback to show login progress."""
        status_messages = {
            "starting_login": "🔐 Starting login process...",
            "login_success": "✅ Login successful!",
            "login_failed": f"❌ Login failed: {data}",
            "login_error": f"💥 Login error: {data}",
            "api_key_extracted": "🔑 API key extracted successfully",
            "api_initialized": "🚀 WandB API initialized",
            "cleanup_complete": "🧹 Session cleaned up",
        }
        message = status_messages.get(status, f"ℹ️  {status}: {data}")
        print(f"  {message}")

    try:
        print(f"📧 Using email: {credentials.username}")
        print("🌐 Starting browser session...")

        with WandBSeleniumSession(credentials, browser="firefox", headless=True, callback=status_callback) as session:
            # Attempt login
            if session.login():
                print("\n🎉 Login successful!")

                # Visit some WandB pages
                print("\n📱 Testing page navigation...")
                pages = ["https://wandb.ai/home", "https://wandb.ai/profile", "https://wandb.ai/settings"]

                for page in pages:
                    if session.visit_wandb_page(page):
                        print(f"  ✅ Successfully visited: {page}")
                    else:
                        print(f"  ❌ Failed to visit: {page}")

                # Test API initialization if key is available
                if session.credentials.api_key:
                    print("\n🔧 Testing WandB API...")
                    if session.initialize_wandb_api():
                        print("  ✅ WandB API ready for use")
                        print(f"  🔑 API key: {session.credentials.api_key[:10]}...")
                    else:
                        print("  ❌ Failed to initialize WandB API")
                else:
                    print("\n⚠️  No API key available")

                return True
            else:
                print("\n❌ Login failed")
                return False

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        return False


def demo_threaded_login():
    """Demonstrate threaded login functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Threaded WandB Login")
    print("=" * 60)

    credentials = get_credentials_from_env()
    if not credentials:
        print("⚠️  Skipping threaded demo - no credentials available")
        return True

    def status_callback(status: str, data=None):
        """Non-blocking callback for threaded demo."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] 🧵 {status}: {data}")

    session = WandBSeleniumSession(credentials, browser="firefox", headless=True, callback=status_callback)

    try:
        print("🚀 Starting login in background thread...")
        thread = session.run_threaded()

        # Simulate doing other work while login happens
        for i in range(10):
            print(f"⚙️  Doing other work... step {i + 1}/10")
            time.sleep(1)

            # Check if login completed
            if session.is_logged_in:
                print("🎉 Background login completed!")
                break

        # Wait for thread to complete
        print("⏳ Waiting for login thread to complete...")
        thread.join(timeout=30)

        if session.is_logged_in:
            print("✅ Threaded login successful!")
            print(f"🔑 API key available: {'Yes' if session.credentials.api_key else 'No'}")
            return True
        else:
            print("❌ Threaded login failed")
            return False

    finally:
        session.stop()
        print("🏁 Threaded demo completed")


def demo_api_usage():
    """Demonstrate actual WandB API usage after login."""
    print("\n" + "=" * 60)
    print("DEMO: WandB API Usage")
    print("=" * 60)

    credentials = get_credentials_from_env()
    if not credentials:
        print("⚠️  Skipping API demo - no credentials available")
        return True

    try:
        with WandBSeleniumSession(credentials, browser="firefox", headless=True) as session:
            if session.login() and session.initialize_wandb_api():
                print("✅ Ready to use WandB API")

                # Import wandb here to avoid issues if API not initialized
                import wandb

                # Initialize a test run
                print("🧪 Creating test experiment...")
                run = wandb.init(
                    project="selenium-login-test",
                    name="demo-run",
                    tags=["selenium", "demo"],
                    notes="Test run created via selenium login",
                )

                # Log some dummy metrics
                print("📊 Logging test metrics...")
                for step in range(5):
                    wandb.log({"accuracy": 0.8 + (step * 0.02), "loss": 1.0 - (step * 0.1), "step": step})
                    time.sleep(0.5)

                print("💾 Finishing experiment...")
                wandb.finish()

                print("✅ WandB API demo completed successfully!")
                return True
            else:
                print("❌ Failed to initialize WandB API")
                return False

    except Exception as e:
        print(f"💥 API demo failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("🎭 WandB Selenium Login - Full Demonstration")
    print("=" * 80)

    results = []

    # Demo 1: Basic login
    try:
        results.append(("Basic Login", demo_basic_login()))
    except Exception as e:
        print(f"💥 Basic login demo failed: {e}")
        results.append(("Basic Login", False))

    # Demo 2: Threaded login
    try:
        results.append(("Threaded Login", demo_threaded_login()))
    except Exception as e:
        print(f"💥 Threaded login demo failed: {e}")
        results.append(("Threaded Login", False))

    # Demo 3: API usage
    try:
        results.append(("API Usage", demo_api_usage()))
    except Exception as e:
        print(f"💥 API usage demo failed: {e}")
        results.append(("API Usage", False))

    # Summary
    print("\n" + "=" * 80)
    print("📋 DEMO RESULTS SUMMARY")
    print("=" * 80)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name:20} {status}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nThe WandB Selenium login script is ready for production use.")
    else:
        print("\n⚠️  Some demos failed. Check the output above for details.")

    print("\n💡 Usage Tips:")
    print("  - Set WANDB_VIEWER_MAIL and WANDB_VIEWER_PW for full testing")
    print("  - Use headless=False to see the browser in action")
    print("  - Implement custom callbacks for progress tracking")
    print("  - Use threaded mode for non-blocking login")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
