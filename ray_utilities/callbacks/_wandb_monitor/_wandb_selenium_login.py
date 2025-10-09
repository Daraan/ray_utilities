"""
Selenium script for WandB login with threadable design.
This script handles automated login to wandb.ai and provides a foundation
for visiting other WandB websites and using the WandB API.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import wandb
from ray_utilities.callbacks._wandb_monitor._wandb_session_cache import WandBSessionCache

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WandBCredentials:
    """Container for WandB login credentials."""

    username: str
    password: str
    api_key: Optional[str] = None
    team_name: Optional[str] = None  # Optional team name to verify loggin


class WandBSeleniumSession:
    """
    Threadable WandB Selenium session for automated login and web interactions.

    This class provides a thread-safe way to manage WandB login via Selenium
    and subsequent API interactions.
    """

    def __init__(
        self,
        credentials: WandBCredentials | None,
        *,
        browser: str = "chrome",
        headless: bool = True,
        timeout: int = 30,
        callback: Optional[Callable[[str, Any], None]] = None,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the WandB Selenium session.

        Args:
            credentials: WandB login credentials
            browser: Browser to use ("chrome" or "firefox")
            headless: Whether to run browser in headless mode
            timeout: Default timeout for web elements
            callback: Optional callback function for status updates
            use_cache: Whether to use cookie/session caching (default: True)
            cache_dir: Directory for cache files (default: ~/.wandb_session_cache)
        """
        if credentials is None:
            credentials = WandBCredentials(
                username=os.getenv("WANDB_VIEWER_MAIL", ""),
                password=os.getenv("WANDB_VIEWER_PW", ""),
                api_key=os.getenv("WANDB_API_KEY", None),
                team_name=os.getenv("WANDB_VIEWER_TEAM_NAME", None),
            )
        if not credentials.username or not credentials.password:
            logger.warning("WandB credentials not provided or incomplete. For public workspaces this is fine.")
        self.credentials = credentials
        self.browser = browser.lower()
        self.headless = headless
        self.timeout = timeout
        self.callback = callback
        self.use_cache = use_cache

        # Initialize session cache
        self.session_cache = WandBSessionCache(cache_dir) if use_cache else None

        self.driver: Optional[webdriver.Remote] = None
        self.is_logged_in = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _notify(self, status: str, data: Any = None) -> None:
        """Send notification via callback if available."""
        if self.callback:
            try:
                self.callback(status, data)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Callback error: %s", e)

    def _setup_driver(self) -> webdriver.Remote:
        """Setup and return the appropriate WebDriver."""
        if self.browser == "chrome":
            options = ChromeOptions()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            self.driver = webdriver.Chrome(options=options)
            return self.driver

        if self.browser == "firefox":
            options = FirefoxOptions()
            if self.headless:
                options.add_argument("--headless")
            # Explicitly set Firefox binary path to avoid detection issues
            # Check for snap Firefox first, then fallback to traditional path
            firefox_paths = [
                "/snap/firefox/current/usr/lib/firefox/firefox",
                "/usr/bin/firefox",
                "/usr/lib/firefox/firefox",
            ]
            for path in firefox_paths:
                if os.path.exists(path):
                    options.binary_location = path
                    break
            self.driver = webdriver.Firefox(options=options)
            return self.driver

        raise ValueError(f"Unsupported browser: {self.browser}")

    def _wait_for_element(
        self,
        by: str,
        locator: str,
        timeout: Optional[int] = None,
    ) -> Any:
        """Wait for an element to be present and return it."""
        if not self.driver:
            raise RuntimeError("Driver not initialized")
        wait_time = timeout or self.timeout
        wait = WebDriverWait(self.driver, wait_time)
        return wait.until(EC.presence_of_element_located((by, locator)))

    def _wait_for_clickable(
        self,
        by: str,
        locator: str,
        timeout: Optional[int] = None,
    ) -> Any:
        """Wait for an element to be clickable and return it."""
        if not self.driver:
            raise RuntimeError("Driver not initialized")
        wait_time = timeout or self.timeout
        wait = WebDriverWait(self.driver, wait_time)
        return wait.until(EC.element_to_be_clickable((by, locator)))

    def _load_cached_cookies(self) -> bool:
        """
        Load and apply cached cookies if available.

        Returns:
            True if cached cookies were loaded and session is valid, False otherwise
        """
        if not self.use_cache or not self.session_cache:
            return False

        if not self.driver:
            return False

        try:
            # Load cached cookies
            cached_cookies = self.session_cache.load_cookies(
                self.credentials.username,
                self.browser,
                max_age_hours=24.0,  # Consider cookies valid for 24 hours
            )

            if not cached_cookies:
                return False

            # Navigate to WandB domain first to set cookies
            self.driver.get("https://wandb.ai")
            time.sleep(2)  # Wait for page load

            # Apply cached cookies
            cookies_applied = 0
            for cookie in cached_cookies:
                try:
                    # Remove problematic fields that selenium doesn't need
                    cookie_data = {
                        k: v
                        for k, v in cookie.items()
                        if k in ["name", "value", "domain", "path", "secure", "httpOnly"]
                    }

                    # Skip cookies that might cause issues
                    if cookie_data.get("domain") and not cookie_data["domain"].endswith("wandb.ai"):
                        continue

                    self.driver.add_cookie(cookie_data)
                    cookies_applied += 1
                except WebDriverException as e:
                    logger.debug("Failed to add cookie %s: %s", cookie.get("name", "unknown"), e)
                    continue

            logger.debug("Applied %d out of %d cached cookies", cookies_applied, len(cached_cookies))

            # Test session validity by trying to access the project - it might be public (treat like we are logged in)
            # if its not public it will show "404" and "stumbled on an empty page" if not authenticated
            test_run_url = "https://wandb.ai/daraan/dev-workspace/"
            self.driver.get(test_run_url)
            time.sleep(7)  # Wait for page load and potential redirects

            current_url = self.driver.current_url
            page_source = self.driver.page_source.lower()

            logger.debug("Current URL after applying cookies: %s", current_url)

            # Check for authentication indicators
            # If we're authenticated, we should either see the run page or be redirected to login
            # If we get "stumbled on an empty page", that means we're not authenticated
            is_authenticated = (
                "wandb.ai" in current_url
                and "auth0.com" not in current_url
                and "stumbled on an empty page" not in page_source  # This is the key 404 indicator
                and ("login" not in current_url or "runs/" in current_url)  # Either on run page or login redirect
            )

            if is_authenticated:
                logger.info("Successfully restored session using cached cookies")
                self.is_logged_in = True
                self._notify("cached_login_success")

                # Load cached session data (API key, etc.)
                if not self.credentials.api_key:
                    session_data = self.session_cache.load_session_data(self.credentials.username)
                    if session_data and session_data.get("api_key"):
                        self.credentials.api_key = session_data["api_key"]
                        logger.debug("Loaded cached API key")

                return True

            logger.debug("Cached session validation failed - URL: %s", current_url)
            if "stumbled on an empty page" in page_source:
                logger.debug("Got 'stumbled on an empty page' - session is definitely invalid")
            elif "login" in current_url:
                logger.debug("Redirected to login page - session expired")
            elif "404" in page_source:
                logger.debug("Got 404 page - session may be invalid")
            else:
                logger.debug("Session validation failed for unknown reason")
            return False

        except Exception as e:
            logger.debug("Failed to load cached cookies: %s", e)
            return False

    def _save_current_cookies(self) -> None:
        """Save current browser cookies to cache."""
        if not self.use_cache or not self.session_cache or not self.driver:
            return

        try:
            # Get all cookies from current session
            cookies = self.driver.get_cookies()

            # Save cookies to cache
            if cookies:
                self.session_cache.save_cookies(self.credentials.username, self.browser, cookies)
                logger.debug("Saved %d cookies to cache", len(cookies))

            # Save session data
            self.session_cache.save_session_data(self.credentials.username, self.credentials.api_key)

        except Exception as e:
            logger.debug("Failed to save cookies to cache: %s", e)

    def login(self) -> bool:
        """
        Perform WandB login via Selenium.

        Returns:
            True if login successful, False otherwise
        """
        if not self.driver:
            self.driver = self._setup_driver()

        try:
            self._notify("starting_login")

            # First, try to use cached cookies if available
            if self._load_cached_cookies():
                return True

            # If cached login failed, proceed with manual login
            logger.info("Cached login failed or not available, performing manual login")

            # Navigate to WandB login page (new URL that redirects to Auth0)
            login_url = "https://app.wandb.ai/login?_gl=1*1njlh40*_ga*MjAzMDY0NTMxOC4xNjg2NzMzODEw*_ga_JH1SJHJQXJ*MTY5MDk5MDgxNS4xMjEuMS4xNjkwOTkxMDExLjYwLjAuMA.."
            self.driver.get(login_url)
            logger.info("Navigated to WandB login page")

            # Wait for redirect to Auth0 or check if we're already on Auth0
            time.sleep(2)  # Give time for redirect

            # Check if we're on Auth0 domain
            current_url = self.driver.current_url
            logger.info("Current URL after navigation: %s", current_url)

            # Check if we're already logged in (redirected to wandb.ai instead of auth0)
            if (
                "wandb.ai" in current_url
                and "auth0.com" not in current_url
                and "login" not in current_url
                and ("home" in current_url or "dashboard" in current_url)
            ):
                logger.info("Already logged in, no manual login needed")
                self.is_logged_in = True
                self._notify("login_success")

                # Save cookies and session data for future use
                self._save_current_cookies()
                return True

            # Look for email field with Auth0-specific selectors
            email_selectors = [
                "input[id='1-email']",  # Auth0 specific ID (valid CSS syntax)
                "input.auth0-lock-input[type='email']",  # Auth0 specific class + type
                "input[name='email'].auth0-lock-input",  # Auth0 class + name
                "input[name='email']",
                "input[type='email']",
                "input[placeholder*='email' i]",
                "input[data-testid='email']",
                "input#email",
                "input[autocomplete='email']",
            ]

            username_field = None
            for selector in email_selectors:
                try:
                    username_field = self._wait_for_element(By.CSS_SELECTOR, selector, timeout=5)
                    logger.info("Found email field with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not username_field:
                logger.error("Could not find email input field")
                return False

            # Scroll to element and make sure it's visible
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", username_field
            )
            time.sleep(1)  # Wait for scroll to complete
            breakpoint()

            username_field.clear()
            username_field.send_keys(self.credentials.username)
            logger.info("Entered username: %s", self.credentials.username[0] + "***" + self.credentials.username[-4:])

            # Look for password field with Auth0-specific selectors
            password_selectors = [
                "input[id='1-password']",  # Auth0 specific ID (valid CSS syntax)
                "input.auth0-lock-input[type='password']",  # Auth0 specific class + type
                "input[name='password'].auth0-lock-input",  # Auth0 class + name
                "input[name='password']",
                "input[type='password']",
                "input[data-testid='password']",
                "input#password",
                "input[autocomplete='current-password']",
            ]

            password_field = None
            for selector in password_selectors:
                try:
                    password_field = self._wait_for_element(By.CSS_SELECTOR, selector, timeout=5)
                    logger.info("Found password field with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not password_field:
                logger.error("Could not find password input field")
                return False

            # Scroll to element and make sure it's visible
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", password_field
            )
            time.sleep(1)  # Wait for scroll to complete

            password_field.clear()
            password_field.send_keys(self.credentials.password)
            logger.info("Entered password")

            # Look for login button with Auth0-specific selectors
            login_button_selectors = [
                "button[type='submit']",  # Most common
                "button.auth0-lock-submit",  # Auth0 specific submit button
                ".auth0-lock-submit",  # Auth0 submit class
                "input[type='submit']",
                "button[data-testid='submit']",
                "button:contains('Log in')",
                "button:contains('Sign in')",
                "button:contains('Continue')",
                "button.auth0-lock-submit-button",
            ]

            login_button = None
            for selector in login_button_selectors:
                try:
                    login_button = self._wait_for_clickable(By.CSS_SELECTOR, selector, timeout=5)
                    logger.info("Found login button with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not login_button:
                logger.error("Could not find login button")
                return False

            # Scroll to button and make sure it's visible
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", login_button
            )
            time.sleep(2)  # Wait for scroll to complete

            login_button.click()
            logger.info("Clicked login button")

            # Wait for successful login - check for redirect back to WandB
            try:
                # Wait for either dashboard, profile, or any wandb.ai domain (not auth0)
                WebDriverWait(self.driver, self.timeout).until(
                    lambda driver: (
                        "wandb.ai" in driver.current_url
                        and "auth0.com" not in driver.current_url
                        and (
                            "home" in driver.current_url
                            or "profile" in driver.current_url
                            or "dashboard" in driver.current_url
                            or driver.find_elements(
                                By.CSS_SELECTOR, "[data-test*='dashboard'], .dashboard, [class*='dashboard']"
                            )
                        )
                    )
                )

                self.is_logged_in = True
                logger.info("Successfully logged in to WandB")
                logger.info("Final URL: %s", self.driver.current_url)
                self._notify("login_success")

                # Save cookies and session data for future use
                self._save_current_cookies()

            except TimeoutException:
                # Check for error messages on Auth0 or WandB
                error_selectors = [
                    ".error",
                    ".alert-danger",
                    "[class*='error']",
                    "[class*='invalid']",
                    ".auth0-lock-error",
                    ".auth0-global-message-error",
                    "[data-testid='error']",
                    ".notification-error",
                ]

                error_text = "Unknown error"
                for selector in error_selectors:
                    error_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if error_elements and error_elements[0].text.strip():
                        error_text = error_elements[0].text
                        break

                logger.error("Login failed: %s", error_text)
                logger.error("Current URL: %s", self.driver.current_url)
                self._notify("login_failed", error_text)
                return False

        except (TimeoutException, WebDriverException) as e:
            logger.error("Login error: %s", e)
            logger.error("Current URL: %s", self.driver.current_url if self.driver else "Unknown")
            self._notify("login_error", str(e))
            return False

        return True

    def visit_wandb_page(self, url: str) -> bool:
        """
        Visit a WandB page and wait for it to load.

        Args:
            url: WandB URL to visit

        Returns:
            True if page loaded successfully, False otherwise
        """
        if not self.is_logged_in:
            logger.warning("Not logged in, cannot visit page")
            return False

        if not self.driver:
            raise RuntimeError("Driver not initialized")

        try:
            self.driver.get(url)
            # Wait for page to load (basic check)
            WebDriverWait(self.driver, self.timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )

            # Check if we ended up on a 404 page (private page without authentication)
            page_source = self.driver.page_source.lower()
            current_url = self.driver.current_url

            if "stumbled on an empty page" in page_source or ("404" in page_source and "wandb.ai" in current_url):
                logger.error("Accessed private page without authentication: %s", url)
                logger.error("Got 404 error - insufficient permissions or invalid session")
                self._notify(
                    "page_visit_failed", {"url": url, "error": "404 - Private page or authentication required"}
                )
                return False

            logger.info("Successfully visited: %s", url)
            self._notify("page_visited", url)

        except (TimeoutException, WebDriverException) as e:
            logger.error("Failed to visit %s: %s", url, e)
            self._notify("page_visit_failed", {"url": url, "error": str(e)})
            return False

        return True

    def initialize_wandb_api(self) -> bool:
        """
        Initialize WandB API using extracted or provided API key.

        Returns:
            True if API initialized successfully, False otherwise
        """
        if not self.credentials.api_key:
            logger.error("No API key available for WandB API initialization")
            return False

        try:
            wandb.login(key=self.credentials.api_key)
            logger.info("Successfully initialized WandB API")
            self._notify("api_initialized")

        except (wandb.Error, ValueError, ConnectionError) as e:
            logger.error("Failed to initialize WandB API: %s", e)
            self._notify("api_init_failed", str(e))
            return False

        return True

    def run_threaded(self) -> threading.Thread:
        """
        Start the login process in a separate thread.

        Returns:
            Thread object for the login process
        """
        if self.thread and self.thread.is_alive():
            logger.warning("Thread already running")
            return self.thread

        self.thread = threading.Thread(target=self._threaded_run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started WandB login thread")
        return self.thread

    def _threaded_run(self) -> None:
        """Internal method for threaded execution."""
        try:
            with self._lock:
                self._setup_driver()
                self._notify("driver_initialized")

            # Perform login
            success = self.login()

            if success and self.credentials.api_key:
                self.initialize_wandb_api()

            # Keep the session alive until stop is requested
            while not self._stop_event.wait(1):
                if not self.driver:
                    break

                # Basic health check
                try:
                    _ = self.driver.current_url  # Health check
                except WebDriverException:
                    logger.warning("WebDriver session lost")
                    break

        except (WebDriverException, TimeoutException, ValueError) as e:
            logger.error("Thread execution error: %s", e)
            self._notify("thread_error", str(e))

        finally:
            self.cleanup()

    def stop(self) -> None:
        """Stop the threaded session."""
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
            logger.info("Stopped WandB session thread")

    def clear_cache(self) -> bool:
        """
        Clear cached session data for the current user.

        Returns:
            True if cache cleared successfully, False otherwise
        """
        if not self.use_cache or not self.session_cache:
            logger.warning("Cache not enabled, cannot clear cache")
            return False

        return self.session_cache.clear_cache(self.credentials.username)

    def is_cache_valid(self, max_age_hours: float = 24.0) -> bool:
        """
        Check if cached session data exists and is still valid.

        Args:
            max_age_hours: Maximum age of cached data in hours

        Returns:
            True if valid cached data exists, False otherwise
        """
        if not self.use_cache or not self.session_cache:
            return False

        cached_cookies = self.session_cache.load_cookies(
            self.credentials.username, self.browser, max_age_hours=max_age_hours
        )

        return cached_cookies is not None

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            if self.driver:
                try:
                    self.driver.quit()
                except (WebDriverException, ConnectionError) as e:
                    logger.warning("Error during driver cleanup: %s", e)
                finally:
                    self.driver = None
                    self.is_logged_in = False

        self._notify("cleanup_complete")
        logger.info("Cleaned up WandB session")

    def __enter__(self):
        """Context manager entry."""
        self._setup_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:  # noqa: BLE001
            pass


def example_usage():
    """Example usage of the WandB Selenium session."""

    def status_callback(status: str, data: Any = None):
        """Example callback for status updates."""
        print(f"Status: {status} - {data}")

    # Setup credentials
    credentials = WandBCredentials(username=os.environ["WANDB_VIEWER_MAIL"], password=os.environ["WANDB_VIEWER_PW"])

    # Example 1: Basic usage with context manager
    print("Example 1: Context manager usage")
    try:
        with WandBSeleniumSession(credentials, callback=status_callback) as session:
            if session.login():
                print("Login successful!")

                # Visit some WandB pages
                session.visit_wandb_page("https://wandb.ai/home")
                session.visit_wandb_page("https://wandb.ai/profile")

                # Initialize API if key is available
                if session.credentials.api_key:
                    session.initialize_wandb_api()
                    # Now you can use the wandb API
                    # wandb.init(project="test-project")
                    # wandb.log({"metric": 1})
                    # wandb.finish()

                time.sleep(5)  # Do some work
            else:
                print("Login failed!")

    except (WebDriverException, TimeoutException) as e:
        print(f"Error: {e}")

    # Example 2: Threaded usage
    print("\nExample 2: Threaded usage")
    session = WandBSeleniumSession(credentials, callback=status_callback)

    try:
        # Start in thread
        session.run_threaded()

        # Do other work while login happens in background
        print("Doing other work while login happens...")
        time.sleep(10)

        # Check if login was successful
        if session.is_logged_in:
            print("Background login successful!")

            # Use the session
            session.visit_wandb_page("https://wandb.ai/home")

            if session.credentials.api_key:
                session.initialize_wandb_api()

        # Wait a bit more
        time.sleep(5)

    finally:
        session.stop()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run example (you'll need to provide real credentials)
    example_usage()
