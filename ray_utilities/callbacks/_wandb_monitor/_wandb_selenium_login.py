"""
Selenium script for WandB login with threadable design.
This script handles automated login to wandb.ai and provides a foundation
for visiting other WandB websites and using the WandB API.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
import weakref
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


# Global registry to track active WebDriver sessions for cleanup
_active_sessions: weakref.WeakSet[WandBSeleniumSession] = weakref.WeakSet()
_cleanup_registered = False


def _emergency_cleanup():
    """Emergency cleanup function called at exit to kill orphaned driver processes."""
    logger.debug("Emergency cleanup: checking for orphaned WebDriver sessions")
    for session in list(_active_sessions):
        try:
            if session.driver:
                logger.warning("Emergency cleanup: force killing WebDriver for session %s", id(session))
                session._force_kill_driver()
        except Exception as e:  # noqa: BLE001, PERF203
            logger.debug("Error during emergency cleanup: %s", e)


def _register_exit_cleanup():
    """Register the emergency cleanup function to run at exit."""
    global _cleanup_registered  # noqa: PLW0603
    if not _cleanup_registered:
        atexit.register(_emergency_cleanup)
        _cleanup_registered = True
        logger.debug("Registered emergency cleanup handler")


def _get_process_children(pid: int) -> list[int]:
    """Get child process IDs for a given parent PID."""
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [int(child_pid) for child_pid in result.stdout.strip().split("\n") if child_pid]
    except (subprocess.SubprocessError, ValueError, OSError):
        return []
    else:
        return []


def _kill_process_tree(pid: int) -> bool:
    """Kill a process and all its children."""
    try:
        # First check if process exists
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
        except (OSError, ProcessLookupError):
            logger.debug("Process %s does not exist", pid)
            return False  # Process doesn't exist

        # Get all child processes first
        children = _get_process_children(pid)

        # Kill children first
        for child_pid in children:
            try:
                os.kill(child_pid, signal.SIGTERM)
                logger.debug("Killed child process %s", child_pid)
            except (OSError, ProcessLookupError):  # noqa: PERF203
                pass  # Process already dead

        # Kill main process
        try:
            os.kill(pid, signal.SIGTERM)
            logger.debug("Sent SIGTERM to process %s", pid)

            # Wait a bit and then use SIGKILL if still alive
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # Check if still alive
                os.kill(pid, signal.SIGKILL)
                logger.debug("Sent SIGKILL to process %s", pid)
            except (OSError, ProcessLookupError):
                pass  # Process already dead from SIGTERM

        except (OSError, ProcessLookupError):
            pass  # Process already dead
    except Exception as e:  # noqa: BLE001
        logger.warning("Error killing process tree for PID %s: %s", pid, e)
        return False
    else:
        return True


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
        credentials: WandBCredentials | None = None,
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
        self._driver_pid: Optional[int] = None  # Track driver process ID for force cleanup
        self.is_logged_in = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Track run page tabs: {run_url: tab_handle}
        self._run_tabs: dict[str, str] = {}

        # Register this session for global cleanup tracking
        _active_sessions.add(self)
        _register_exit_cleanup()

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
            self._capture_driver_pid()
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
            self._capture_driver_pid()
            return self.driver

        raise ValueError(f"Unsupported browser: {self.browser}")

    def _capture_driver_pid(self) -> None:
        """Capture the process ID of the driver for later cleanup."""
        if not self.driver:
            return

        try:
            # Try to get the service/process from the driver
            if hasattr(self.driver, "service") and hasattr(self.driver.service, "process"):
                if self.driver.service.process:
                    self._driver_pid = self.driver.service.process.pid
                    logger.debug("Captured driver PID: %s", self._driver_pid)
                    return

            # Fallback: try to find the browser process by name
            browser_process_name = "chrome" if self.browser == "chrome" else "firefox"
            try:
                result = subprocess.run(
                    ["pgrep", "-f", browser_process_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Get the most recent process (last in the list)
                    pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid]
                    if pids:
                        self._driver_pid = pids[-1]  # Use the most recent one
                        logger.debug("Found %s process PID via pgrep: %s", browser_process_name, self._driver_pid)
            except (subprocess.SubprocessError, ValueError, OSError) as e:
                logger.debug("Failed to find driver PID via pgrep: %s", e)

        except Exception as e:  # ruff: noqa: BLE001
            logger.debug("Failed to capture driver PID: %s", e)

    def _force_kill_driver(self) -> bool:
        """Force kill the WebDriver process and any child processes."""
        if not self._driver_pid:
            logger.debug("No driver PID to kill")
            return False

        logger.warning("Force killing WebDriver process %s", self._driver_pid)
        try:
            success = _kill_process_tree(self._driver_pid)
            if success:
                logger.debug("Successfully force killed driver process %s", self._driver_pid)
                self._driver_pid = None
        except Exception as e:  # ruff: noqa: BLE001
            logger.warning("Error force killing driver: %s", e)
            return False
        return success

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

    def _check_team_present(self) -> bool | None:
        """
        Check if the expected team name profile is visible in the top right corner.

        Returns:
            True if expected team is found, False if logged out or wrong team, None if no team specified
        """
        if not self.driver:
            return False

        try:
            # Get expected team name from environment variable or credentials
            expected_team_name = self.credentials.team_name or os.getenv("WANDB_VIEWER_TEAM_NAME", None)
            if not expected_team_name:
                logger.debug("No team name specified for verification, skipping check")
                return None  # No specific team to check, assume success

            # First check if we're logged out by looking for logged-out indicators
            logged_out_indicators = [
                ".logged-out-menu",
                "[data-test='login-button']",
                "[data-test='signup']",
                "button:contains('Log in')",
                "button:contains('Sign up')",
            ]

            for selector in logged_out_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.debug("Found logged-out indicator: %s", selector)
                        return False
                except Exception:
                    continue

            # Look for the profile selector element that contains the expected team name
            profile_selectors = [
                "[data-test='nav-profile-account-selector']",
                ".nav-profile-account-selector",
                "[aria-label*='Toggle profile']",
                "[class*='profile']",
            ]

            for selector in profile_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if expected_team_name in element.text:
                            logger.debug("Found expected team name '%s' in profile indicator", expected_team_name)
                            return True
                except Exception:
                    continue

            # Fallback: check page source for expected team name
            page_source = self.driver.page_source
            if expected_team_name in page_source:
                logger.debug("Found expected team name '%s' in page source", expected_team_name)
                return True

            logger.debug("Expected team name '%s' not found in profile", expected_team_name)
            return False

        except Exception as e:
            logger.debug("Error checking for team profile: %s", e)
            return False

    def _check_run_page_loaded(self, url: str) -> bool:
        """
        Check if a WandB run page has fully loaded by looking for 'Run details' button.

        Args:
            url: The URL being checked

        Returns:
            True if run page is fully loaded or not a run page, False if run page but not loaded
        """
        if not self.driver:
            return False

        try:
            # Check if this is a run page URL
            if "/runs/" not in url:
                # Not a run page, skip this check
                return True

            # Look for "Run details" button to confirm run page is loaded
            run_details_selectors = [
                # XPath selectors that can search by text content
                "//a[contains(text(), 'Run details')]",
                "//button[contains(text(), 'Run details')]",
                "//a[@data-component='Button' and contains(text(), 'Run details')]",
                "//a[contains(@href, '/overview') and contains(text(), 'Run details')]",
                # CSS selectors for button components (without text matching)
                "a[data-component='Button'][href*='/overview']",
                "a[href*='/overview']",
            ]

            for selector in run_details_selectors:
                try:
                    if selector.startswith("//"):
                        # XPath selector
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        # CSS selector
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    if elements:
                        logger.debug("Found 'Run details' button - run page loaded successfully")
                        return True
                except Exception:
                    continue

            # Fallback: check page source for "Run details" text
            page_source = self.driver.page_source
            if "Run details" in page_source:
                logger.debug("Found 'Run details' text in page source")
                return True

            logger.debug("'Run details' button not found - run page may not be fully loaded")
            return False

        except Exception as e:
            logger.debug("Error checking if run page loaded: %s", e)
            return False

    def _load_cached_cookies(self) -> bool:
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

            # Check basic authentication indicators
            basic_auth_check = (
                "wandb.ai" in current_url
                and "auth0.com" not in current_url
                and "stumbled on an empty page" not in page_source
                and ("login" not in current_url or "runs/" in current_url)
            )

            # Check for Team profile to verify specific user authentication
            team_profile_check = self._check_team_present() if basic_auth_check else False

            is_authenticated = basic_auth_check and (team_profile_check is not False)

            if is_authenticated:
                expected_team_name = self.credentials.team_name or os.getenv("WANDB_VIEWER_TEAM_NAME", None)
                logger.info(
                    "Successfully restored session using cached cookies with %s profile verified",
                    expected_team_name,
                )
                self.is_logged_in = True
                self._notify("cached_login_success")

                # Load cached session data (API key, etc.)
                if not self.credentials.api_key:
                    session_data = self.session_cache.load_session_data(self.credentials.username)
                    if session_data and session_data.get("api_key"):
                        self.credentials.api_key = session_data["api_key"]
                        logger.debug("Loaded cached API key")

                return True

            if basic_auth_check and not team_profile_check:
                expected_team_name = self.credentials.team_name or os.getenv("WANDB_VIEWER_TEAM_NAME", None)
                logger.warning(
                    "Basic authentication passed but %s profile not found - cached session may be for different user",
                    expected_team_name,
                )

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

            # Special verification for run pages - wait for content to load
            if "/runs/" in url:
                logger.debug("Run page detected, waiting for run content to load...")

                # Wait up to 10 seconds for run details to appear
                run_loaded = False
                for _ in range(10):
                    if self._check_run_page_loaded(url):
                        run_loaded = True
                        break
                    time.sleep(1)

                if not run_loaded:
                    logger.warning("Run page may not have fully loaded - 'Run details' button not found: %s", url)
                    # Don't fail the visit, just warn - some run pages might have different layouts
                else:
                    logger.info("Run page fully loaded with 'Run details' button verified: %s", url)

            logger.debug("Successfully visited: %s", url)
            self._notify("page_visited", url)

        except (TimeoutException, WebDriverException) as e:
            logger.error("Failed to visit %s: %s", url, e)
            self._notify("page_visit_failed", {"url": url, "error": str(e)})
            return False

        return True

    def open_new_tab(self, *, switch_to: bool = True) -> str:
        """
        Open a new browser tab.

        Args:
            switch_to: Whether to switch to the new tab immediately

        Returns:
            Handle of the new tab

        Raises:
            RuntimeError: If driver is not initialized or failed to create new tab
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized")

        try:
            # Store current tab handle
            original_tab = self.driver.current_window_handle

            # Open new tab using JavaScript
            self.driver.execute_script("window.open('', '_blank');")

            # Get all window handles
            all_tabs = self.driver.window_handles

            # Find the new tab (should be the last one)
            new_tab = None
            for tab in all_tabs:
                if tab != original_tab:
                    new_tab = tab
                    break

            if not new_tab:
                raise RuntimeError("Failed to create new tab")

            # Switch to new tab if requested
            if switch_to:
                self.driver.switch_to.window(new_tab)
                logger.info("Opened and switched to new tab")
            else:
                logger.info("Opened new tab without switching")

            self._notify("new_tab_opened", {"tab_handle": new_tab, "switched": switch_to})

        except WebDriverException as e:
            logger.error("Failed to open new tab: %s", e)
            self._notify("new_tab_failed", str(e))
            raise
        else:
            return new_tab

    def visit_run_page(self, entity: str, project: str, run_id: str) -> bool:
        """
        Visit a specific WandB run page, managing dedicated tabs for each run.

        Each unique run gets its own tab. If the same run is requested again,
        the method switches to the existing tab and refreshes the page.

        Args:
            entity: WandB entity (username or team name)
            project: WandB project name
            run_id: WandB run ID

        Returns:
            True if run page loaded successfully, False otherwise
        """
        # Construct the run URL
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        if not self.is_logged_in:
            logger.warning("Not logged in, cannot visit run page")
            return False

        if not self.driver:
            raise RuntimeError("Driver not initialized")

        try:
            # Check if we already have a tab for this run
            if run_url in self._run_tabs:
                existing_tab = self._run_tabs[run_url]

                # Verify the tab still exists
                current_tabs = self.get_all_tab_handles()
                if existing_tab in current_tabs:
                    # Switch to existing tab
                    self.switch_tab(existing_tab)
                    logger.debug("Switched to existing tab for run: %s/%s/%s", entity, project, run_id)

                    # Wait a short bit before refreshing
                    time.sleep(2)

                    # Refresh the page
                    self.driver.refresh()
                    logger.info("Refreshed run page: %s", run_url)

                    # Wait for page to load after refresh
                    WebDriverWait(self.driver, self.timeout).until(
                        lambda driver: driver.execute_script("return document.readyState") == "complete"
                    )

                    # Verify the run page is fully loaded
                    if "/runs/" in run_url:
                        run_loaded = False
                        for _ in range(10):
                            if self._check_run_page_loaded(run_url):
                                run_loaded = True
                                break
                            time.sleep(1)

                        if run_loaded:
                            logger.info("Run page refreshed and fully loaded: %s", run_url)
                        else:
                            logger.warning("Run page refreshed but 'Run details' button not found: %s", run_url)

                    self._notify(
                        "run_page_refreshed",
                        {"entity": entity, "project": project, "run_id": run_id, "tab_handle": existing_tab},
                    )
                    return True

                # Tab no longer exists, remove from tracking
                del self._run_tabs[run_url]
                logger.debug("Removed stale tab reference for run: %s", run_url)

            # Create new tab for this run
            try:
                new_tab = self.open_new_tab(switch_to=True)
                logger.info("Opened new dedicated tab for run: %s/%s/%s", entity, project, run_id)
            except RuntimeError as e:
                logger.error("Failed to open new tab for run page: %s", e)
                return False

            # Visit the run page in the new tab
            success = self.visit_wandb_page(run_url)

            if success:
                # Track this tab for this run
                self._run_tabs[run_url] = new_tab
                logger.debug("Successfully visited run page: %s/%s/%s in tab %s", entity, project, run_id, new_tab)
                self._notify(
                    "run_page_visited",
                    {
                        "entity": entity,
                        "project": project,
                        "run_id": run_id,
                        "new_tab": True,
                        "tab_handle": new_tab,
                    },
                )
            else:
                logger.error("Failed to visit run page: %s/%s/%s", entity, project, run_id)
                self._notify(
                    "run_page_visit_failed",
                    {"entity": entity, "project": project, "run_id": run_id, "new_tab": True},
                )

        except (TimeoutException, WebDriverException) as e:
            logger.error("Error visiting run page %s: %s", run_url, e)
            self._notify("run_page_error", {"url": run_url, "error": str(e)})
            return False
        else:
            return success

    def switch_tab(self, tab_handle: str) -> bool:
        """
        Switch to a specific browser tab.

        Args:
            tab_handle: Handle of the tab to switch to

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized")

        try:
            self.driver.switch_to.window(tab_handle)
            logger.debug("Switched to tab: %s", tab_handle)
            self._notify("tab_switched", tab_handle)

        except WebDriverException as e:
            logger.error("Failed to switch to tab %s: %s", tab_handle, e)
            self._notify("tab_switch_failed", {"tab_handle": tab_handle, "error": str(e)})
            return False
        else:
            return True

    def _cleanup_run_tab_tracking(self, tab_handle: str) -> None:
        """
        Clean up run tab tracking when a tab is closed.

        Args:
            tab_handle: Handle of the tab being closed
        """
        # Find and remove any run URLs that reference this tab
        urls_to_remove = [url for url, handle in self._run_tabs.items() if handle == tab_handle]
        for url in urls_to_remove:
            del self._run_tabs[url]
            logger.debug("Removed run tab tracking for URL: %s", url)

    def close_current_tab(self) -> bool:
        """
        Close the currently active tab.

        Returns:
            True if successful, False otherwise

        Note:
            If this is the last tab, the browser will be closed
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized")

        try:
            current_tab = self.driver.current_window_handle
            all_tabs = self.driver.window_handles

            # Clean up run tab tracking for the tab being closed
            self._cleanup_run_tab_tracking(current_tab)

            # Close current tab
            self.driver.close()

            # If there are other tabs, switch to the first available one
            if len(all_tabs) > 1:
                remaining_tabs = [tab for tab in all_tabs if tab != current_tab]
                if remaining_tabs:
                    self.driver.switch_to.window(remaining_tabs[0])
                    logger.info("Closed tab and switched to remaining tab")
                else:
                    logger.warning("No remaining tabs to switch to")
            else:
                logger.info("Closed last tab - browser will close")

            self._notify("tab_closed", current_tab)

        except WebDriverException as e:
            logger.error("Failed to close current tab: %s", e)
            self._notify("tab_close_failed", str(e))
            return False
        else:
            return True

    def get_current_tab_handle(self) -> str:
        """
        Get the handle of the currently active tab.

        Returns:
            Current tab handle

        Raises:
            RuntimeError: If driver is not initialized
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized")

        return self.driver.current_window_handle

    def get_all_tab_handles(self) -> list[str]:
        """
        Get handles of all open tabs.

        Returns:
            List of all tab handles

        Raises:
            RuntimeError: If driver is not initialized
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized")

        return self.driver.window_handles

    def get_run_tabs(self) -> dict[str, str]:
        """
        Get currently tracked run tabs.

        Returns:
            Dictionary mapping run URLs to their tab handles
        """
        return self._run_tabs.copy()

    def get_run_tab_count(self) -> int:
        """
        Get the number of currently tracked run tabs.

        Returns:
            Number of tracked run tabs
        """
        return len(self._run_tabs)

    def close_run_tab(self, entity: str, project: str, run_id: str) -> bool:
        """
        Close the tab for a specific run.

        Args:
            entity: WandB entity (username or team name)
            project: WandB project name
            run_id: WandB run ID

        Returns:
            True if tab was found and closed successfully, False otherwise
        """
        # Construct the run URL
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        if run_url not in self._run_tabs:
            logger.info("No tab found for run: %s/%s/%s", entity, project, run_id)
            return False

        if not self.driver:
            raise RuntimeError("Driver not initialized")

        tab_handle = self._run_tabs[run_url]

        try:
            # Verify the tab still exists
            current_tabs = self.get_all_tab_handles()
            if tab_handle not in current_tabs:
                # Tab no longer exists, remove from tracking
                del self._run_tabs[run_url]
                logger.debug("Removed stale tab reference for run: %s", run_url)
                return False

            # Get current tab to restore later if needed
            current_tab = self.driver.current_window_handle

            # Switch to the target tab
            self.driver.switch_to.window(tab_handle)
            logger.debug("Switched to tab for run: %s/%s/%s", entity, project, run_id)

            # Close the tab
            self.driver.close()

            # Remove from tracking
            del self._run_tabs[run_url]

            # Switch back to a remaining tab if we're not on the closed tab
            remaining_tabs = self.get_all_tab_handles()
            if remaining_tabs:
                if current_tab in remaining_tabs:
                    # Switch back to the original tab if it still exists
                    self.driver.switch_to.window(current_tab)
                else:
                    # Switch to the first available tab
                    self.driver.switch_to.window(remaining_tabs[0])
                logger.debug("Switched to remaining tab after closing run tab")
            else:
                logger.info("Closed last tab - browser will close")

            logger.info("Successfully closed tab for run: %s/%s/%s", entity, project, run_id)
            self._notify("run_tab_closed", {"entity": entity, "project": project, "run_id": run_id})

        except WebDriverException as e:
            logger.error("Failed to close tab for run %s/%s/%s: %s", entity, project, run_id, e)
            self._notify(
                "run_tab_close_failed", {"entity": entity, "project": project, "run_id": run_id, "error": str(e)}
            )
            return False
        else:
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
        """Clean up resources safely and thoroughly."""
        with self._lock:
            if self.driver:
                try:
                    # First try graceful shutdown
                    self.driver.quit()
                    logger.debug("Driver quit gracefully")
                except (WebDriverException, ConnectionError) as e:
                    logger.warning("Error during graceful driver cleanup: %s", e)
                    # If graceful shutdown failed, try force kill
                    logger.info("Attempting force kill of driver process")
                    self._force_kill_driver()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Unexpected error during driver cleanup: %s", e)
                    # Try force kill as last resort
                    self._force_kill_driver()
                finally:
                    self.driver = None
                    self.is_logged_in = False

            # Always try force kill as safety net if we have a PID
            elif self._driver_pid:
                logger.warning("Driver object is None but PID exists, force killing")
                self._force_kill_driver()

        # Clear run tab tracking
        self._run_tabs.clear()
        logger.debug("Cleared run tab tracking")

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
        """
        Destructor to ensure cleanup happens even if explicitly not called.

        This is a safety net for resource cleanup in case the object is
        garbage collected without proper cleanup being called.
        """
        try:
            # Only log if we actually have resources to clean up
            if self.driver is not None or self._driver_pid is not None:
                logger.debug("__del__ called, performing cleanup")
                self.cleanup()
        except Exception:
            try:
                if self._driver_pid:
                    self._force_kill_driver()
            except Exception:  # noqa: BLE001
                pass
            raise


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

                # Example of visiting a specific run page in a new tab
                success = session.visit_run_page("daraan", "dev-workspace", "example-run-id")
                if success:
                    print("Successfully opened run page in dedicated tab!")
                else:
                    print("Failed to open run page - run may not exist")

                # Example of visiting the same run again (will switch to existing tab and refresh)
                success2 = session.visit_run_page("daraan", "dev-workspace", "example-run-id")
                if success2:
                    print("Switched to existing tab and refreshed!")

                # Example of opening another run (will open a new tab)
                session.visit_run_page("daraan", "dev-workspace", "another-run-id")

                # Example of run tab management
                run_tabs = session.get_run_tabs()
                print(f"Currently tracking {len(run_tabs)} run tabs: {list(run_tabs.keys())}")

                all_tabs = session.get_all_tab_handles()
                print(f"Total browser tabs open: {len(all_tabs)}")

                # Example of closing a specific run tab
                if session.close_run_tab("daraan", "dev-workspace", "example-run-id"):
                    print("Successfully closed run tab!")
                    print(f"Run tabs remaining: {session.get_run_tab_count()}")

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

            # Example of opening run pages in threaded mode (each gets its own tab)
            session.visit_run_page("daraan", "dev-workspace", "run-1")
            session.visit_run_page("daraan", "dev-workspace", "run-2")
            session.visit_run_page("daraan", "dev-workspace", "run-1")  # Will switch to existing tab and refresh

            # Example of run tab management in threaded mode
            run_tabs = session.get_run_tabs()
            print(f"Tracking {len(run_tabs)} run tabs in background session")

            # Example of closing a specific run tab
            session.close_run_tab("daraan", "dev-workspace", "run-1")
            print(f"Run tabs after closing: {session.get_run_tab_count()}")

            # Example of switching between remaining tabs
            tabs = session.get_all_tab_handles()
            if len(tabs) > 1:
                print(f"Switching between {len(tabs)} tabs")
                # Switch to first tab
                session.switch_tab(tabs[0])
                time.sleep(2)
                # Switch back to second tab
                session.switch_tab(tabs[1])

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
