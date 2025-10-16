"""
Selenium script for WandB login with multiprocessing design.
This script handles automated login to wandb.ai and provides a foundation
for visiting other WandB websites and using the WandB API.

This is a multiprocessing version of the original threading implementation,
maintaining the same API while running Selenium operations in a separate process.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import time
from typing_extensions import deprecated
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import wandb
from ray_utilities.callbacks._wandb_monitor._wandb_session_cache import WandBSessionCache

logger = logging.getLogger(__name__)


@dataclass
class WandBCredentials:
    """Container for WandB login credentials."""

    username: str
    password: str
    api_key: Optional[str] = None


@dataclass
class SeleniumCommand:
    """Command structure for inter-process communication."""

    action: str  # 'login', 'visit_page', 'cleanup', 'stop'
    data: Optional[dict] = None
    request_id: Optional[str] = None  # For response correlation instead of queue


@dataclass
class SeleniumResponse:
    """Response structure for inter-process communication."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


def _selenium_worker_process(
    credentials: WandBCredentials,
    browser: str,
    headless: bool,
    timeout: int,
    use_cache: bool,
    cache_dir: Optional[str],
    command_queue: multiprocessing.Queue,
    status_queue: multiprocessing.Queue,
) -> None:
    """
    Worker process that handles all Selenium operations.

    This runs in a separate process and communicates via queues.
    """
    # Setup logging for the worker process
    worker_logger = logging.getLogger(f"{__name__}.worker")

    driver: Optional[webdriver.Remote] = None
    session_cache: Optional[WandBSessionCache] = None
    is_logged_in = False

    def notify_status(status: str, data: Any = None) -> None:
        """Send status notification via queue."""
        try:
            status_queue.put((status, data), timeout=1)
        except queue.Full:
            worker_logger.warning("Status queue full, dropping notification: %s", status)

    def setup_driver() -> webdriver.Remote:
        """Setup and return the appropriate WebDriver."""
        nonlocal driver

        if browser == "chrome":
            options = ChromeOptions()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)
            return driver

        if browser == "firefox":
            options = FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            # Explicitly set Firefox binary path to avoid detection issues
            firefox_paths = [
                "/snap/firefox/current/usr/lib/firefox/firefox",
                "/usr/bin/firefox",
                "/usr/lib/firefox/firefox",
            ]
            for path in firefox_paths:
                if os.path.exists(path):
                    options.binary_location = path
                    break
            driver = webdriver.Firefox(options=options)
            return driver

        raise ValueError(f"Unsupported browser: {browser}")

    def wait_for_element(by: str, locator: str, element_timeout: Optional[int] = None) -> Any:
        """Wait for an element to be present and return it."""
        if not driver:
            raise RuntimeError("Driver not initialized")
        wait_time = element_timeout or timeout
        wait = WebDriverWait(driver, wait_time)
        return wait.until(EC.presence_of_element_located((by, locator)))

    def wait_for_clickable(by: str, locator: str, element_timeout: Optional[int] = None) -> Any:
        """Wait for an element to be clickable and return it."""
        if not driver:
            raise RuntimeError("Driver not initialized")
        wait_time = element_timeout or timeout
        wait = WebDriverWait(driver, wait_time)
        return wait.until(EC.element_to_be_clickable((by, locator)))

    def check_team_present() -> bool | None:
        """Check if the expected team name profile is visible in the top right corner."""
        if not driver:
            return False

        try:
            # Get expected team name from environment variable, default to "DaraanWandB"
            expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", None)
            if not expected_team_name:
                worker_logger.debug("No team name specified for verification, skipping check")
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
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        worker_logger.debug("Found logged-out indicator: %s", selector)
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
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if expected_team_name in element.text:
                            worker_logger.debug(
                                "Found expected team name '%s' in profile indicator", expected_team_name
                            )
                            return True
                except Exception:
                    continue

            # Fallback: check page source for expected team name
            page_source = driver.page_source
            if expected_team_name in page_source:
                worker_logger.debug("Found expected team name '%s' in page source", expected_team_name)
                return True

            worker_logger.debug("Expected team name '%s' not found in profile", expected_team_name)
            return False

        except Exception as e:
            worker_logger.debug("Error checking for team profile: %s", e)
            return False

    def check_run_page_loaded(url: str) -> bool:
        """Check if a WandB run page has fully loaded by looking for 'Run details' button."""
        if not driver:
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
                        elements = driver.find_elements(By.XPATH, selector)
                    else:
                        # CSS selector
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)

                    if elements:
                        worker_logger.debug("Found 'Run details' button - run page loaded successfully")
                        return True
                except Exception:
                    continue

            # Fallback: check page source for "Run details" text
            page_source = driver.page_source
            if "Run details" in page_source:
                worker_logger.debug("Found 'Run details' text in page source")
                return True

            worker_logger.debug("'Run details' button not found - run page may not be fully loaded")
            return False

        except Exception as e:
            worker_logger.debug("Error checking if run page loaded: %s", e)
            return False

    def load_cached_cookies() -> bool:
        """Load and apply cached cookies if available."""
        nonlocal is_logged_in

        if not use_cache or not session_cache:
            return False

        if not driver:
            return False

        try:
            # Load cached cookies
            cached_cookies = session_cache.load_cookies(
                credentials.username,
                browser,
                max_age_hours=24.0,
            )

            if not cached_cookies:
                return False

            # Navigate to WandB domain first to set cookies
            driver.get("https://wandb.ai")
            time.sleep(2)

            # Apply cached cookies
            cookies_applied = 0
            for cookie in cached_cookies:
                try:
                    cookie_data = {
                        k: v
                        for k, v in cookie.items()
                        if k in ["name", "value", "domain", "path", "secure", "httpOnly"]
                    }

                    if cookie_data.get("domain") and not cookie_data["domain"].endswith("wandb.ai"):
                        continue

                    driver.add_cookie(cookie_data)
                    cookies_applied += 1
                except WebDriverException as e:
                    worker_logger.debug("Failed to add cookie %s: %s", cookie.get("name", "unknown"), e)
                    continue

            worker_logger.debug("Applied %d out of %d cached cookies", cookies_applied, len(cached_cookies))

            # Test session validity
            test_run_url = "https://wandb.ai/daraan/dev-workspace/"
            driver.get(test_run_url)
            time.sleep(7)

            current_url = driver.current_url
            page_source = driver.page_source.lower()

            worker_logger.debug("Current URL after applying cookies: %s", current_url)

            # Check basic authentication indicators
            basic_auth_check = (
                "wandb.ai" in current_url
                and "auth0.com" not in current_url
                and "stumbled on an empty page" not in page_source
                and ("login" not in current_url or "runs/" in current_url)
            )

            # Check for Team profile to verify specific user authentication
            team_profile_check = check_team_present() if basic_auth_check else False

            is_authenticated = basic_auth_check and (team_profile_check is not False)

            if is_authenticated:
                expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", None)
                worker_logger.info(
                    "Successfully restored session using cached cookies with %s profile verified", expected_team_name
                )
                is_logged_in = True
                notify_status("cached_login_success")

                # Note: API key is not cached for security reasons
                # It should be provided via environment variables only

                return True

            if basic_auth_check and not team_profile_check:
                expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", None)
                worker_logger.warning(
                    "Basic authentication passed but %s profile not found - cached session may be for different user",
                    expected_team_name,
                )

            worker_logger.debug("Cached session validation failed - URL: %s", current_url)
            return False

        except Exception as e:
            worker_logger.debug("Failed to load cached cookies: %s", e)
            return False

    def save_current_cookies() -> None:
        """Save current browser cookies to cache."""
        if not use_cache or not session_cache or not driver:
            return

        try:
            cookies = driver.get_cookies()
            if cookies:
                session_cache.save_cookies(credentials.username, browser, cookies)
                worker_logger.debug("Saved %d cookies to cache", len(cookies))

            # Note: API key is not saved to cache for security reasons
            # It should be provided via environment variables only

        except Exception as e:
            worker_logger.debug("Failed to save cookies to cache: %s", e)

    def perform_login() -> bool:
        """Perform WandB login via Selenium."""
        nonlocal is_logged_in

        if not driver:
            setup_driver()
            assert driver

        try:
            notify_status("starting_login")

            # First, try to use cached cookies if available
            if load_cached_cookies():
                return True

            # If cached login failed, proceed with manual login
            worker_logger.info("Cached login failed or not available, performing manual login")

            # Navigate to WandB login page
            login_url = "https://app.wandb.ai/login?_gl=1*1njlh40*_ga*MjAzMDY0NTMxOC4xNjg2NzMzODEw*_ga_JH1SJHJQXJ*MTY5MDk5MDgxNS4xMjEuMS4xNjkwOTkxMDExLjYwLjAuMA.."
            driver.get(login_url)
            worker_logger.info("Navigated to WandB login page")

            time.sleep(2)
            current_url = driver.current_url
            worker_logger.info("Current URL after navigation: %s", current_url)

            # Check if already logged in
            if (
                "wandb.ai" in current_url
                and "auth0.com" not in current_url
                and "login" not in current_url
                and ("home" in current_url or "dashboard" in current_url)
            ):
                # Additional verification: check for expected team profile
                if check_team_present():
                    expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", "DaraanWandB")
                    worker_logger.info(
                        "Already logged in with %s profile verified, no manual login needed", expected_team_name
                    )
                    is_logged_in = True
                    notify_status("login_success")
                    save_current_cookies()
                    return True
                else:
                    expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", "DaraanWandB")
                    worker_logger.warning(
                        "Detected login state but %s profile not found - proceeding with manual login",
                        expected_team_name,
                    )

            # Look for email field
            email_selectors = [
                "input[id='1-email']",
                "input.auth0-lock-input[type='email']",
                "input[name='email'].auth0-lock-input",
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
                    username_field = wait_for_element("css selector", selector, 5)
                    worker_logger.info("Found email field with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not username_field:
                worker_logger.error("Could not find email input field")
                return False

            # Scroll to element and make sure it's visible
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", username_field)
            time.sleep(1)

            username_field.clear()
            username_field.send_keys(credentials.username)
            worker_logger.info("Entered username: %s", credentials.username[0] + "***" + credentials.username[-4:])

            # Look for password field
            password_selectors = [
                "input[id='1-password']",
                "input.auth0-lock-input[type='password']",
                "input[name='password'].auth0-lock-input",
                "input[name='password']",
                "input[type='password']",
                "input[data-testid='password']",
                "input#password",
                "input[autocomplete='current-password']",
            ]

            password_field = None
            for selector in password_selectors:
                try:
                    password_field = wait_for_element("css selector", selector, 5)
                    worker_logger.info("Found password field with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not password_field:
                worker_logger.error("Could not find password input field")
                return False

            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", password_field)
            time.sleep(1)

            password_field.clear()
            password_field.send_keys(credentials.password)
            worker_logger.info("Entered password")

            # Look for login button
            login_button_selectors = [
                "button[type='submit']",
                "button.auth0-lock-submit",
                ".auth0-lock-submit",
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
                    login_button = wait_for_clickable("css selector", selector, 5)
                    worker_logger.info("Found login button with selector: %s", selector)
                    break
                except TimeoutException:
                    continue

            if not login_button:
                worker_logger.error("Could not find login button")
                return False

            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", login_button)
            time.sleep(2)

            login_button.click()
            worker_logger.info("Clicked login button")

            # Wait for successful login
            try:
                WebDriverWait(driver, timeout).until(
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

                # Additional verification: check for expected team profile
                if check_team_present():
                    is_logged_in = True
                    expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", "DaraanWandB")
                    worker_logger.info("Successfully logged in to WandB with %s profile verified", expected_team_name)
                    worker_logger.info("Final URL: %s", driver.current_url)
                    notify_status("login_success")
                    save_current_cookies()
                else:
                    expected_team_name = os.getenv("WANDB_VIEWER_TEAM_NAME", "DaraanWandB")
                    worker_logger.warning(
                        "Login detected but %s profile not found - may be logged in as different user",
                        expected_team_name,
                    )
                    return False

            except TimeoutException:
                # Check for error messages
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
                    error_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if error_elements and error_elements[0].text.strip():
                        error_text = error_elements[0].text
                        break

                worker_logger.error("Login failed: %s", error_text)
                worker_logger.error("Current URL: %s", driver.current_url)
                notify_status("login_failed", error_text)
                return False

        except (TimeoutException, WebDriverException) as e:
            worker_logger.error("Login error: %s", e)
            worker_logger.error("Current URL: %s", driver.current_url if driver else "Unknown")
            notify_status("login_error", str(e))
            return False

        return True

    def visit_page(url: str) -> bool:
        """Visit a WandB page and wait for it to load."""
        if not is_logged_in:
            worker_logger.warning("Not logged in, cannot visit page")
            return False

        if not driver:
            raise RuntimeError("Driver not initialized")

        try:
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )

            # Check if we ended up on a 404 page
            page_source = driver.page_source.lower()
            current_url = driver.current_url

            if "stumbled on an empty page" in page_source or ("404" in page_source and "wandb.ai" in current_url):
                worker_logger.error("Accessed private page without authentication: %s", url)
                notify_status(
                    "page_visit_failed", {"url": url, "error": "404 - Private page or authentication required"}
                )
                return False

            # Special verification for run pages - wait for content to load
            if "/runs/" in url:
                worker_logger.debug("Run page detected, waiting for run content to load...")

                # Wait up to 10 seconds for run details to appear
                run_loaded = False
                for _ in range(10):
                    if check_run_page_loaded(url):
                        run_loaded = True
                        break
                    time.sleep(1)

                if not run_loaded:
                    worker_logger.warning(
                        "Run page may not have fully loaded - 'Run details' button not found: %s", url
                    )
                    # Don't fail the visit, just warn - some run pages might have different layouts
                else:
                    worker_logger.info("Run page fully loaded with 'Run details' button verified: %s", url)

            worker_logger.info("Successfully visited: %s", url)
            notify_status("page_visited", url)

        except (TimeoutException, WebDriverException) as e:
            worker_logger.error("Failed to visit %s: %s", url, e)
            notify_status("page_visit_failed", {"url": url, "error": str(e)})
            return False

        return True

    def cleanup_driver() -> None:
        """Clean up driver resources."""
        nonlocal driver, is_logged_in

        if driver:
            try:
                driver.quit()
            except (WebDriverException, ConnectionError) as e:
                worker_logger.warning("Error during driver cleanup: %s", e)
            finally:
                driver = None
                is_logged_in = False

    # Initialize session cache
    if use_cache:
        try:
            cache_path = Path(cache_dir) if cache_dir else None
            session_cache = WandBSessionCache(cache_path)
        except (ImportError, OSError, ValueError) as e:
            worker_logger.warning("Failed to initialize session cache: %s", e)
            session_cache = None

    notify_status("worker_process_started")

    try:
        # Main worker loop
        while True:
            try:
                # Get command from queue (blocking with timeout)
                command: SeleniumCommand = command_queue.get(timeout=1)

                if command.action == "stop":
                    worker_logger.info("Received stop command")
                    break

                # Handle setup_driver command
                if command.action == "setup_driver":
                    try:
                        setup_driver()
                        notify_status("driver_initialized")
                        if command.request_id:
                            notify_status("response_setup_driver", {"request_id": command.request_id, "success": True})
                    except (WebDriverException, ValueError, OSError) as e:
                        worker_logger.error("Failed to setup driver: %s", e)
                        notify_status("driver_init_failed", str(e))
                        if command.request_id:
                            notify_status(
                                "response_setup_driver",
                                {"request_id": command.request_id, "success": False, "error": str(e)},
                            )

                elif command.action == "login":
                    try:
                        success = perform_login()
                        if command.request_id:
                            response_data = {"is_logged_in": is_logged_in}
                            notify_status(
                                "response_login",
                                {"request_id": command.request_id, "success": success, "data": response_data},
                            )
                    except (WebDriverException, TimeoutException, ValueError) as e:
                        worker_logger.error("Login failed: %s", e)
                        if command.request_id:
                            notify_status(
                                "response_login", {"request_id": command.request_id, "success": False, "error": str(e)}
                            )

                elif command.action == "visit_page":
                    try:
                        url = command.data.get("url") if command.data else None
                        if not url:
                            msg = "No URL provided for visit_page command"
                            raise ValueError(msg)
                        success = visit_page(url)
                        if command.request_id:
                            notify_status("response_visit_page", {"request_id": command.request_id, "success": success})
                    except (WebDriverException, TimeoutException, ValueError) as e:
                        worker_logger.error("Failed to visit page: %s", e)
                        if command.request_id:
                            notify_status(
                                "response_visit_page",
                                {"request_id": command.request_id, "success": False, "error": str(e)},
                            )

                elif command.action == "cleanup":
                    cleanup_driver()
                    notify_status("cleanup_complete")
                    if command.request_id:
                        notify_status("response_cleanup", {"request_id": command.request_id, "success": True})

                else:
                    worker_logger.warning("Unknown command: %s", command.action)
                    if command.request_id:
                        notify_status(
                            "response_unknown",
                            {
                                "request_id": command.request_id,
                                "success": False,
                                "error": f"Unknown command: {command.action}",
                            },
                        )

            except queue.Empty:
                # No command received, continue the loop
                continue
            except (ValueError, RuntimeError) as e:
                worker_logger.error("Error processing command: %s", e)
                continue

    except KeyboardInterrupt:
        worker_logger.info("Worker process interrupted")
    except (WebDriverException, TimeoutException) as e:
        worker_logger.error("Worker process error: %s", e)
    finally:
        cleanup_driver()
        notify_status("worker_process_stopped")
        worker_logger.info("Worker process finished")


class WandBSeleniumSession:
    """
    Multiprocessing WandB Selenium session for automated login and web interactions.

    This class provides a process-safe way to manage WandB login via Selenium
    running in a separate process, maintaining the same API as the threading version.
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

        # Process and communication setup
        self.process: Optional[Any] = None  # Can be either multiprocessing.Process or spawn context Process
        self.command_queue: Optional[multiprocessing.Queue] = None
        self.status_queue: Optional[multiprocessing.Queue] = None
        self.is_logged_in = False

        # For maintaining API compatibility
        self.driver = None  # This won't be used but maintains API compatibility

    def _notify(self, status: str, data: Any = None) -> None:
        """Send notification via callback if available."""
        if self.callback:
            try:
                self.callback(status, data)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Callback error: %s", e)

    def _send_command(
        self, action: str, data: Optional[dict] = None, *, wait_for_response: bool = True
    ) -> Optional[SeleniumResponse]:
        """Send a command to the worker process and optionally wait for response."""
        if not self.command_queue:
            raise RuntimeError("Process not started")

        request_id = str(uuid.uuid4()) if wait_for_response else None
        command = SeleniumCommand(action=action, data=data, request_id=request_id)

        try:
            self.command_queue.put(command, timeout=5)
        except queue.Full:
            logger.error("Command queue full, could not send command: %s", action)
            return SeleniumResponse(success=False, error="Command queue full")

        if wait_for_response and request_id:
            # Wait for response with the matching request_id in status updates
            start_time = time.time()
            timeout_duration = self.timeout + 10

            while time.time() - start_time < timeout_duration:
                try:
                    # Check for responses in status queue
                    while True:
                        try:
                            status, data = self.status_queue.get_nowait()
                            if status.startswith("response_") and data and data.get("request_id") == request_id:
                                # Found our response
                                return SeleniumResponse(
                                    success=data.get("success", False), data=data.get("data"), error=data.get("error")
                                )
                            else:
                                # Regular status update, forward to callback
                                self._notify(status, data)
                        except queue.Empty:
                            break

                    time.sleep(0.1)  # Small sleep to avoid busy waiting
                except Exception as e:  # ruff: noqa: BLE001
                    logger.warning("Error while waiting for response: %s", e)
                    break

            logger.error("Timeout waiting for response to command: %s", action)
            return SeleniumResponse(success=False, error="Timeout waiting for response")

        return None

    def _process_status_updates(self) -> None:
        """Process status updates from the worker process."""
        if not self.status_queue:
            return

        while True:
            try:
                status, data = self.status_queue.get_nowait()
                self._notify(status, data)
            except queue.Empty:
                break
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Error processing status update: %s", e)
                break

    @deprecated("Dummy method for thread API compatibility.")
    def _setup_driver(self) -> webdriver.Remote | object:
        """Setup driver in the worker process (for API compatibility)."""
        if not self.process or not self.process.is_alive():
            self._start_process()

        response = self._send_command("setup_driver")
        if response and response.success:
            self._notify("driver_initialized")
            # Return a dummy object for API compatibility
            return type("DummyDriver", (), {})()
        else:
            error_msg = response.error if response else "Unknown error"
            raise RuntimeError(f"Failed to setup driver: {error_msg}")

    def _start_process(self) -> None:
        """Start the worker process."""
        if self.process and self.process.is_alive():
            logger.warning("Process already running")
            return

        # If we have a dead process, clean it up first
        if self.process and not self.process.is_alive():
            logger.debug("Cleaning up previous dead process")
            self.process = None
            self.command_queue = None
            self.status_queue = None

        # Use spawn context for better isolation with Selenium
        ctx = multiprocessing.get_context("spawn")

        # Create communication queues
        self.command_queue = ctx.Queue()
        self.status_queue = ctx.Queue()

        # Start worker process
        cache_dir_str = str(self.session_cache.cache_dir) if self.session_cache else None

        self.process = ctx.Process(
            target=_selenium_worker_process,
            args=(
                self.credentials,
                self.browser,
                self.headless,
                self.timeout,
                self.use_cache,
                cache_dir_str,
                self.command_queue,
                self.status_queue,
            ),
            name="wandb-selenium-worker",
            daemon=True,
        )
        self.process.start()
        logger.info("Started WandB Selenium worker process (PID: %s)", self.process.pid)

        # Give the process a moment to initialize
        time.sleep(1)
        self._process_status_updates()

    def login(self) -> bool:
        """
        Perform WandB login via Selenium in the worker process.

        Returns:
            True if login successful, False otherwise
        """
        if not self.process or not self.process.is_alive():
            self._start_process()

        self._notify("starting_login")

        response = self._send_command("login")
        self._process_status_updates()

        if response and response.success:
            self.is_logged_in = response.data.get("is_logged_in", False) if response.data else False
            # Note: API key is handled separately via environment variables for security
            return True

        error_msg = response.error if response else "Unknown error"
        logger.error("Login failed: %s", error_msg)
        return False

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

        if not self.process or not self.process.is_alive():
            logger.error("Worker process not running")
            return False

        response = self._send_command("visit_page", {"url": url})
        self._process_status_updates()

        if response and response.success:
            return True

        error_msg = response.error if response else "Unknown error"
        logger.error("Failed to visit page %s: %s", url, error_msg)
        return False

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

    def run_threaded(self) -> Any:
        """
        Start the login process in a separate process (maintains API compatibility).

        Returns:
            Process object for the worker process
        """
        if self.process and self.process.is_alive():
            logger.warning("Process already running")
            return self.process

        self._start_process()

        # Automatically perform login in the background
        login_success = self.login()

        if login_success and self.credentials.api_key:
            self.initialize_wandb_api()

        logger.info("Started WandB login process")
        return self.process

    def stop(self) -> None:
        """Stop the worker process."""
        if self.command_queue:
            try:
                self._send_command("stop", wait_for_response=False)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Error sending stop command: %s", e)

        if self.process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                logger.warning("Process did not stop gracefully, terminating")
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    logger.error("Process still alive after termination")
            logger.info("Stopped WandB session process")

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
        if self.command_queue:
            try:
                self._send_command("cleanup", wait_for_response=False)
            except Exception as e:  # ruff: noqa: BLE001
                logger.warning("Error sending cleanup command: %s", e)

        self.stop()

        # Clean up queues
        if self.command_queue:
            try:  # noqa: SIM105
                # Clear any remaining items
                while not self.command_queue.empty():
                    self.command_queue.get_nowait()
            except Exception:  # ruff: noqa: BLE001
                pass
            self.command_queue = None

        if self.status_queue:
            try:  # noqa: SIM105
                # Process any remaining status updates
                self._process_status_updates()
            except Exception:  # ruff: noqa: BLE001
                pass
            self.status_queue = None

        self.process = None
        self.is_logged_in = False
        self.driver = None

        self._notify("cleanup_complete")
        logger.info("Cleaned up WandB session")

    def __enter__(self):
        """Context manager entry."""
        self._start_process()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:  # noqa: SIM105
            self.cleanup()
        except Exception:  # noqa: BLE001
            pass


def example_usage():
    """Example usage of the WandB Selenium session."""

    def status_callback(status: str, data: Any = None):
        """Example callback for status updates."""
        # Note: Only print status, not data which might contain sensitive information
        print(f"Status: {status}")

    # Setup credentials (using environment variables - not shown for security)
    credentials = WandBCredentials(
        username=os.environ.get("WANDB_VIEWER_MAIL", ""),
        password=os.environ.get("WANDB_VIEWER_PW", ""),
        api_key=os.environ.get("WANDB_API_KEY", None),
    )

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

                time.sleep(5)  # Do some work
            else:
                print("Login failed!")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Process-based usage
    print("\nExample 2: Process-based usage")
    session = WandBSeleniumSession(credentials, callback=status_callback)

    try:
        # Start in process
        process = session.run_threaded()

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
