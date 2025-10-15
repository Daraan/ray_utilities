import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WandBSessionCache:
    """
    Manages cached session data (cookies, API keys) for WandB sessions.

    This class provides persistent storage of authentication data to avoid
    repeated logins across script executions.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the session cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.wandb_session_cache
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".wandb_session_cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, mode=0o700)  # Secure directory permissions

        self.cookies_file = self.cache_dir / "cookies.json"
        self.session_file = self.cache_dir / "session_data.json"

    def _get_cache_key(self, username: str, browser: str) -> str:
        """Generate a unique cache key for username/browser combination."""
        return f"{username}_{browser}".replace("@", "_at_").replace(".", "_dot_")

    def save_cookies(self, username: str, browser: str, cookies: List[Dict[str, Any]]) -> bool:
        """
        Save cookies for a specific user/browser combination.

        Args:
            username: WandB username/email
            browser: Browser type ("chrome" or "firefox")
            cookies: List of cookie dictionaries from selenium

        Returns:
            True if cookies saved successfully, False otherwise
        """
        try:
            cache_key = self._get_cache_key(username, browser)

            # Load existing cookies data or create new
            cookies_data = {}
            if self.cookies_file.exists():
                with open(self.cookies_file, "r") as f:
                    cookies_data = json.load(f)

            # Store cookies with timestamp
            cookies_data[cache_key] = {
                "cookies": cookies,
                "timestamp": time.time(),
                "username": username,
                "browser": browser,
            }

            # Write back to file with secure permissions
            with open(self.cookies_file, "w") as f:
                json.dump(cookies_data, f, indent=2)

            # Set secure file permissions
            self.cookies_file.chmod(0o600)

            logger.debug("Saved %d cookies for %s (%s)", len(cookies), username, browser)
            return True

        except Exception as e:
            logger.warning("Failed to save cookies: %s", e)
            return False

    def load_cookies(
        self, username: str, browser: str, max_age_hours: float = 21 * 24.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached cookies for a specific user/browser combination.

        Args:
            username: WandB username/email
            browser: Browser type ("chrome" or "firefox")
            max_age_hours: Maximum age of cookies in hours before considering them stale

        Returns:
            List of cookie dictionaries if valid cookies found, None otherwise
        """
        try:
            if not self.cookies_file.exists():
                return None

            with open(self.cookies_file, "r") as f:
                cookies_data = json.load(f)

            cache_key = self._get_cache_key(username, browser)

            if cache_key not in cookies_data:
                logger.debug("No cached cookies found for USERNAME (%s)", browser)
                return None

            cached_entry = cookies_data[cache_key]

            # Check if cookies are too old
            age_hours = (time.time() - cached_entry["timestamp"]) / 3600
            if age_hours > max_age_hours:
                logger.debug(
                    "Cached cookies for USERNAME (%s) are %.1f hours old, considered stale", browser, age_hours
                )
                return None

            cookies = cached_entry["cookies"]
            logger.debug(
                "Loaded %d cached cookies for USERNAME (%s), age: %.1f hours", len(cookies), browser, age_hours
            )
            return cookies

        except Exception as e:
            logger.warning("Failed to load cookies: %s", e)
            return None

    def save_session_data(self, username: str, api_key: Optional[str] = None) -> bool:
        """
        Save session data. Note: API keys are not saved for security reasons.

        Args:
            username: WandB username/email
            api_key: WandB API key (ignored for security - use environment variables)

        Returns:
            True if session data saved successfully, False otherwise
        """
        try:
            # Load existing session data or create new
            session_data = {}
            if self.session_file.exists():
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)

            cache_key = self._get_cache_key(username, "")

            # Store session data with timestamp (API key not stored for security)
            session_data[cache_key] = {"timestamp": time.time(), "username": username}

            # Write back to file with secure permissions
            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)

            # Set secure file permissions
            self.session_file.chmod(0o600)

            logger.debug("Saved session data for %s", username)
            return True

        except Exception as e:
            logger.warning("Failed to save session data: %s", e)
            return False

    def load_session_data(self, username: str, max_age_hours: float = 168.0) -> Optional[Dict[str, Any]]:
        """
        Load cached session data for a specific user.

        Args:
            username: WandB username/email
            max_age_hours: Maximum age of session data in hours (default: 7 days)

        Returns:
            Dictionary with session data if valid data found, None otherwise
        """
        try:
            if not self.session_file.exists():
                return None

            with open(self.session_file, "r") as f:
                session_data = json.load(f)

            cache_key = self._get_cache_key(username, "")

            if cache_key not in session_data:
                return None

            cached_entry = session_data[cache_key]

            # Check if session data is too old
            age_hours = (time.time() - cached_entry["timestamp"]) / 3600
            if age_hours > max_age_hours:
                logger.debug("Cached session data for %s is %.1f hours old, considered stale", username, age_hours)
                return None

            logger.debug("Loaded cached session data for %s, age: %.1f hours", username, age_hours)
            return cached_entry

        except Exception as e:
            logger.warning("Failed to load session data: %s", e)
            return None

    def clear_cache(self, username: Optional[str] = None) -> bool:
        """
        Clear cached data for a specific user or all users.

        Args:
            username: Username to clear cache for, or None to clear all

        Returns:
            True if cache cleared successfully, False otherwise
        """
        try:
            if username is None:
                # Clear all cache files
                if self.cookies_file.exists():
                    self.cookies_file.unlink()
                if self.session_file.exists():
                    self.session_file.unlink()
                logger.info("Cleared all cached session data")
                return True
            else:
                # Clear specific user's data
                cache_key_prefix = self._get_cache_key(username, "")

                # Clear cookies
                if self.cookies_file.exists():
                    with open(self.cookies_file, "r") as f:
                        cookies_data = json.load(f)

                    keys_to_remove = [k for k in cookies_data.keys() if k.startswith(cache_key_prefix)]
                    for key in keys_to_remove:
                        del cookies_data[key]

                    with open(self.cookies_file, "w") as f:
                        json.dump(cookies_data, f, indent=2)

                # Clear session data
                if self.session_file.exists():
                    with open(self.session_file, "r") as f:
                        session_data = json.load(f)

                    if cache_key_prefix in session_data:
                        del session_data[cache_key_prefix]

                    with open(self.session_file, "w") as f:
                        json.dump(session_data, f, indent=2)

                logger.info("Cleared cached session data for %s", username)
                return True

        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)
            return False
