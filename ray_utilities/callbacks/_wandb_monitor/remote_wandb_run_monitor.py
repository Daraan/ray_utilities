from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import ray
from ray.actor import ActorHandle, ActorProxy

from .wandb_run_monitor import WandbRunMonitor
from queue import Queue

if TYPE_CHECKING:
    from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login_mp import WandBCredentials
    from wandb import Api


class _RemoteWandbRunMonitor(WandbRunMonitor):
    def __init__(
        self,
        credentials: WandBCredentials | None = None,
        *,
        project: str,
        entity: str | None = None,
        browser: str = "firefox",
        headless: bool = True,
        timeout: int = 30,
        callback: Callable[[str, Any], None] | None = None,
        wandb_api: Api | None = None,
    ):
        super().__init__(
            credentials,
            project=project,
            entity=entity,
            browser=browser,
            headless=headless,
            timeout=timeout,
            callback=callback,
            wandb_api=wandb_api,
        )
        self._command_queue = Queue()
        self._started = False

    async def send_command(self, command: str, *args, **kwargs) -> Any:
        """Send a command to the monitor and wait for the result."""
        self._command_queue.put((command, args, kwargs))

    def run(self):
        if self._started:
            return
        self._started = True
        self.initialize()


RemoteWandbRunMonitor = ray.remote(_RemoteWandbRunMonitor)

REMOTE_WANDB_MONITOR_NAME = "remote_wandb_run_monitor"


def get_remote_wandb_run_monitor(
    credentials: WandBCredentials | None = None,
    *,
    project: str,
    entity: str | None = None,
    browser: str = "firefox",
    headless: bool = True,
    timeout: int = 30,
    callback: Callable[[str, Any], None] | None = None,
    wandb_api: Api | None = None,
    # actor options
    num_cpus: int = 1,
    name: str = REMOTE_WANDB_MONITOR_NAME,
    actor_options: dict[str, Any] | None = None,
) -> ActorHandle[_RemoteWandbRunMonitor] | ActorProxy[_RemoteWandbRunMonitor]:
    """Create a remote WandbRunMonitor actor."""
    if actor_options is None:
        actor_options = {
            "num_cpus": num_cpus,
            "max_restarts": -1,
            "max_task_retries": -1,
        }
    else:
        actor_options = {
            "num_cpus": num_cpus,
            "max_restarts": -1,
            "max_task_retries": -1,
            **actor_options,
        }
    return RemoteWandbRunMonitor.options(name=name, **actor_options, get_if_exists=True, runtime_env={}).remote(
        credentials,
        project=project,
        entity=entity,
        browser=browser,
        headless=headless,
        timeout=timeout,
        callback=callback,
        wandb_api=wandb_api,
    )
