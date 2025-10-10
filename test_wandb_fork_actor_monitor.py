from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
import dotenv

import ray
import wandb
import wandb.sdk
from ray_utilities.callbacks._wandb_monitor.remote_wandb_run_monitor import get_remote_wandb_run_monitor
from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import RemoteWandbRunMonitor, WandbRunMonitor
from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login import logger as session_logger
from ray.air.util.node import _force_on_current_node


from ray_utilities.callbacks.tuner._adv_wandb_logging_actor import _WandbLoggingActorWithArtifactSupport
from ray.util.queue import Queue

if TYPE_CHECKING:
    from ray._private.worker import RemoteFunction3
    from ray.actor import ActorHandle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
session_logger.setLevel(logging.DEBUG)

ENTITY = "daraan"
PROJECT = "dev-workspace"

dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())


class TestLoggingActor(_WandbLoggingActorWithArtifactSupport):
    def run(self, retries=0):
        global_monitor = RemoteWandbRunMonitor.get_remote_monitor(project=self.kwargs["project"])
        breakpoint()
        super().run(retries)


def setup_actor(**wandb_init_kwargs) -> tuple[ActorHandle[_WandbLoggingActorWithArtifactSupport], Queue]:
    # define remote logger class
    remote_decorator = ray.remote(
        num_cpus=0,
        **_force_on_current_node(),
        runtime_env={
            "env_vars": {
                "WANDB_VIEWER_MAIL": os.getenv("WANDB_VIEWER_MAIL", ""),
                "WANDB_VIEWER_PWD": os.getenv("WANDB_VIEWER_PWD", ""),
            }
        },
        max_restarts=-1,
        max_task_retries=-1,
    )
    logger_class: RemoteFunction3[_WandbLoggingActorWithArtifactSupport, Any, Any, Any, Any] = remote_decorator(
        _WandbLoggingActorWithArtifactSupport
    )  # pyright: ignore[reportAssignmentType]

    queue = Queue(
        actor_options={
            "num_cpus": 0,
            **_force_on_current_node(),
            "max_restarts": -1,
            "max_task_retries": -1,
        }
    )
    # name: The globally unique name for the actor, which can be used
    # to retrieve the actor via ray.get_actor(name) as long as the
    # actor is still alive.
    # instantiate remote actor
    actor_handle: ActorHandle = logger_class.remote(
        logdir="./",  # pyright: ignore[reportCallIssue]  # expects pos only
        queue=queue,
        exclude=[],
        to_config=[],
        **wandb_init_kwargs,
    )
    return actor_handle, queue


def upload_run(run_id, run_dir="./"):
    # sort because there are multiple
    path = sorted(Path(run_dir).parent.parent.glob("*run-*" + run_id))[-1]
    print("Syncing run 1", run_id, "from path", path)
    subprocess.run(["wandb", "sync", path, "--append"], check=True)


def make_history_artifact_name(run: wandb.sdk.wandb_run.Run | RunData, version=0) -> str:
    if isinstance(version, int):
        return f"{run.entity}/{run.project}/run-{run.id}-history:v{version}"
    return f"{run.entity}/{run.project}/run-{run.id}-history:{version}"


def wait_for_artifact(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    monitor: ActorHandle[WandbRunMonitor],
    max_wait_time: int = 300,
    *,
    version=0,
) -> bool:
    history_artifact_name = make_history_artifact_name(
        SimpleNamespace(entity=entity, project=project, id=run_id),
        version=version,  # pyright: ignore[reportArgumentType]
    )
    total_wait_time = 0
    if not api.artifact_exists(history_artifact_name):
        future_thread = monitor.monitor_run_threaded.remote(run_id, version=version, max_wait_time=max_wait_time)
        done, remaining = ray.wait([future_thread])
        while not api.artifact_exists(history_artifact_name) or remaining:
            print(
                f"Artifact does not exist yet, waiting for 5 seconds (total: {total_wait_time}s)... Thread status:",
            )
            done, remaining = ray.wait([future_thread], timeout=5)
            total_wait_time += 5
            if total_wait_time > 294:
                print("Timeout waiting for history artifact to appear.")
                break
        return bool(done)
    return True


from ray.air.integrations.wandb import _QueueItem
from ray.tune.experiment import Trial

RUN_DIR = "./"
RUN1_MODE = "offline"
RUN2_MODE = "online"


@dataclass
class RunData:
    entity: str
    project: str
    id: str
    mode: str = RUN1_MODE


def main():
    api = wandb.Api()
    # Initialize the first run and log some metrics
    run_1_id = "test_" + Trial.generate_id()
    run1 = RunData(entity=ENTITY, project=PROJECT, id=run_1_id, mode=RUN1_MODE)
    actor_handle1, queue = setup_actor(dir=RUN_DIR, id=run_1_id, entity=ENTITY, project=PROJECT, mode=RUN1_MODE)
    future1 = actor_handle1.run.remote()
    global_monitor = RemoteWandbRunMonitor.get_remote_monitor(project=PROJECT)
    for i in range(300):
        queue.put((_QueueItem.RESULT, ({"metric": i},), {"step": i}))
    queue.put((_QueueItem.END, (), {}))
    ray.wait([future1])
    if RUN1_MODE == "offline":
        upload_run(run_1_id, RUN_DIR)
        time.sleep(10)
    print("Finished run 1", run_1_id)
    print("\n\n===================================================\n\n")

    fork_id = run_1_id
    time.sleep(1)
    global_monitor.initialize.remote()
    start = time.time()
    wait_for_artifact(api, entity=ENTITY, project=PROJECT, run_id=run_1_id, monitor=global_monitor, version=0)
    total_wait_time = int(time.time() - start)
    run1_artifact_name = make_history_artifact_name(run1, version=0)
    print(f"Artifact {run1_artifact_name} exists after {total_wait_time}s. Proceeding to fork...")

    print("\n\n===================================================\n\n")
    time.sleep(2)

    run2_id = "test_" + Trial.generate_id()
    run2 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        fork_from=f"{fork_id}?_step=200",
        mode="offline",
    )
    actor_handle2, queue2 = setup_actor(dir=RUN_DIR, id=run2_id, entity=ENTITY, project=PROJECT, mode=RUN2_MODE)

    print("Started run 2, forked from run 1 at step 200", run2.id)
    # Continue logging in the new run
    # For the first few steps, log the metric as is from run1
    # After step 250, start logging the spikey pattern
    func2 = make_increasing_function(start_step=250, factor=1.2)
    range2 = range(201, 300)
    for i in range2:
        if i < 250:
            value = i
        else:
            value = i - func2(i) + (5 * math.sin(i / 3.0))
        data = {"metric": value, "additional_metric": i * 1.1}
        queue2.put((_QueueItem.RESULT, (data,), {"step": i}))
    queue2.put((_QueueItem.END, (), {}))
    future2 = actor_handle2.run.remote()
    ray.wait([future2])
    if RUN2_MODE == "offline":
        upload_run(run2)

    print("\n\n===================================================\n\n")

    # Resume run 1
    func1_2 = make_increasing_function(start_step=300, factor=1.5)
    run1_2 = wandb.init(entity=ENTITY, project=PROJECT, id=run1.id, resume="must", mode="offline")
    actor_handle1_2, queue1_2 = setup_actor(dir=RUN_DIR, id=run1_2.id, entity=ENTITY, project=PROJECT, mode=RUN1_MODE)
    for i in range(300, 400):
        queue1_2.put((_QueueItem.RESULT, ({"metric": i + func1_2(i)},), {"step": i}))
    queue1_2.put((_QueueItem.END, (), {}))
    future1_2 = actor_handle1_2.run.remote()
    ray.wait([future1_2])
    if run1_2.settings.mode == "offline":
        upload_run(run1_2)

    assert run1_2.id == run1.id
    print("Finished run 1 part 2", run1_2.id)

    print("\n\n===================================================\n\n")

    print("Waiting 2 seconds to ensure artifact is ready...")
    time.sleep(0.1)
    artifact_exists = api.artifact_exists(run1_artifact_name)
    print(f"Artifact {run1_artifact_name} exists: {artifact_exists}")
    # Get the version of the history artifact
    history_artifact = api.artifact(run1_artifact_name)
    print("History artifact version:", history_artifact.version)
    print("History artifact id:", history_artifact.id)
    print("History artifact type:", history_artifact.type)
    print("History artifact metadata:", history_artifact.metadata)
    print("History artifact description:", history_artifact.description)
    print("History artifact aliases:", history_artifact.aliases)
    print("History artifact files:", history_artifact.files())
    # if history_artifact.version == "v0":
    # with WandBRunMonitor(entity=run1.entity, project=run1.project) as monitor:

    # try without rechecking history artifact, test if online is compatible - should not
    try:
        run3 = wandb.init(entity=ENTITY, project=PROJECT, fork_from=f"{run1.id}?_step=350", mode="offline")
        initial_success = True
    except Exception as e:
        if "fromStep" not in str(e):
            raise
        print("Got expected error when forking online without history artifact v2:", e)
        wait_for_artifact(api, entity=ENTITY, project=PROJECT, run_id=run1.id, monitor=global_monitor, version=1)
        run3 = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            fork_from=f"{run1.id}?_step=350",
            mode="offline",
        )
        initial_success = False

    print("Started run 3, forked from run 1 at step 350", run3.id, "Had to check for artifact:", not initial_success)
    func3 = make_increasing_function(start_step=350, factor=1.75)
    actor_handle3, queue3 = setup_actor(dir=RUN_DIR, id=run3.id, entity=ENTITY, project=PROJECT, mode=RUN1_MODE)
    for i in range(351, 400):
        queue3.put((_QueueItem.RESULT, ({"metric": i + func1_2(i) + func3(i) + (7 * math.sin(i / 3.0))},), {"step": i}))
    queue3.put((_QueueItem.END, (), {}))
    future3 = actor_handle3.run.remote()
    ray.wait([future3])
    if run3.settings.mode == "offline":
        upload_run(run3)
    wait_for_artifact(api, entity=ENTITY, project=PROJECT, run_id=run3.id, monitor=global_monitor, version=0)
    run4 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        fork_from=f"{run3.id}?_step=375",
        mode="offline",
    )
    func4 = make_increasing_function(start_step=375, factor=-2.5)
    actor_handle4, queue4 = setup_actor(dir=RUN_DIR, id=run4.id, entity=ENTITY, project=PROJECT, mode=RUN1_MODE)
    for i in range(376, 425):
        queue4.put(
            (
                _QueueItem.RESULT,
                ({"metric": i + func1_2(i) + func3(i) + func4(i) + (7 * math.sin(i / 3.0))},),
                {"step": i},
            )
        )
    queue4.put((_QueueItem.END, (), {}))
    future4 = actor_handle4.run.remote()
    ray.wait([future4])
    if run4.settings.mode == "offline":
        upload_run(run4)


def make_increasing_function(start_step: int, factor: float):
    def func(step: int) -> float:
        if step < start_step:
            return 0.0
        return (step - start_step) * factor

    return func


if __name__ == "__main__":
    main()
