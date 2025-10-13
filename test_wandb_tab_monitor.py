from __future__ import annotations

# ruff: noqa: G004
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import dotenv
import ray
import ray.exceptions
from ray.air.integrations.wandb import _QueueItem, _WandbLoggingActor
from ray.air.util.node import _force_on_current_node
from ray.tune.experiment import Trial
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.util.queue import Queue

import wandb
import wandb.sdk
from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login import logger as session_logger
from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
from ray_utilities.callbacks.tuner._adv_wandb_logging_actor import _WandbLoggingActorWithArtifactSupport

if TYPE_CHECKING:
    from ray._private.worker import RemoteFunction3
    from ray.actor import ActorHandle


RUN_DIR = Path("./").resolve()
RUN1_MODE = "offline"
RUN2_MODE = "offline"
RUN1_2_MODE = "offline"
RUN3_MODE = "offline"
RUN4_MODE = "offline"


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
session_logger.setLevel(logging.DEBUG)

ENTITY = "daraan"
PROJECT = "dev-workspace"

dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())

import wandb.errors


class TestLoggingActor(_WandbLoggingActorWithArtifactSupport):
    def run(self, retries=0):
        global_monitor = WandbRunMonitor.get_remote_monitor(project=self.kwargs["project"])
        try:
            _WandbLoggingActor.run(self)
        except wandb.errors.CommError as e:
            logger.exception("WandB Communication Error in logging actor:")
            raise  # for now latter handles restart
            # if "fromStep" in str(e) and FORK_FROM in self.kwargs:
            #    ...


def setup_actor(**wandb_init_kwargs) -> tuple[ActorHandle[_WandbLoggingActorWithArtifactSupport], ray.ObjectRef, Queue]:
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
        # _WandbLoggingActorWithArtifactSupport
        TestLoggingActor,
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
    run_future = actor_handle.run.remote()
    try:
        ray.get(run_future, timeout=3)  # wait a bit to ensure actor is started
    except ray.exceptions.GetTimeoutError:
        # did not timeout then while loop is likely running
        # Still could be in a 90s wandb.init wait
        pass
    except TimeoutError:
        raise
    return actor_handle, run_future, queue


def upload_run(run_id, run_dir: str | Path = "./", *, upload_all=True) -> None:
    # sort because there are multiple
    # Note: a wandb.Run.dir is at a different location
    paths = sorted(Path(run_dir, "wandb").glob("*run-*" + run_id))
    if not paths:
        raise ValueError(f"No wandb run with id {run_id} found in {Path(run_dir, 'wandb')}")
    for idx in [-1] if not upload_all else range(len(paths)):
        path = paths[idx]
        pidx = paths.index(path)
        print("Syncing run", run_id, "from path", path, "part", pidx + 1, "of", len(paths))
        subprocess.run(["wandb", "sync", path, "--append"], check=True)


def make_history_artifact_name(run: wandb.sdk.wandb_run.Run | RunData, version: int | str = 0) -> str:
    if isinstance(version, int):
        return f"{run.entity}/{run.project}/run-{run.id}-history:v{version}"
    return f"{run.entity}/{run.project}/run-{run.id}-history:{version}"


def test_concurrency(
    run_id1: str,
    run_id2: str,
    monitor: ActorHandle[WandbRunMonitor],
    max_wait_time: int = 40,
) -> None:
    future1 = monitor.monitor_run.remote(run_id1, version=0, max_wait_time=max_wait_time // 2)
    future2 = monitor.monitor_run.remote(run_id2, version=0, max_wait_time=max_wait_time // 2)
    done, remaining = ray.wait([future1, future2], timeout=5)
    total_wait_time = 0
    while remaining:
        print(f"{len(done)}/2 Futures not done yet, waiting for 5 seconds (total: {total_wait_time}s)...")
        done, remaining = ray.wait(remaining, timeout=5)
        total_wait_time += 5
        if total_wait_time > max_wait_time:
            print("Timeout waiting for futures to complete.")
            break
    if len(done) == 2:
        results = ray.get(done, timeout=2)
        print("Futures completed with results:", results)
    else:
        print("No futures completed.")


def wait_for_artifact_tab_only(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    monitor: ActorHandle[WandbRunMonitor],
    max_wait_time: int = 90,
    *,
    version: int | str = 0,
    open_page: bool = True,
) -> bool:
    history_artifact_name = make_history_artifact_name(
        SimpleNamespace(entity=entity, project=project, id=run_id),  # pyright: ignore[reportArgumentType]
        version=version,
    )
    total_wait_time = 0
    if not api.artifact_exists(history_artifact_name):
        # future_thread = monitor.monitor_run.remote(run_id, version=version, max_wait_time=max_wait_time)
        if open_page:
            page_future = monitor.visit_run_page.remote(run_id)
            # wait for page to be loaded
            ray.get(page_future, timeout=60)
        # else page should be handled outside
        while not api.artifact_exists(history_artifact_name):
            print(
                f"Artifact does not exist yet, waiting for 5 seconds (total: {total_wait_time}s)...",
            )
            time.sleep(5)
            # done, remaining = ray.wait([future_thread], timeout=1)
            total_wait_time += 6
            if total_wait_time > max_wait_time - 5.2:
                print("Timeout waiting for history artifact to appear.")
            if total_wait_time > max_wait_time:
                break
        done = not api.artifact_exists(history_artifact_name)
        if done:
            print("Artifact exists after", total_wait_time, "s")
        return bool(done)
    print("Artifact exists after", total_wait_time, "s")
    return True


def wait_for_future(future: ray.ObjectRef, timeout: int = 300) -> bool:
    total_wait_time = 0
    done, remaining = ray.wait([future], timeout=5)
    while remaining:
        print(f"Future not done yet, waiting for 5 seconds (total: {total_wait_time}s)...")
        done, remaining = ray.wait([future], timeout=5)
        total_wait_time += 5
        if total_wait_time > timeout:
            print("Timeout waiting for future to complete.")
            break
    if done:
        try:
            result = ray.get(done[0], timeout=2)
        except ray.exceptions.ActorDiedError as e:
            print("Actor died after completing future:", e)
        else:
            print("Future completed with result:", result)
        return True
    return False


@dataclass
class RunData:
    entity: str
    project: str
    id: str
    mode: str = RUN1_MODE


def main():
    assert RUN1_MODE == "offline"
    assert RUN3_MODE == "offline"
    api = wandb.Api()
    # Initialize the first run and log some metrics
    run1_id = "test_1_" + Trial.generate_id()
    run1 = RunData(entity=ENTITY, project=PROJECT, id=run1_id, mode=RUN1_MODE)
    actor_handle1, future1, queue1 = setup_actor(
        dir=RUN_DIR, id=run1_id, entity=ENTITY, project=PROJECT, mode=RUN1_MODE
    )
    for i in range(300):
        queue1.put((_QueueItem.RESULT, {"metric": i, TRAINING_ITERATION: i}))
    queue1.put((_QueueItem.END, None))
    global_monitor = WandbRunMonitor.get_remote_monitor(project=PROJECT)
    global_monitor.initialize.remote()

    print("Finished run 1", run1_id)
    print("\n\n===================================================\n\n")

    run1_artifact_name = make_history_artifact_name(run1, version="latest")

    print("\n\n===================================================\n\n")

    run2_id = "test_2_" + Trial.generate_id()
    actor_handle2, future2, queue2 = setup_actor(
        dir=RUN_DIR,
        id=run2_id,
        entity=ENTITY,
        project=PROJECT,
        mode=RUN2_MODE,
        fork_from=f"{run1_id}?_step=200",
    )

    print("Started run 2, forked from run 1 at step 200", run2_id)
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
        queue2.put((_QueueItem.RESULT, (data | {TRAINING_ITERATION: i})))
    queue2.put((_QueueItem.END, None))

    print("\n\n===================================================\n\n")

    # Resume run 1
    func1_2 = make_increasing_function(start_step=300, factor=1.5)
    run1_2_id = run1_id
    actor_handle1_2, future1_2, queue1_2 = setup_actor(
        dir=RUN_DIR, id=run1_2_id, entity=ENTITY, project=PROJECT, mode=RUN1_2_MODE
    )
    for i in range(300, 400):
        queue1_2.put((_QueueItem.RESULT, {"metric": i + func1_2(i), TRAINING_ITERATION: i}))
    queue1_2.put((_QueueItem.END, None))

    assert run1_2_id == run1.id
    print("Finished run 1 part 2", run1_2_id)

    print("\n\n===================================================\n\n")

    print("Waiting 2 seconds to ensure artifact is ready...")
    time.sleep(0.1)
    try:
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
    except Exception as e:
        print("Error accessing artifact:", e)

    print("\n\n======================== RUN 3 =======================\n\n")

    # try without rechecking history artifact, test if online is compatible - should not
    run3_id = "test_3_" + Trial.generate_id()

    try:
        _actor_handle3, future3, queue3 = setup_actor(
            dir=RUN_DIR, id=run3_id, entity=ENTITY, project=PROJECT, mode=RUN3_MODE, fork_from=f"{run1_2_id}?_step=350"
        )
        try:
            ray.get(future3, timeout=3)  # wait a bit to ensure actor does not fail
        except ray.exceptions.GetTimeoutError:
            # wandb.init did not fail, but could still be trying for 90s
            pass
        initial_success = True
    except Exception as e:
        logger.error("Error setting up actor for run 3: %s", e)
        if not isinstance(e, wandb.errors.CommError) or "fromStep" not in str(e):
            raise
        print("Got expected error when forking online without history artifact v2:", e)
        exists = wait_for_artifact_tab_only(
            api, entity=ENTITY, project=PROJECT, run_id=run1_id, monitor=global_monitor, version=1, max_wait_time=180
        )
        if exists:
            print("v1 artifact exist now")
        else:
            print("v1 artifact does not exist still trying to setup run...")
        _actor_handle3, future3, queue3 = setup_actor(
            dir=RUN_DIR, id=run3_id, entity=ENTITY, project=PROJECT, mode=RUN3_MODE, fork_from=f"{run1_id}?_step=350"
        )
        try:
            ray.get(
                future3, timeout=5
            )  # wait a bit to ensure actor does not fail, but as infinite loop cannot wait long
        except ray.exceptions.GetTimeoutError:
            pass
        initial_success = False

    print("Started run 3, forked from run 1 at step 350", run3_id, "Had to check for artifact:", not initial_success)
    func3 = make_increasing_function(start_step=350, factor=1.75)
    for i in range(351, 400):
        queue3.put(
            (_QueueItem.RESULT, {"metric": i + func1_2(i) + func3(i) + (7 * math.sin(i / 3.0)), TRAINING_ITERATION: i})
        )
    queue3.put((_QueueItem.END, None))

    print("\n\n===================== Run 4 =====================\n\n")

    run4_id = "test_4_" + Trial.generate_id()

    _actor_handle4, future4, queue4 = setup_actor(
        dir=RUN_DIR,
        id=run4_id,
        entity=ENTITY,
        project=PROJECT,
        mode=RUN4_MODE,
        fork_from=f"{run3_id}?_step=375",
    )

    func4 = make_increasing_function(start_step=375, factor=-2.5)
    for i in range(376, 425):
        queue4.put(
            (
                _QueueItem.RESULT,
                {"metric": i + func1_2(i) + func3(i) + func4(i) + (7 * math.sin(i / 3.0)), TRAINING_ITERATION: i},
            )
        )
    queue4.put((_QueueItem.END, None))

    # Batch uploads and artifact waits after all runs
    print("\n\n==================== Batch Uploads & Artifact Waits ====================\n\n")
    wait_for_future(future1)
    queue1.shutdown()
    if RUN1_MODE == "offline":
        # TODO: # CRITICAL this uploads part2
        upload_run(run1_id, RUN_DIR)
        time.sleep(10)
    visit_page1 = global_monitor.visit_run_page.remote(run1_id)
    ray.get(visit_page1, timeout=40)

    wait_for_artifact_tab_only(
        api, entity=ENTITY, project=PROJECT, run_id=run1_id, monitor=global_monitor, version="latest", open_page=False
    )
    wait_for_future(future2)
    queue2.shutdown()
    time.sleep(5)
    if RUN2_MODE == "offline":
        upload_run(run2_id, RUN_DIR)
    visit_page2 = global_monitor.visit_run_page.remote(run2_id)
    wait_for_future(future1_2)
    queue1_2.shutdown()
    if RUN1_2_MODE == "offline":
        upload_run(run1_2_id, RUN_DIR)
    visit_page1_2 = global_monitor.visit_run_page.remote(run1_2_id)
    ray.get(visit_page1_2, timeout=40)
    wait_for_artifact_tab_only(
        api, entity=ENTITY, project=PROJECT, run_id=run1_2_id, monitor=global_monitor, version="latest", open_page=False
    )
    wait_for_future(future3)
    queue3.shutdown()
    time.sleep(5)

    if RUN3_MODE == "offline":
        upload_run(run3_id, RUN_DIR)
    visit_page3 = global_monitor.visit_run_page.remote(run3_id)
    ray.get(visit_page3, timeout=40)
    wait_for_artifact_tab_only(
        api, entity=ENTITY, project=PROJECT, run_id=run3_id, monitor=global_monitor, version="latest", open_page=False
    )
    wait_for_future(future4)
    queue4.shutdown()
    time.sleep(5)

    if RUN4_MODE == "offline":
        upload_run(run4_id, RUN_DIR)
    # wait_for_artifact_tab_only(api, entity=ENTITY, project=PROJECT, run_id=run4_id, monitor=global_monitor, version="latest", open_page=False)

    print("Finished run 4", run4_id)
    print("Cleaning up monitor...")
    cleanup = global_monitor.cleanup.remote()
    done = wait_for_future(cleanup, timeout=20)
    if done:
        print("Monitor cleaned up successfully.")
    else:
        print("Monitor cleanup did not complete in time, killing...")
        ray.get(cleanup)  # check for errors
    terminate_future = global_monitor.__ray_terminate__.remote()
    try:
        monitor_done = wait_for_future(terminate_future, timeout=20)
    except ray.exceptions.ActorDiedError:
        # actor already died
        monitor_done = True
    if not monitor_done:
        ray.get(terminate_future, timeout=2)
        print("Monitor did not terminate in time, killing...")
        ray.kill(global_monitor, no_restart=True)


def make_increasing_function(start_step: int, factor: float):
    def func(step: int) -> float:
        if step < start_step:
            return 0.0
        return (step - start_step) * factor

    return func


if __name__ == "__main__":
    main()
