import logging
import math
import subprocess
import time
from pathlib import Path
import dotenv

import wandb
import wandb.sdk
from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
from ray_utilities.callbacks._wandb_monitor._wandb_selenium_login import logger as session_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
session_logger.setLevel(logging.DEBUG)

ENTITY = "daraan"
PROJECT = "dev-workspace"

dotenv.load_dotenv(Path("~/.wandb_viewer.env").expanduser())


def upload_run(run: wandb.sdk.wandb_run.Run):
    # sort because there are multiple
    path = sorted(Path(run.dir).parent.parent.glob("*run-*" + run.id))[-1]
    print("Syncing run 1", run.id, "from path", path)
    subprocess.run(["wandb", "sync", path, "--append"], check=True)


def make_history_artifact_name(run: wandb.sdk.wandb_run.Run, version=0) -> str:
    if isinstance(version, int):
        return f"{run.entity}/{run.project}/run-{run.id}-history:v{version}"
    return f"{run.entity}/{run.project}/run-{run.id}-history:{version}"


def wait_for_artifact(
    api: wandb.Api,
    run: wandb.sdk.wandb_run.Run,
    monitor: WandbRunMonitor,
    max_wait_time: int = 300,
    *,
    version=0,
) -> bool:
    history_artifact_name = make_history_artifact_name(run, version=version)
    total_wait_time = 0
    if not api.artifact_exists(history_artifact_name):
        thread = monitor.monitor_run_threaded(run.id, version=version, max_wait_time=max_wait_time)
        while not api.artifact_exists(history_artifact_name) and thread.is_alive():
            print(
                f"Artifact does not exist yet, waiting for 2 seconds (total: {total_wait_time}s)... Thread status:",
                thread.is_alive(),
            )
            time.sleep(5)
            total_wait_time += 5
            if total_wait_time > 294:
                print("Timeout waiting for history artifact to appear. Thread is alive:", thread.is_alive())
                break
        print("Artifact exists now, waiting for thread to finish...")
        start = time.time()
        thread.join()
        end = time.time()
        print("Joined thread after", end - start, "seconds. Thread alive:", thread.is_alive())
        return thread.is_alive()
    return True


def main():
    api = wandb.Api()
    # Initialize the first run and log some metrics
    run1 = wandb.init(entity=ENTITY, project=PROJECT, mode="offline")
    for i in range(300):
        run1.log({"metric": i})
    run1.finish()
    if run1.settings.mode == "offline":
        upload_run(run1)
        time.sleep(10)
    print("Finished run 1", run1.id)
    print("\n\n===================================================\n\n")

    fork_id = run1.id
    time.sleep(1)
    monitor = WandbRunMonitor(entity=run1.entity, project=run1.project)
    monitor.initialize()
    start = time.time()
    wait_for_artifact(api, run1, monitor, version=0)
    total_wait_time = int(time.time() - start)

    print("\n\n===================================================\n\n")
    time.sleep(2)

    run1_artifact_name = make_history_artifact_name(run1, version=0)
    print(f"Artifact {run1_artifact_name} exists after {total_wait_time}s. Proceeding to fork...")
    run2 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        fork_from=f"{fork_id}?_step=200",
        mode="offline",
    )

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
            # Introduce the spikey behavior starting from step 250
            value = i - func2(i) + (5 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        data = {"metric": value, "additional_metric": i * 1.1}
        run2.log(data)
        # Additionally log the new metric at all steps
    run2.finish()
    if run2.settings.mode == "offline":
        upload_run(run2)

    print("\n\n===================================================\n\n")

    # Resume run 1
    func1_2 = make_increasing_function(start_step=300, factor=1.5)
    run1_2 = wandb.init(entity=ENTITY, project=PROJECT, id=run1.id, resume="must", mode="offline")
    for i in range(300, 400):
        run1_2.log({"metric": i + func1_2(i)}, step=i)
    # capture output of wandb.finish to get the correct sync command
    run1_2.finish()
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
    except wandb.errors.CommError as e:
        if "fromStep" not in str(e):
            raise
        print("Got expected error when forking online without history artifact v2:", e)
        wait_for_artifact(api, run1, monitor, version=1)
        run3 = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            fork_from=f"{run1.id}?_step=350",
            mode="offline",
        )
        initial_success = False

    print("Started run 3, forked from run 1 at step 350", run3.id, "Had to check for artifact:", not initial_success)
    func3 = make_increasing_function(start_step=350, factor=1.75)
    for i in range(351, 400):
        run3.log({"metric": i + func1_2(i) + func3(i) + (7 * math.sin(i / 3.0))}, step=i)
    run3.finish()
    if run3.settings.mode == "offline":
        upload_run(run3)
    wait_for_artifact(api, run3, monitor, version=0)
    run4 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        fork_from=f"{run3.id}?_step=375",
        mode="offline",
    )
    func4 = make_increasing_function(start_step=375, factor=-2.5)
    for i in range(376, 425):
        run4.log({"metric": i + func1_2(i) + func3(i) + func4(i) + (7 * math.sin(i / 3.0))}, step=i)
    run4.finish()
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
