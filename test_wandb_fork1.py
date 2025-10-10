import math
import subprocess
import time
from pathlib import Path

import wandb

ENTITY = "daraan"
PROJECT = "dev-workspace"


def main():
    api = wandb.Api()
    # Initialize the first run and log some metrics
    run1 = wandb.init(entity=ENTITY, project=PROJECT, mode="online")
    for i in range(300):
        run1.log({"metric": i})
    run1.finish()
    if run1.settings.mode == "offline":
        path = next(Path(run1.dir).parent.parent.glob("*" + run1.id))
        print("Syncing run 1 from path", path)
        subprocess.run(["wandb", "sync", str(path)], check=True)
    print("Finished run 1", run1.id)

    fork_id = run1.id
    time.sleep(5)

    history_artifact_name = f"{run1.entity}/{run1.project}/run-{run1.id}-history:latest"
    total_wait_time = 0
    while not api.artifact_exists(history_artifact_name):
        print(f"Artifact {history_artifact_name} does not exist after {total_wait_time}s. Waiting for 5 seconds ...")
        time.sleep(5)
        total_wait_time += 5
        try:
            a2 = api.artifact(history_artifact_name, type="wandb-history")
        except wandb.CommError:
            continue

    print(f"Artifact {history_artifact_name} exists after {total_wait_time} seconds. Proceeding to fork...")
    time.sleep(5)
    print("forking now")
    run2 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        fork_from=f"{fork_id}?_step=200",
    )

    print("Started run 2, forked from run 1 at step 200", run2.id)
    # Continue logging in the new run
    # For the first few steps, log the metric as is from run1
    # After step 250, start logging the spikey pattern
    range2 = range(201, 300)
    for i in range2:
        if i < 250:
            value = i
        else:
            # Introduce the spikey behavior starting from step 250
            value = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        data = {"metric": value, "additional_metric": i * 1.1}
        run2.log(data, step=i)
        # Additionally log the new metric at all steps
    run2.finish()


if __name__ == "__main__":
    main()
