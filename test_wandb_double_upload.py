import time
import wandb
import math
import pytest


def main():
    # Initialize the first run and log some metrics
    run1 = wandb.init(entity="daraan", project="dev-workspace", mode="online")
    for i in range(201):
        run1.log({"metric": i})
    run1.mark_preempting()
    run1.finish()
    print("Paused run 1", run1.id)
    run1_continued = wandb.init(
        entity="daraan",
        project="dev-workspace",
        id=run1.id,
        resume="must",
        mode="online",
    )
    print("Continued run 1", run1_continued.id)
    for i in range(201, 300):
        run1_continued.log({"metric": i})
    run1_continued.finish()
    assert run1.id == run1_continued.id

    fork_id = run1.id

    print("waiting 20s to fork")
    time.sleep(20)
    print("forking now")
    run2 = wandb.init(
        entity="daraan",
        project="dev-workspace",
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
