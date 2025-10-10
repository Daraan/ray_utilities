import time
import wandb
import math


def main():
    run1_data = []
    run2_data = []

    FORK_ONLY = False

    if not FORK_ONLY:
        # Initialize the first run and log some metrics
        run1 = wandb.init(entity="daraan", project="dev-workspace", mode="online")
        for i in range(300):
            run1.log({"metric": i})
            run1_data.append(i)
            if i == 200:
                # create fork in parallel
                print("finishing run 1", run1.id)
                run1.finish()
                time.sleep(120)
                print("starting run 2")
                run2 = wandb.init(
                    entity="daraan",
                    project="dev-workspace",
                    fork_from=f"{run1.id}?_step=199",
                    reinit="create_new",
                    mode="online",
                )
                time.sleep(1)
                print("resuming run 1", run1.id)
                run1 = wandb.init(
                    entity="daraan",
                    project="dev-workspace",
                    id=run1.id,
                    resume="must",
                    mode="online",
                    reinit="create_new",
                )
            if i >= 201:
                if i < 250:
                    value = i
                else:
                    # Introduce the spikey behavior starting from step 250
                    value = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
                data = {"metric": value, "additional_metric": i * 1.1}
                run2.log({"metric": value}, step=i)
                run2.log({"additional_metric": i * 1.1}, step=i)
                run2_data.append(value)
        run1.finish()
        run2.finish()
        print("Finished run 1", run1.id)
        assert run1_data == list(range(300))

    if False:
        fork_id = "ohchysp6"
        # Fork from the first run at a specific step and log the metric starting from step 200
        # run2 = wandb.init(entity="daraan", project="dev-workspace", fork_from=f"{run1.id}?_step=200", reinit="create_new")
        import wandb
        import math

        run1_data = range(300)
        run2_data = []

        run2 = wandb.init(entity="daraan", project="dev-workspace", fork_from="ohchysp6?_step=200")

        print("Started run 2, forked from run 1 at step 200", run2.id)
        # Continue logging in the new run
        # For the first few steps, log the metric as is from run1
        # After step 250, start logging the spikey pattern
        for i in range(200, 300):
            if i < 250:
                value = i
            else:
                # Introduce the spikey behavior starting from step 250
                value = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
            data = {"metric": value, "additional_metric": i * 1.1}
            run2.log({"metric": value}, step=i)
            run2.log({"additional_metric": i * 1.1}, step=i)
            run2_data.append(value)
            # Additionally log the new metric at all steps
        run2.finish()

    import matplotlib.pyplot as plt

    plt.plot(range(300), run1_data, label="Run 1 Metric")
    plt.plot(range(201, 300), run2_data, label="Run 2 Metric (Forked)")
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title("Comparison of Metrics from Run 1 and Forked Run 2")
    plt.legend()
    plt.show()

    # Upload to unknown fork base:

    # Initialize the first run and log some metrics
    run1 = wandb.init(entity="daraan", project="dev-workspace", mode="offline")
    for i in range(300):
        run1.log({"metric": i})
    run1.mark_preempting()
    run1.finish()
    print("Finished run 1", run1.id)
    assert run1_data == list(range(300))

    run2 = wandb.init(
        entity="daraan",
        project="dev-workspace",
        fork_from=f"{run1.id}?_step=200",
        settings=wandb.Settings(init_timeout=300),
        mode="offline",
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
