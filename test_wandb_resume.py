import math
import subprocess
import time
from pathlib import Path

import wandb

ENTITY = "daraan"
PROJECT = "dev-workspace"

MODE = "offline"  # "offline" or "online"
UPLOAD_EARLY = False


def main():
    api = wandb.Api()
    # Initialize the first run and log some metrics
    run1 = wandb.init(entity=ENTITY, project=PROJECT, mode=MODE)

    for i in range(200):
        run1.log({"metric": i})
    run1.finish()

    if run1.settings.mode == "offline" and UPLOAD_EARLY:
        path = next(Path(run1.dir).parent.parent.glob("*" + run1.id))
        print("Syncing run 1 from path", path)
        subprocess.run(["wandb", "sync", str(path)], check=True)
    print("Finished run 1", run1.id)
    time.sleep(1.5)  # +1 sec in timestamp for next run
    run2 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        mode=MODE,
        id=run1.id,
        resume="must",
        reinit="create_new",
    )

    for i in range(200, 300):
        run2.log({"metric": i + (i - 200) * 0.5}, step=i)
    run2.finish()

    if run2.settings.mode == "offline" and UPLOAD_EARLY:
        path = sorted(Path(run2.dir).parent.parent.glob("*" + run2.id))[-1]
        print("Syncing run 2 from path", path)
        subprocess.run(["wandb", "sync", str(path), "--append"], check=True)
    print("Finished run 2", run2.id)

    time.sleep(1.5)  # +1 sec in timestamp for next run
    run3 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        resume="must",
        id=run1.id,
        mode=MODE,
        reinit="create_new",
    )

    for i in range(300, 400):
        run3.log({"metric": i + (i - 200) * 0.5 - (i - 300) * 1.5}, step=i)
    run3.finish()

    if run3.settings.mode == "offline" and UPLOAD_EARLY:
        path = sorted(Path(run3.dir).parent.parent.glob("*" + run3.id))[-1]
        print("Syncing run 3 from path", path)
        subprocess.run(["wandb", "sync", str(path), "--append"], check=True)
    print("Finished run 3", run3.id)
    wandb.finish()
    if MODE == "offline" and not UPLOAD_EARLY:
        assert run1.id == run2.id == run3.id
        print("Uploading all 3 parts of", run3.id, "in one go")
        paths = sorted(Path(run3.dir).parent.parent.glob("*" + run3.id))
        breakpoint()
        assert len(paths) == 3
        subprocess.run(["wandb", "sync", *paths, "--append"], check=True)


if __name__ == "__main__":
    main()
