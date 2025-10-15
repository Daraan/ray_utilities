"""
WandB's fork feature (https://docs.wandb.ai/guides/runs/forking/) has currently a bug that
forks cannot be started or offline uploaded to runs that not yet have their history artifact
created (entity/project/run-<id>-history). This history is currently only created when the run is viewed
in the web interface.

This submodule provides a workaround that uses a selenium browser to log in to wandb and
view the run page, which triggers the creation of the history artifact. This is done in a
background thread, so that the main process can continue to run and wait for the artifact to appear
"""

from .wandb_run_monitor import WandbRunMonitor

__all__ = ["WandbRunMonitor"]
