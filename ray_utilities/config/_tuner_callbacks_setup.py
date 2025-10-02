from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Optional

from dotenv import load_dotenv
from typing_extensions import TypeVar

from ray_utilities.callbacks.tuner import AdvCometLoggerCallback, create_tuner_callbacks
from ray_utilities.callbacks.tuner._log_result_grouping import exclude_results
from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback

try:
    import wandb
except ImportError:
    pass

if TYPE_CHECKING:
    from ray.air.integrations.wandb import WandbLoggerCallback
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune import Callback

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup import ExperimentSetupBase


class _TunerCallbackSetupBase(ABC):
    @abstractmethod
    def create_callbacks(self, *args, **kwargs) -> list[Callback]:
        """Create a list of initialized callbacks for the tuner."""


__all__ = [
    "TunerCallbackSetup",
]

ConfigTypeT = TypeVar("ConfigTypeT", bound="AlgorithmConfig")
ParserTypeT = TypeVar("ParserTypeT", bound="DefaultArgumentParser")
_AlgorithmType_co = TypeVar("_AlgorithmType_co", bound="Algorithm", covariant=True)

logger = logging.getLogger(__name__)


class TunerCallbackSetup(_TunerCallbackSetupBase):
    """
    Setup to create callbacks for the tuner.

    Methods:
        - create_callbacks: Create a list of callbacks for the tuner.
           In turn calls:
            - create_wandb_logger: Create a WandB logger callback.
            - create_comet_logger: Create a Comet logger callback.
    """

    EXCLUDE_METRICS: ClassVar[list[str]] = [
        *exclude_results,
        # "time_since_restore",
        # "iterations_since_restore",
        # "timestamp",  # autofilled
        # "num_agent_steps_sampled_lifetime",
        # "learners", # NEW: filtered by log_stats
        # "timers",
        # "fault_tolerance",
        # "training_iteration", #  needed for the callback
    ]

    def __init__(
        self,
        *,
        setup: ExperimentSetupBase[ParserTypeT, ConfigTypeT, _AlgorithmType_co],
        extra_tags: Optional[list[str]] = None,
    ):
        self._setup = setup
        self._extra_tags = extra_tags

    def get_tags(self) -> list[str]:
        tags = self._setup.create_tags()
        if self._extra_tags:
            tags.extend(self._extra_tags)
        return tags

    def create_wandb_logger(self) -> WandbLoggerCallback:
        """
        Create wandb logger

        For more keywords see: https://docs.wandb.ai/ref/python/init/
        """
        args = self._setup.args
        mode: Literal["offline", "disabled", "online"]
        if args.wandb in (False, "disabled"):
            mode = "disabled"
        else:
            mode = args.wandb.split("+")[0]  # pyright: ignore[reportAssignmentType]

        # Note: Settings are overwritten by the keywords provided below (or by ray)
        try:
            adv_settings = wandb.Settings(  # pyright: ignore[reportPossiblyUnboundVariable]
                disable_code=args.test,
                disable_git=args.test,
                # Internal setting
                # Disable system metrics collection.
                x_disable_stats=args.test,
                # Disable check for latest version of wandb, from PyPI.
                # x_disable_update_check=not args.test,  # not avail in latest version
            )
        except NameError as e:
            if "name 'wandb' is not defined" not in str(e):
                raise
            warnings.warn(
                "wandb is not installed, cannot create WandbLoggerCallback. This will likely result in a RuntimeError",
                stacklevel=2,
            )
            adv_settings = None
        except Exception:
            logger.exception("Error creating wandb.Settings")
            adv_settings = None
        # TODO: make use of resume/resume_from/fork_from when using from_checkpoint
        # see: https://docs.wandb.ai/ref/python/sdk/functions/init/ format: {id}?_step={step}
        return AdvWandbLoggerCallback(
            project=self._setup.project_name,
            group=self._setup.group_name,  # if not set trainable name is used
            excludes=[
                *self.EXCLUDE_METRICS,
                # "fault_tolerance",
            ],
            upload_checkpoints=False,
            save_code=False,  # Code diff
            # For more keywords see: https://docs.wandb.ai/ref/python/init/
            # Log gym
            # https://docs.wandb.ai/guides/integrations/openai-gym/
            monitor_gym=False,
            # Special comment
            notes=args.comment or None,
            tags=self.get_tags(),
            mode=mode,
            job_type="train",
            log_config=False,  # Log "config" key of results; useful if params change. Defaults to False.
            # settings advanced wandb.Settings
            settings=adv_settings,
            upload_offline_experiments=self._setup.args.wandb and "upload" in self._setup.args.wandb,
        )

    def create_comet_logger(
        self,
        *,
        disabled: Optional[bool] = None,
        api_key: Optional[str] = None,
    ) -> Callback:
        args = self._setup.args
        env_var_set = self._set_comet_api_key()
        if not env_var_set and not api_key:
            raise ValueError("Comet API not loadable check _set_comet_api_key or provide api_key")
        use_comet_offline: bool = getattr(
            self._setup.args,
            "use_comet_offline",
            self._setup.args.comet and self._setup.args.comet.lower().startswith("offline"),
        )

        return AdvCometLoggerCallback(
            # new key
            upload_offline_experiments=self._setup.args.comet and "upload" in self._setup.args.comet,
            api_key=api_key,
            disabled=not args.comet and args.test if disabled is None else disabled,
            online=not use_comet_offline,  # do not upload
            workspace=self._setup.project_name,
            project_name=self._setup.group_name,  # "general" for Uncategorized Experiments
            save_checkpoints=False,
            tags=self.get_tags(),
            # Other keywords see: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
            auto_metric_step_rate=10,  # How often batch metrics are logged. Default 10
            auto_histogram_epoch_rate=1,  # How often histograms are logged. Default 1
            parse_args=False,
            log_git_metadata=not args.test,  # disabled by rllib; might cause throttling -> needed for Reproduce button
            log_git_patch=False,
            log_graph=False,  # computation graph, Default True
            log_code=False,  # Default True; use if not using git_metadata
            log_env_details=not args.test,
            # Subkeys of env details:
            log_env_network=False,
            log_env_disk=False,
            log_env_gpu=args.num_jobs <= 5 and args.gpu,
            log_env_host=False,
            log_env_cpu=args.num_jobs <= 5,
            # ---
            auto_log_co2=False,  # needs codecarbon
            auto_histogram_weight_logging=False,  # Default False
            auto_histogram_gradient_logging=False,  # Default False
            auto_histogram_activation_logging=False,  # Default False
            # Custom keywords of Adv Callback
            exclude_metrics=self.EXCLUDE_METRICS,
            log_to_other=(
                "comment",
                "cli_args/comment",
                "cli_args/test",
                "cli_args/num_jobs",
                "evaluation/env_runners/environments/seeds",
                "env_runners/environments/seeds",
            ),
            log_cli_args=True,
            log_pip_packages=True,  # only relevant if log_env_details=False
        )

    @staticmethod
    def _set_comet_api_key() -> bool:
        if "COMET_API_KEY" in os.environ:
            return True
        logger.debug("COMET_API_KEY not in environment variables, trying to load from ~/.comet_api_key.env")
        return load_dotenv(Path("~/.comet_api_key.env").expanduser())

    def create_callbacks(self, *, adv_loggers: bool | None = None) -> list[Callback]:
        """
        Create a list of initialized callbacks for the tuner.

        Args:
            adv_loggers: Whether to include advanced variants of the standard CSV, TBX, JSON loggers.
                If ``None``, will be set to ``True`` if :attr:`~DefaultArgumentParser.render_mode` is set in ``args`` of
                the setup.
                Its recommended to use ``True`` when using schedulers working with ``FORK_FROM``.
        """
        if adv_loggers is None:
            adv_loggers = bool(self._setup.args.render_mode)
        callbacks: list[Callback] = create_tuner_callbacks(adv_loggers=adv_loggers)
        if self._setup.args.wandb or self._setup.args.test:
            callbacks.append(self.create_wandb_logger())
            logger.info("Created WanbB logger" if self._setup.args.wandb else "Created WandB logger - for testing")
        else:
            logger.info("Not logging to WandB")
        if self._setup.args.comet or self._setup.args.test:
            callbacks.append(self.create_comet_logger())
            logger.info("Created comet logger" if self._setup.args.comet else "Created comet logger - for testing")
        else:
            logger.info("Not logging to Comet")
        return callbacks
