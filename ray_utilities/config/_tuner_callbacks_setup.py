from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from typing_extensions import TypeVar

from dotenv import load_dotenv
from ray.air.integrations.wandb import WandbLoggerCallback

from ray_utilities.callbacks.tuner import AdvCometLoggerCallback, create_tuner_callbacks

if TYPE_CHECKING:
    from ray.tune import Callback

    from ray.rllib.algorithms import AlgorithmConfig
    from ray_utilities.config.experiment_base import ExperimentSetupBase, DefaultArgumentParser


class _TunerCallbackSetupBase(ABC):
    @abstractmethod
    def create_callbacks(self) -> list[Callback]: ...


__all__ = [
    "TunerCallbackSetup",
]

ConfigType = TypeVar("ConfigType", bound="AlgorithmConfig")
ParserType = TypeVar("ParserType", bound="DefaultArgumentParser", default="DefaultArgumentParser")

logger = logging.getLogger(__name__)


class TunerCallbackSetup(_TunerCallbackSetupBase):
    def __init__(self, *, setup: ExperimentSetupBase[ConfigType, ParserType], extra_tags: Optional[list[str]] = None):
        self._setup = setup
        self._extra_tags = extra_tags

    def get_tags(self) -> list[str]:
        tags = self._setup.create_tags()
        if self._extra_tags:
            tags.extend(self._extra_tags)
        return tags

    def create_wandb_logger(self) -> WandbLoggerCallback:
        """Create wandb logger"""
        args = self._setup.args
        mode: Literal["offline", "disabled", "online"]
        if args.wandb in (False, "disabled"):
            mode = "disabled"
        else:
            mode = args.wandb.split("+")[0]  # pyright: ignore[reportAssignmentType]

        return WandbLoggerCallback(
            project=self._setup.project_name,
            group=self._setup.group_name,  # if not set Tuner name is used
            excludes=["system/*"],
            upload_checkpoints=False,
            save_code=False,  # Code diff
            # For more keywords see: https://docs.wandb.ai/ref/python/init/
            # Log gym
            # https://docs.wandb.ai/guides/integrations/openai-gym/
            monitor_gym=False,
            # Special comment
            notes=args.comment if args.comment else None,
            tags=self.get_tags(),
            mode=mode,
            job_type="train",
            # config_exclude_keys
            # config_include_keys
            # settings advanced wandb.Settings
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
            exclude_metrics=(
                "time_since_restore",
                "iterations_since_restore",
                "timestamp",
                # "training_iteration", #  needed for the callback
            ),
            log_to_other=("comment", "cli_args/comment", "cli_args/test", "cli_args/num_jobs"),
            log_cli_args=True,
            log_pip_packages=True,  # only relevant if log_env_details=False
        )

    @staticmethod
    def _set_comet_api_key() -> bool:
        return load_dotenv(Path("~/.comet_api_key.env").expanduser())

    def create_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = create_tuner_callbacks(render=bool(self._setup.args.render_mode))
        if self._setup.args.wandb:
            callbacks.append(self.create_wandb_logger())
        else:
            logger.info("Not logging to WandB")
        if self._setup.args:
            callbacks.append(self.create_comet_logger())
        else:
            logger.info("Not logging to Comet")
        return callbacks
