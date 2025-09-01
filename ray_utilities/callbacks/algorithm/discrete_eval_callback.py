from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, cast

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import EVALUATION_RESULTS

from ray_utilities.constants import EVALUATED_THIS_STEP
from ray_utilities.discrete_evaluation import discrete_evaluate_on_local_env_runner

if TYPE_CHECKING:
    from interpretable_ddts.rllib_port.ddt_ppo_module import DDTModule  # TODO: upstream
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import ResultDict

__all__ = ["DiscreteEvalCallback"]

logger = logging.getLogger(__name__)

DiscreteEvalFunctionType = Callable[
    ["Algorithm", "SingleAgentEnvRunner", "MetricsLogger"],
    tuple["ResultDict | None", int, int, list["SampleBatch | MultiAgentBatch"]],
]


class DiscreteEvalCallback(DefaultCallbacks):
    _warned_once: bool = False

    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,  # noqa: ARG002
    ) -> None:
        env_runner = algorithm.env_runner
        eval_workers = algorithm.eval_env_runner_group
        if eval_workers is None:  # type: ignore[comparison-overlap]
            env_runner = algorithm.env_runner_group.local_env_runner  # type: ignore[attr-defined]
        elif eval_workers.num_healthy_remote_workers() == 0:
            env_runner = algorithm.eval_env_runner
        elif eval_workers.local_env_runner:
            if not self._warned_once:
                logger.info(
                    "Parallel discrete evaluation not implemented. Using one local env_runner for discrete evaluation."
                )
                self._warned_once = True
            env_runner = eval_workers.local_env_runner
        else:
            if not self._warned_once:
                logger.warning("No eval workers available for discrete evaluation. Using local env_runner.")
                self._warned_once = True
            env_runner = algorithm.eval_env_runner
            # possibly still use eval_env_runner
            # raise NotImplementedError("Parallel discrete evaluation not implemented")
        env_runner = cast("SingleAgentEnvRunner", env_runner)
        if metrics_logger is None:
            logger.warning("No metrics logger provided for discrete evaluation")
            metrics_logger = algorithm.metrics
        module: DDTModule = env_runner.module  # type: ignore[assignment]
        evaluation_metrics[EVALUATED_THIS_STEP] = True  # Note: NotRequired key
        if not getattr(module, "CAN_USE_DISCRETE_EVAL", False):
            return
        assert metrics_logger
        module.switch_mode(discrete=True)
        assert module.is_discrete
        (
            discrete_eval_results,
            _env_steps,
            _agent_steps,
            _batches,
        ) = discrete_evaluate_on_local_env_runner(algorithm, env_runner, metrics_logger)
        module.switch_mode(discrete=False)
        assert module.is_discrete is False
        assert discrete_eval_results is None  # new API stack
        if discrete_eval_results is None:  # and algorithm.config.enable_env_runner_and_connector_v2:
            # FIXME: reduce changed in ray version, does not take arguments anympre
            try:
                discrete_eval_results = metrics_logger.reduce((EVALUATION_RESULTS, "discrete"), return_stats_obj=False)  # pyright: ignore[reportCallIssue]
            except TypeError:
                # new ray versions; untested, might reduce too much
                discrete_eval_results = metrics_logger.reduce()[EVALUATION_RESULTS]["discrete"]
        evaluation_metrics["discrete"] = discrete_eval_results
