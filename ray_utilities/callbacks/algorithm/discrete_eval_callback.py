import logging
from typing import TYPE_CHECKING, Optional, cast


from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import EVALUATION_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from interpretable_ddts.agents.rllib_port.discrete_evaluation import discrete_evaluate_on_local_env_runner
from ray_utilities.constants import EVALUATED_THIS_STEP

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from interpretable_ddts.agents.ddt_ppo_module import DDTModule

__all__ = ["DiscreteEvalCallback"]

logger = logging.getLogger(__name__)


class DiscreteEvalCallback(DefaultCallbacks):
    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
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
        else:
            # possibly still use eval_env_runner
            raise NotImplementedError("Parallel discrete evaluation not implemented")
        env_runner = cast("SingleAgentEnvRunner", env_runner)
        if metrics_logger is None:
            logger.warning("No metrics logger provided for discrete evaluation")
            metrics_logger = algorithm.metrics
        module: DDTModule = env_runner.module  # type: ignore[assignment]
        evaluation_metrics[EVALUATED_THIS_STEP] = True  # Note: NotRequired key
        if not getattr(module, "CAN_USE_DISCRETE_EVAL", False):
            return
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
        assert discrete_eval_results is None
        if discrete_eval_results is None:  # and algorithm.config.enable_env_runner_and_connector_v2:
            discrete_eval_results = metrics_logger.reduce((EVALUATION_RESULTS, "discrete"), return_stats_obj=False)
        evaluation_metrics["discrete"] = discrete_eval_results
