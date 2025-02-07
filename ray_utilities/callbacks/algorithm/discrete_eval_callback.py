from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import EVALUATION_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from interpretable_ddts.agents.rllib_port.discrete_evaluation import (
    discrete_evaluate_on_local_env_runner,
)

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm

    from interpretable_ddts.agents.ddt_ppo_module import DDTModule


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
        if eval_workers is None:
            env_runner = algorithm.env_runner_group.local_env_runner
        elif eval_workers.num_healthy_remote_workers() == 0:
            env_runner = algorithm.eval_env_runner
        else:
            # possibly still use eval_env_runner
            raise NotImplementedError("Parallel discrete evaluation not implemented")
        module: DDTModule = env_runner.module
        if not getattr(module, "CAN_USE_DISCRETE_EVAL", False):
            return
        module.switch_mode(discrete=True)
        assert module.is_discrete
        # new_metrics_logger = MetricsLogger()  # use a new metrics logger to avoid interference
        (
            eval_results,
            env_steps,
            agent_steps,
            batches,
        ) = discrete_evaluate_on_local_env_runner(algorithm, env_runner, metrics_logger)
        module.switch_mode(discrete=False)
        assert module.is_discrete is False
        assert eval_results is None
        if eval_results is None:  # and algorithm.config.enable_env_runner_and_connector_v2:
            eval_results = metrics_logger.reduce((EVALUATION_RESULTS, "discrete"), return_stats_obj=False)
        evaluation_metrics["discrete"] = eval_results
