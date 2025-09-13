from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ray.rllib.algorithms.algorithm import Algorithm

from ray_utilities.callbacks.algorithm.callback_mixins import BudgetMixin
from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback
from ray_utilities.dynamic_config.dynamic_buffer_update import get_dynamic_evaluation_intervals

if TYPE_CHECKING:
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.algorithms.algorithm import Algorithm


class AdjustConfigWhenBatchSizeChanges(BudgetMixin, DynamicHyperparameterCallback):
    def on_algorithm_init(self, *, algorithm: Algorithm, metrics_logger: MetricsLogger | None = None, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        self._set_budget_on_algorithm_init(algorithm=algorithm, **kwargs)
        self._evaluation_intervals: dict[int, int] = dict(
            zip(
                self._budget["step_sizes"],
                (
                    get_dynamic_evaluation_intervals(
                        self._budget["step_sizes"],
                        eval_freq=algorithm.config.evaluation_interval,  # pyright: ignore[reportOptionalMemberAccess]
                        batch_size=algorithm.config.train_batch_size_per_learner,  # pyright: ignore[reportOptionalMemberAccess]
                        take_root=True,
                    )
                    # 0 for no evaluation
                    if algorithm.config.evaluation_interval  # pyright: ignore[reportOptionalMemberAccess]
                    else [0] * len(self._budget["step_sizes"])
                ),
            )
        )

    def adjust_config_based_on_batch_size(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """
        on_algorithm_init callback

        Large / small batch sizes call for different config values:
        e.g.
        - smaller / larger evaluation_interval
        - broader / narrower smoothing window for metrics
        """
        # NOTE: Can interfere with DynamicEvalCallback when evaluation_interval changes
        self._update_algorithm(algorithm, key="evaluation_interval", value=...)
        # Cannot change window of metric logger when already logged
        self._update_algorithm(
            algorithm=algorithm,
            key="metrics_num_episodes_for_smoothing",
            value=max(30, min(300, int(math.sqrt(algorithm.config.train_batch_size_per_learner)))),  # pyright: ignore[reportOptionalMemberAccess]
        )

    def on_train_result(self, *, algorithm: Algorithm, metrics_logger: MetricsLogger | None = None, **kwargs) -> None:
        self.adjust_config_based_on_batch_size(algorithm=algorithm, **kwargs)
