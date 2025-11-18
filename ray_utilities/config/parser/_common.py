import logging
from tap import Tap
from typing_extensions import Literal

from ray_utilities.constants import DEFAULT_EVAL_METRIC, EVAL_METRIC_RETURN_MEAN, EVAL_METRIC_RETURN_MEAN_EMA

logger = logging.getLogger(__name__)


class GoalParser(Tap):
    mode: Literal["min", "max"] = "max"
    """One of {min, max}.Determines whether objective is minimizing or maximizing the metric attribute."""

    metric: str = EVAL_METRIC_RETURN_MEAN_EMA
    """
    The metric to be optimized as flat key, e.g. 'evaluation/env_runners/episode_return_mean'.

    Note:
        This attribute can be superseded by subparsers like :class:`PopulationBasedTrainingParser`.
    """

    no_eval_ema: bool = False
    """
    This is a short switch to change the metric attribute from the new:
    :const:`EVAL_METRIC_RETURN_MEAN_EMA`, that is calculated by the :func:`EvalEMAMetricCallback`,
    to the standard :const:`EVAL_METRIC_RETURN_MEAN` which is the default evaluation metric in RLlib.
    """

    def process_args(self) -> None:
        if self.no_eval_ema and self.metric in (
            EVAL_METRIC_RETURN_MEAN_EMA,
            DEFAULT_EVAL_METRIC,
            "DEFAULT_EVAL_METRIC",
        ):
            logger.info("Switching evaluation metric from EMA to standard return mean.")
            self.metric = EVAL_METRIC_RETURN_MEAN
        elif not self.no_eval_ema and self.metric in (
            DEFAULT_EVAL_METRIC,
            "DEFAULT_EVAL_METRIC",
        ):
            logger.info("Switching DEFAULT_EVAL_METRIC from standard to EMA.")
            self.metric = EVAL_METRIC_RETURN_MEAN_EMA
        super().process_args()
        assert self.metric is not None
