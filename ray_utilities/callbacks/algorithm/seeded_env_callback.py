from __future__ import annotations

import logging
from inspect import isclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Optional, Sequence, overload

import numpy as np
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from ray_utilities.misc import get_current_step

try:
    from ray.rllib.callbacks.callbacks import RLlibCallback
except ImportError:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks as RLlibCallback

from ray.rllib.env.env_context import EnvContext

from ray_utilities.constants import ENVIRONMENT_RESULTS, RAY_METRICS_V2, SEED, SEEDS

try:
    from ray.tune.callback import _CallbackMeta
except ImportError:
    from abc import ABCMeta as _CallbackMeta  # in case meta is removed in future versions


if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from typing_extensions import TypeIs


NUM_ENV_RUNNERS_0_1_EQUAL = True
FIX_EVAL_SEED = True
"""If True, this is closer to original EnvRunner behavior, but each evaluation will use the same seeds."""


def _is_async(env: gym.Env | Any) -> TypeIs[gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv]:
    """Check if the environment is an async vectorized environment."""
    # NOTE expects unwrapped env, currently RLlib passed unpacked envs into the callbacks
    return hasattr(env, "set_attr")


def _to_tuple(seq):
    if isinstance(seq, Sequence) and not isinstance(seq, str):
        return tuple(_to_tuple(item) for item in seq)
    return seq


logger = logging.getLogger(__name__)
_NestedIntSequence = tuple["int | _NestedIntSequence", ...]
EnvSeedType = int | None | _NestedIntSequence


class _SeededEnvCallbackMeta(_CallbackMeta):  # pyright: ignore[reportGeneralTypeIssues]  # base is union type
    env_seed: ClassVar[EnvSeedType] = 0

    def __eq__(cls, value):  # pyright: ignore[reportSelfClsParameterName]
        if not isclass(value):
            return False
        # Subclass check does not work here, will cause recursion
        if SeedEnvsCallbackBase in (b for b in value.mro() if not isinstance(b, _SeededEnvCallbackMeta)):
            real_base = next(b for b in cls.mro() if not isinstance(b, _SeededEnvCallbackMeta))
            other_real_base = next(b for b in value.mro() if not isinstance(b, _SeededEnvCallbackMeta))
            if real_base is not other_real_base:
                return False
            return cls.env_seed == value.env_seed
        return False

    def __hash__(cls):  # pyright: ignore[reportSelfClsParameterName]
        if isinstance(cls.env_seed, Sequence):
            return hash(RLlibCallback) + hash(_to_tuple(cls.env_seed)) + hash(cls.__name__)
        return hash(RLlibCallback) + hash(cls.env_seed) + hash(cls.__name__)

    def __repr__(cls):  # pyright: ignore[reportSelfClsParameterName]
        return f"<class {cls.__name__} env_seed={cls.env_seed}>"


class SeedEnvsCallbackBase(RLlibCallback):
    """Base class for environment seeding callbacks.

    Handles common functionality for seeding environments in different ways.
    Subclasses should implement the seed_environment method.
    """

    env_seed: ClassVar[int | None | Sequence[int]] = 0
    """A common seed that is used for all workers and vector indices.

    If None, the environment will not be seeded. Making this callback a no-op.
    """

    __logged_env_seed_none = False

    def __call__(self, **kwargs):
        """Instance is used as a callback."""
        return self.on_environment_created(**kwargs)

    def __init__(self, **kwargs):  # treat like a callback function
        self._env_reset_count = 0
        self._env_start_count = 0
        self._episode_reset_count = 0
        self._episode_reset_count_on_create = 0
        if "env_context" in kwargs:  # Instance called on_environment_created
            self.on_environment_created(**kwargs)

    def _get_worker_info(self, env_context: EnvContext) -> tuple[int, str]:
        """Get worker index and suffix for logging.

        Returns:
            Tuple of (worker_index, suffix) for logging purposes.
        """
        # Trick to make num_env_runners=0 and num_env_runners=1 equal:
        if NUM_ENV_RUNNERS_0_1_EQUAL and (
            env_context.worker_index == 0 and env_context.num_workers == 0 and env_context.vector_index == 0
        ):
            worker_index = 1
            suffix = " (changed worker_index from 0/0 to 1 to be equal to num_env_runners=1)"
        else:
            worker_index = env_context.worker_index
            suffix = ""
        return worker_index, suffix

    def on_environment_created(
        self,
        *,
        env_runner: EnvRunner,  # noqa: ARG002
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.vector.AsyncVectorEnv | gym.vector.VectorEnv | gym.Env | gym.vector.SyncVectorEnv,
        env_context: EnvContext,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Callback run when a new environment object has been created.

        Note: This only applies to the new API stack. The env used is usually a
        gym.Env (or more specifically a gym.vector.Env).

        Args:
            env_runner: Reference to the current EnvRunner instance.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics after environment creation.
            env: The environment object that has been created on `env_runner`. This is
                usually a gym.Env (or a gym.vector.Env) object.
            env_context: The `EnvContext` object that has been passed to the
                `gym.make()` call as kwargs (and to the gym.Env as `config`). It should
                have all the config key/value pairs in it as well as the
                EnvContext-typical properties: `worker_index`, `num_workers`, and
                `remote`.
            kwargs: Forward compatibility placeholder.

        Note:
            This callback sets the EnvRunner's seed to `None`. This changes how env.reset works.
            In vanilla RLlib the reset is *always* to the same key of the EnvRunner, i.e. during evaluations
            the same initial state is used. This is not the case for this callback.
        """
        env_seed = self.env_seed
        if env_seed is None:
            if not self.__logged_env_seed_none:
                logger.debug("Environment not seeded, env_seed is None. Callback is deactivated.")
                self.__logged_env_seed_none = True
            return
        if env_context.recreated_worker:
            # Worker restart, potentially add flag to seed
            logger.warning("Recreated worker detected. Will be seeded with initial seed, potentially change seed.")

        worker_index, _ = self._get_worker_info(env_context)
        seed_sequence = np.random.SeedSequence(
            env_seed,
            spawn_key=(worker_index, env_context.vector_index, env_runner.config.in_evaluation),
        )

        # Delegate to subclass implementation
        self.seed_environment(seed_sequence, env, env_context, env_runner, metrics_logger)

        # NOTE: Need to set env_runner._seed to None for the custom seeds to be used.
        if env_runner.config.in_evaluation and FIX_EVAL_SEED:
            rng = np.random.default_rng(seed_sequence)
            env_runner._seed = rng.integers(0, 2**31 - 1, size=env.num_envs if _is_async(env) else 1).tolist()
        else:
            env_runner._seed = None
        logger.debug("Setting EnvRunner seed to None, to use seed of %s", type(self).__name__)

    def seed_environment(
        self,
        seed_sequence: np.random.SeedSequence,
        env: gym.vector.AsyncVectorEnv | gym.vector.VectorEnv | gym.Env | gym.vector.SyncVectorEnv,
        env_context: EnvContext,
        env_runner: EnvRunner,
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> None:
        """Seed the environment using the provided seed sequence.

        This method should be implemented by subclasses to handle specific seeding strategies.

        Args:
            seed_sequence: The numpy SeedSequence to use for seeding.
            env: The environment object to seed.
            env_context: The environment context containing worker information.
            env_runner: Reference to the current EnvRunner instance.
            metrics_logger: Optional metrics logger for recording seed information.
        """
        raise NotImplementedError("Subclasses must implement seed_environment method")

    def __eq__(self, other):
        """Equality check for the callback."""
        return (
            isinstance(other, SeedEnvsCallbackBase)
            and self.env_seed == other.env_seed
            and (
                # verify subclass type equality
                type(self) is type(other) or issubclass(type(self), type(other)) or issubclass(type(other), type(self))
            )
        )

    def __hash__(self) -> int:  # PLW1641: Need an explicit __hash__ when using __eq__
        return hash(type(self)) + hash(self.env_seed)


class ResetSeedEnvsCallback(SeedEnvsCallbackBase):
    """Environment seeding callback using env.reset(seed=...) method.

    This callback seeds environments by calling env.reset() with a specific seed.
    Corresponds to the original SEED_RNG_DIRECTLY=False behavior.
    """

    def seed_environment(
        self,
        seed_sequence: np.random.SeedSequence,
        env: gym.vector.AsyncVectorEnv | gym.vector.VectorEnv | gym.Env | gym.vector.SyncVectorEnv,
        env_context: EnvContext,
        env_runner: EnvRunner,
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> None:
        """Seed the environment using env.reset(seed=...) method."""
        worker_index, suffix = self._get_worker_info(env_context)

        starting_seed = seed_sequence.generate_state(1)
        first_observation, _info = env.reset(seed=int(starting_seed[0]))
        logger.debug(
            "Seeding %s envs with seed=%s - "
            "created from env_seed=%s, worker_index %s/%s, evaluation=%s, vector_index=%s.%s. "
            "First observation (each env):\n%s",
            "training" if not env_runner.config.in_evaluation else "evaluation",
            starting_seed[0],
            self.env_seed,
            worker_index,
            env_context.num_workers,  # not used for seed
            env_runner.config.in_evaluation,
            env_context.vector_index,
            suffix,
            first_observation[:, 0],
        )
        if metrics_logger:
            if RAY_METRICS_V2:
                metrics_logger.log_value(
                    (ENVIRONMENT_RESULTS, SEED, "initial_seed"),
                    [int(starting_seed[0])],  # assure int and not numpy int
                    clear_on_reduce=False,
                    reduce="item",
                    window=(env_context.num_workers or 1),
                )
            else:
                # Old interface
                metrics_logger.log_value(
                    (ENVIRONMENT_RESULTS, SEED, "initial_seed"),
                    [int(starting_seed[0])],  # assure int and not numpy int
                    clear_on_reduce=False,
                    reduce=None,
                    window=(env_context.num_workers or 1),
                )


class DirectRngSeedEnvsCallback(SeedEnvsCallbackBase):
    """Environment seeding callback using direct RNG seeding.

    This callback seeds environments by directly setting their random number generators.
    Corresponds to the original SEED_RNG_DIRECTLY=True behavior.
    """

    def seed_environment(
        self,
        seed_sequence: np.random.SeedSequence,
        env: gym.vector.AsyncVectorEnv | gym.vector.VectorEnv | gym.Env | gym.vector.SyncVectorEnv,
        env_context: EnvContext,
        env_runner: EnvRunner,
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> None:
        """Seed the environment by directly setting random number generators."""
        worker_index, suffix = self._get_worker_info(env_context)

        log_seeds = seed_sequence.generate_state(env.num_envs if _is_async(env) else 1)
        rng = np.random.default_rng(seed_sequence)
        rngs = rng.spawn(env.num_envs if _is_async(env) else 1)
        logger.debug(
            "Seeding envs with seed=%s - "
            "created from env_seed=%s, worker_index %s/%s, evaluation=%s, vector_index=%s.%s",
            seed_sequence,
            self.env_seed,
            worker_index,
            env_context.num_workers,  # not used for seed
            env_runner.config.in_evaluation,
            env_context.vector_index,
            suffix,
        )
        # Set random generators for the environments
        if _is_async(env=env):
            env.set_attr("np_random", rngs)
        else:
            env.np_random = rngs[0]

        if metrics_logger:
            # HACK: Set clear_on_reduce=True and remove window again when https://github.com/ray-project/ray/issues/54324 is solved  # noqa: E501
            metrics_logger.log_value(
                (ENVIRONMENT_RESULTS, SEEDS, "seed_sequence"),
                list(map(int, log_seeds.tolist())),  # assure int and not numpy int
                clear_on_reduce=False,
                reduce=None,
                # HACK 2: clear_on_reduce=True is forced when no window is provided
                window=len(log_seeds) * (env_context.num_workers or 1),  # remove when bug is fixed
            )


class SeedEnvsCallback(ResetSeedEnvsCallback):
    """Backward-compatible environment seeding callback.

    This class maintains backward compatibility by inheriting from ResetSeedEnvsCallback,
    which provides the original default behavior (equivalent to SEED_RNG_DIRECTLY=False).

    Use make_seeded_env_callback(None) for pure randomness.
    Use make_seeded_env_callback(fixed_seed) to create reproducible runs.
    make_seeded_env_callback(0) is equivalent to using this class directly.
    """


def _env_runner_get_context(env_runner: EnvRunner | SingleAgentEnvRunner, vector_index: int = 0) -> EnvContext:
    """Get the EnvContext from the EnvRunner."""
    return EnvContext(
        env_config=env_runner.config.env_config,
        worker_index=env_runner.worker_index,
        num_workers=env_runner.num_workers,
        vector_index=vector_index,  # old API?
        # old API
        # remote=getattr(env_runner.config, "remote_worker_envs", False),
        # recreated_worker=getattr(env_runner, "recreated_worker", False),
    )


class AlwaysSeedEvaluationEnvsCallback(SeedEnvsCallback):
    """
    For reproducible and comparable environment results on evaluation.

    This class extends :class:`SeedEnvsCallback` which seeds the initial environment reset
    to also reset and re-seed the environment on every evaluation start.
    For comparable results the current_step of the algorithm is included in the seed sequence.

    Note:
        Only supports evaluation through :meth:`on_evaluate_start`.
        For training a reset is not easily portable and guaranteed to
        work as the callback logic is different.
    """

    def _reset_env_on_evaluate(self, env_runner: EnvRunner | SingleAgentEnvRunner, current_step: int):
        """
        Reset and reseed the environment on the EnvRunner in the same way as :class:`SeedEnvsCallback`

        This method is meant to be used on the callback that the EnvRunner.config holds, to
        minimize object storage usage:

        Example:
            algorithm.eval_env_runner_group.foreach_env_runner(lambda er: _execute_callback_on_env_runner(er, current_step))

            def _execute_callback_on_env_runner(env_runner: EnvRunner | SingleAgentEnvRunner, current_step: int):
                # find callback on env_runner
                for cb in env_runner._callbacks:  # pyright: ignore[reportAttributeAccessIssue]
                    if isinstance(cb, AlwaysSeedEvaluationEnvsCallback):
                        cb._reset_env_on_evaluate(env_runner, current_step)

        Args:
            env_runner: The :class:`EnvRunner` or :class:`SingleAgentEnvRunner` to reset and reseed.
            current_step: The current training step, used as part of the seed sequence for reproducibility.
        """
        env_context = _env_runner_get_context(env_runner)
        seed_sequence: np.random.SeedSequence = np.random.SeedSequence(
            self.env_seed,
            spawn_key=(
                env_context.worker_index,
                env_context.vector_index,
                env_runner.config.in_evaluation,
                current_step,
            ),
        )
        logger.debug(
            "Resetting and reseeding evaluation envs at step %s with sequence: %r", current_step, seed_sequence
        )
        env: gym.vector.VectorWrapper | gym.vector.VectorEnv = env_runner.env  # pyright: ignore[reportAssignmentType]
        # In the super class we do not reset the env seed, need to set the runner's seed to None to apply ours
        env_runner._seed = None
        self.seed_environment(
            seed_sequence=seed_sequence,
            env=env.unwrapped,
            env_context=env_context,
            env_runner=env_runner,
            # do not log seed change for every step
            metrics_logger=None,
        )

    def on_evaluate_start(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        super().on_evaluate_start(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        if algorithm.eval_env_runner_group is None:  # pyright: ignore[reportUnnecessaryComparison]
            return
        # This serializes self :/
        # This callback is on the algorithm but it will also be present with a copy on the env_runner
        # so we call the callback there
        assert metrics_logger is not None
        current_step = get_current_step(metrics_logger)
        algorithm.eval_env_runner_group.foreach_env_runner(lambda er: _execute_callback_on_env_runner(er, current_step))


def _execute_callback_on_env_runner(env_runner: EnvRunner | SingleAgentEnvRunner, current_step: int):
    """
    Execute a present :class:`AlwaysSeedEvaluationEnvsCallback` evaluation environment reset callback
    on a given :class:`EnvRunner`.

    This method is intended to be used with :meth:`EnvRunnerGroup.foreach_env_runner`.
    To minimize object storage wrapping it in a lambda is recommended:

    Example:
        algorithm.eval_env_runner_group.foreach_env_runner(lambda er: _execute_callback_on_env_runner(er, current_step))

    Args:
        env_runner: The :class:`EnvRunner` or :class:`SingleAgentEnvRunner` instance to search for the callback.
        current_step: The current training step, used as part of the seed sequence for environment reseeding.
    """
    # find callback on env_runner
    for cb in env_runner._callbacks:  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(cb, AlwaysSeedEvaluationEnvsCallback):
            cb._reset_env_on_evaluate(env_runner, current_step)


@overload
def make_seeded_env_callback(
    env_seed_: int | None | Sequence[int], *, seed_env_directly: Literal[False] = False, every_eval: Literal[False]
) -> type[SeedEnvsCallbackBase | ResetSeedEnvsCallback]: ...


@overload
def make_seeded_env_callback(
    env_seed_: int | None | Sequence[int],
    *,
    seed_env_directly: Literal[False] = False,
    every_eval: Literal[True] = True,
) -> type[AlwaysSeedEvaluationEnvsCallback]: ...


@overload
def make_seeded_env_callback(
    env_seed_: int | None | Sequence[int], *, seed_env_directly: Literal[True], every_eval: Literal[False]
) -> type[DirectRngSeedEnvsCallback]: ...


@overload
def make_seeded_env_callback(
    env_seed_: int | None | Sequence[int], *, seed_env_directly: Literal[True], every_eval: Literal[True] = True
) -> NoReturn: ...


def make_seeded_env_callback(
    env_seed_: int | None | Sequence[int], *, seed_env_directly: bool = False, every_eval: bool = True
) -> type[SeedEnvsCallbackBase | ResetSeedEnvsCallback | DirectRngSeedEnvsCallback]:
    """Create a callback that seeds the environment.

    Args:
        env_seed_: The seed to use for environment seeding. If None, environments
            will not be seeded and the callback will be deactivated.
        seed_env_directly: If True, use :class:`DirectRngSeedEnvsCallback` for seeding which sets the RNGs directly,
            otherwise use :class:`ResetSeedEnvsCallback` which calls env.reset(seed=...).
            Both use a seed sequence derived from env_seed_, worker index, vector index, and evaluation mode.

    Returns:
        A callback class that can be used for environment seeding.
    """
    if env_seed_ is None:
        logger.info(
            "Using None as env_seed, this will create non-reproducible runs. The callback is deactivated.", stacklevel=2
        )

    if seed_env_directly:
        if every_eval:
            raise NotImplementedError("re-seeding on every eval is not implemented by default for direct env seeding")

        class FixedDirectSeedEnvsCallback(DirectRngSeedEnvsCallback, metaclass=_SeededEnvCallbackMeta):
            env_seed = env_seed_

        return FixedDirectSeedEnvsCallback

    if every_eval:  # default option

        class FixedAlwaysSeedEvaluationEnvsCallback(AlwaysSeedEvaluationEnvsCallback, metaclass=_SeededEnvCallbackMeta):
            env_seed = env_seed_

        return FixedAlwaysSeedEvaluationEnvsCallback

    class FixedSeedResetEnvsCallback(ResetSeedEnvsCallback, metaclass=_SeededEnvCallbackMeta):
        env_seed = env_seed_

    return FixedSeedResetEnvsCallback
