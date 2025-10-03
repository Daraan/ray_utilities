"""Algorithm configuration creation utilities for Ray RLlib experiments.

This module provides comprehensive utilities for creating and configuring Ray RLlib
algorithm configurations based on parsed command-line arguments and experiment
specifications. It handles the complex setup of algorithms with proper environment
configuration, callback registration, learner setup, and module specifications.

The main function :func:`create_algorithm_config` serves as a centralized point
for translating experiment parameters into properly configured RLlib algorithm
instances, supporting both the new and legacy RLlib APIs.

Key Components:
    - :func:`create_algorithm_config`: Main configuration creation function
    - Environment setup with seeding and rendering support
    - Callback system integration and management
    - Learner class configuration and mixing
    - RL module and catalog class handling

The module bridges the gap between high-level experiment specifications and
low-level RLlib configuration requirements, providing a clean interface for
complex algorithm setup scenarios.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, TypeVar, cast

from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.callbacks.algorithm.model_config_saver_callback import save_model_config_and_architecture
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
import gymnasium as gym
from gymnasium.envs.registration import VectorizeMode
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModule, RLModuleSpec
from ray.tune import logger as tune_logger

from ray_utilities.callbacks.algorithm.discrete_eval_callback import DiscreteEvalCallback
from ray_utilities.callbacks.algorithm.env_render_callback import make_render_callback
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback
from ray_utilities.config import add_callbacks_to_config, seed_environments_for_config
from ray_utilities.learners import mix_learners
from ray_utilities.learners.leaner_with_debug_connector import LearnerWithDebugConnectors

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.core.learner import Learner
    from ray.rllib.core.models.catalog import Catalog

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import NamespaceType
    from ray_utilities.typing.generic_rl_module import CatalogWithConfig, RLModuleWithConfig

_ConfigType = TypeVar("_ConfigType", bound="AlgorithmConfig")

_ModelConfig = TypeVar("_ModelConfig", bound="None | dict | Any")

logger = logging.getLogger(__name__)


def create_algorithm_config(
    args: dict[str, Any] | NamespaceType[DefaultArgumentParser],
    env_type: Optional[str | gym.Env] = None,
    env_seed: Optional[int] = None,
    *,
    new_api: Optional[bool] = True,
    module_class: Optional[type[RLModule | RLModuleWithConfig[_ModelConfig]]],
    catalog_class: Optional[type[Catalog | CatalogWithConfig[_ModelConfig]]],
    learner_class: Optional[type["Learner"]] = None,
    model_config: dict[str, Any] | _ModelConfig,
    config_class: Optional[type[_ConfigType]] = PPOConfig,
    framework: Literal["torch", "tf2"] | Any,
    base_config: Optional[_ConfigType] = None,
    discrete_eval: bool = False,
) -> tuple[_ConfigType, RLModuleSpec]:
    """Create a comprehensive Ray RLlib algorithm configuration from experiment parameters.

    This function provides a centralized way to create properly configured RLlib
    algorithm instances based on parsed command-line arguments and experiment
    specifications. It handles environment setup, callback registration, learner
    configuration, and RL module specifications.

    The function supports both new and legacy RLlib APIs and provides extensive
    customization options for advanced use cases while maintaining sensible
    defaults for common scenarios.

    Args:
        args: Parsed command-line arguments containing experiment configuration.
            Can be a dictionary or a :class:`DefaultArgumentParser` instance.
        env_type: Environment identifier or Gymnasium environment instance.
            If not provided, will be extracted from args.
        env_seed: Environment seed for reproducibility. Deprecated in favor of
            :class:`~ray_utilities.callbacks.algorithm.seeded_env_callback.SeedEnvsCallback`.
        new_api: Whether to use the new RLlib API (RLModule/Learner). Defaults to ``True``.
        module_class: Custom RL module class for the algorithm. Should inherit from
            :class:`ray.rllib.core.rl_module.RLModule`. If ``None``, requires manual
            configuration afterward.
        catalog_class: Catalog class used with the ``module_class`` for model creation.
            Should inherit from :class:`ray.rllib.core.models.catalog.Catalog`.
        learner_class: Custom learner class for training. If ``None``, uses algorithm defaults.
        model_config: Configuration dictionary for the neural network model implementation.
            Structure depends on the chosen framework and module class.
        config_class: RLlib algorithm configuration class. Defaults to
            :class:`ray.rllib.algorithms.ppo.PPOConfig`.
        framework: Deep learning framework to use (``"torch"`` or ``"tf2"``).
        base_config: Optional existing configuration instance to update instead of creating new.
        discrete_eval: Whether to add discrete evaluation capabilities through
            :class:`~ray_utilities.callbacks.algorithm.discrete_eval_callback.DiscreteEvalCallback`.

    Returns:
        A tuple containing:
            - **config**: Configured RLlib algorithm configuration instance
            - **module_spec**: RL module specification for the algorithm

    Raises:
        ExceptionGroup: When neither ``base`` nor ``config_class`` is provided,
            or when ``config_class`` is not a proper type.

    Example:
        >>> args = parser.parse_args(["--env", "CartPole-v1", "--framework", "torch"])
        >>> config, module_spec = create_algorithm_config(
        ...     args,
        ...     module_class=MyRLModule,
        ...     catalog_class=MyCatalog,
        ...     model_config={"hidden_layers": [256, 256]},
        ...     framework="torch",
        ... )

    Note:
        The function automatically configures environment seeding, callback registration,
        and evaluation settings based on the provided arguments. For discrete evaluation,
        ensure the environment supports the required interface.

    See Also:
        :class:`DefaultArgumentParser`: Argument parsing
        :class:`~ray_utilities.callbacks.algorithm.seeded_env_callback.SeedEnvsCallback`: Environment seeding
        :class:`~ray_utilities.callbacks.algorithm.discrete_eval_callback.DiscreteEvalCallback`: Discrete evaluation
    """
    if not base_config and not isinstance(config_class, type):
        raise ExceptionGroup(
            "base or config_class must be provided",
            [
                ValueError("base or config_class must be provided"),
                TypeError(f"config_class must be a type, not {type(config_class)}"),
            ],
        )
    if not isinstance(args, dict):
        if hasattr(args, "as_dict"):  # Tap
            args = cast("dict[str, Any]", args.as_dict())
        else:
            args = vars(args).copy()
    if not env_type and not args["env_type"]:
        raise ValueError("No environment specified")
    env_spec: Final = env_type or args["env_type"]
    del env_type
    assert env_spec, "No environment specified"
    if base_config:
        config = base_config
        if config_class and not issubclass(config_class, type(base_config)):
            logger.warning(
                "base config of type %s is not a subclass of config_class %s, "
                "change config_class or set config_class to None to avoid this warning.",
                type(base_config),
                config_class,
            )
    else:
        assert config_class is not None
        config = config_class()

    env_config: dict[str, Any] = {}  # kwargs for environment __init__
    if args["render_mode"]:
        env_config["render_mode"] = args["render_mode"]
    if env_seed is not None:
        logger.warning(
            "env_seed is deprecated, use SeedEnvsCallback/seed_environments_for_config(config, env_seed) instead, "
            "env creation might fail."
        )
        env_config.update({"seed": env_seed, "env_type": env_spec})
        # Will use a SeededEnvCallback to apply seed and generators
    config.environment(env_spec, env_config=env_config)
    if args["test"]:
        # increase time in case of debugging the sampler
        config.env_runners(sample_timeout_s=1000)
    try:
        config.env_runners(
            # experimental
            gym_env_vectorize_mode=(VectorizeMode.ASYNC if args["num_envs_per_env_runner"] > 1 else VectorizeMode.SYNC),  # pyright: ignore[reportArgumentType]
        )
    except TypeError:
        logger.error("Current ray version does not support AlgorithmConfig.env_runners(gym_env_vectorize_mode=...)")
    config.resources(
        # num_gpus=1 if args["gpu"] else 0,4
        # process that runs Algorithm.training_step() during Tune
        num_cpus_for_main_process=1,
        # num_learner_workers=4 if args["parallel"] else 1,
        # num_cpus_per_learner_worker=1,
        # num_cpus_per_worker=1,
    )
    config.env_runners(
        num_env_runners=2 if args["parallel"] and args["num_env_runners"] < 2 else args["num_env_runners"],
        num_cpus_per_env_runner=1,  # num_cpus_per_worker
        # How long an rollout episode lasts, for "auto" calculated from batch_size
        # total_train_batch_size / (num_envs_per_env_runner * num_env_runners)
        # rollout_fragment_length=1,  # Default: "auto"
        num_envs_per_env_runner=args["num_envs_per_env_runner"],
        # validate_env_runners_after_construction=args["test"],
        # 1) "truncate_episodes": Each call to `EnvRunner.sample()` returns a
        #    batch of at most `rollout_fragment_length * num_envs_per_env_runner` in
        #    size. The batch is exactly `rollout_fragment_length * num_envs`
        #    in size if postprocessing does not change batch sizes.
        # Use if not using GAE
        # 2) "complete_episodes": Each call to `EnvRunner.sample()` returns a
        #    batch of at least `rollout_fragment_length * num_envs_per_env_runner` in
        #    size. Episodes aren't truncated, but multiple episodes
        #    may be packed within one batch to meet the (minimum) batch size.
        batch_mode="truncate_episodes",
    )
    config.learners(
        # for fractional GPUs, you should always set num_learners to 0 or 1
        num_learners=1 if args["parallel"] else 0,
        num_cpus_per_learner=0 if args["test"] and args["num_jobs"] < 2 else 1,
        num_gpus_per_learner=1 if args["gpu"] else 0,
    )
    config.framework(framework)
    learner_mix: list[type[Learner]] = [learner_class or config.learner_class]
    if not args.get("keep_masked_samples", False):
        from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner  # noqa: PLC0415

        learner_mix.insert(0, RemoveMaskedSamplesLearner)
    if False:  # NOTE: Must always be the first in the mix
        learner_mix.insert(0, LearnerWithDebugConnectors)
    learner_config_dict = {
        "dynamic_buffer": args["dynamic_buffer"],
        "dynamic_batch": args["dynamic_batch"],
        "total_steps": args["total_steps"],
        "remove_masked_samples": not args["keep_masked_samples"],
        "min_dynamic_buffer_size": args["min_step_size"],
        "max_dynamic_buffer_size": args["max_step_size"],
        "accumulate_gradients_every": args["accumulate_gradients_every"],
    }
    if len(learner_mix) > 1:
        mixed_learner_class = mix_learners(learner_mix)
        try:  # new upcoming interface  # TODO: Check when this is changed ray 2.50+
            config.learners(learner_class=mixed_learner_class)  # pyright: ignore[reportCallIssue]
        except TypeError:  # old interface
            config.training(learner_class=mixed_learner_class)
    try:  # new upcoming interface
        config.learners(learner_config_dict=learner_config_dict)  # pyright: ignore[reportCallIssue]
    except TypeError:  # old interface
        config.training(learner_config_dict=learner_config_dict)

    config.training(
        gamma=0.99,
        # with a growing number of Learners and to increase the learning rate as follows:
        # lr = [original_lr] * ([num_learners] ** 0.5)
        lr=args["lr"],
        # The total effective batch size is then
        # `num_learners` x `train_batch_size_per_learner` and you can
        # access it with the property `AlgorithmConfig.total_train_batch_size`.
        train_batch_size_per_learner=args["train_batch_size_per_learner"],
        grad_clip=0.5,
    )
    try:
        cast("PPOConfig", config).training(
            minibatch_size=args["minibatch_size"],
        )
    except TypeError:
        cast("PPOConfig", config).training(
            sgd_minibatch_size=args["minibatch_size"],
        )
    if isinstance(config, PPOConfig):
        config.training(
            # PPO Specific
            use_critic=True,
            clip_param=0.2,
            # grad_clip_by="norm",
            entropy_coeff=0.01,
            # vf_clip_param=10,
            use_kl_loss=False,
            use_gae=True,  # Must be true to use "truncate_episodes"
        )
    # Create a single agent RL module spec.
    # NOTE: This might needs adjustment when using VectorEnv
    if isinstance(config.env, str) and config.env != "seeded_env":
        init_env = gym.make(config.env)
    elif config.env == "seeded_env":
        if isinstance(env_spec, str):
            init_env = gym.make(env_spec)
        else:
            init_env = env_spec
    else:
        assert not TYPE_CHECKING or config.env
        init_env = gym.make(config.env.unwrapped.spec.id)  # pyright: ignore[reportOptionalMemberAccess]
    # Note: legacy keys are updated below
    module_spec = RLModuleSpec(
        module_class=module_class,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config=cast("dict[str, Any]", model_config),
        catalog_class=catalog_class,
    )
    # module = module_spec.build()
    config.rl_module(  # only in RLModuleSpec is not sufficient
        rl_module_spec=module_spec,
    )
    if model_config is not None:
        config.rl_module(model_config=model_config)
    # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
    evaluation_duration = 30
    config.evaluation(
        evaluation_interval=16,  # Note can be adjusted dynamically by DynamicEvalInterval
        evaluation_duration=evaluation_duration,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=(
            2 if args["parallel"] and args["evaluation_num_env_runners"] < 2 else args["evaluation_num_env_runners"]
        ),
        # NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # results in the evaluation workers not using this optimal policy!
        evaluation_config=PPOConfig.overrides(
            explore=False,
            num_envs_per_env_runner=min(5, args["num_envs_per_env_runner"]),
            metrics_num_episodes_for_smoothing=evaluation_duration,  # take metrics over all eval episodes
        ),
    )
    # Stateless callbacks
    if not args["no_exact_sampling"]:
        add_callbacks_to_config(config, on_sample_end=exact_sampling_callback)
    add_callbacks_to_config(config, on_algorithm_init=save_model_config_and_architecture)
    # add_callbacks_to_config(config, on_sample_end=reset_episode_metrics_each_iteration)

    # region Stateful callbacks
    callbacks: list[type[DefaultCallbacks]] = []
    base_callbacks = None
    if base_config is not None and base_config.callbacks_class:
        if isinstance(base_config.callbacks_class, list):
            base_callbacks = base_config.callbacks_class
        else:
            base_callbacks = [base_config.callbacks_class]
    if discrete_eval:
        callbacks.append(DiscreteEvalCallback)
    if args["env_seeding_strategy"] == "sequential":
        # Must set this in the trainable with seed_environments_for_config(config, env_seed)
        logger.info(
            "Using sequential env seed strategy, "
            "Remember to call seed_environments_for_config(config, env_seed) with a seed acquired from the trial."
        )
    elif args["env_seeding_strategy"] == "same":
        # TODO: could use env_seed here, allows to sample a constant random seed != args["seed"]
        seed_environments_for_config(config, args["seed"])
    elif args["env_seeding_strategy"] == "constant":
        seed_environments_for_config(config, SeedEnvsCallback.env_seed)
    if args["render_mode"]:
        callbacks.append(make_render_callback())
    if not args["no_dynamic_eval_interval"]:
        callbacks.append(DynamicEvalInterval)
    if base_callbacks:
        same = set(base_callbacks).intersection(set(callbacks))
        only_in_base = set(base_callbacks).difference(set(callbacks))
        only_in_new = set(callbacks).difference(set(base_callbacks))
        if only_in_base or only_in_new:
            logger.warning(
                "Callbacks of base differs from callbacks in new config:\n%s vs\n%s", only_in_base, only_in_new
            )
        callbacks = [*(cb for cb in base_callbacks if cb not in same), *callbacks]
    if callbacks:
        if len(callbacks) == 1:
            callback_class = callbacks[0]
        else:
            callback_class = callbacks
        if False:
            # OLD API
            multi_callback = make_multi_callbacks(callback_class)
            # Necessary patch for new_api, cannot use this callback with new API
            multi_callback.on_episode_created = DefaultCallbacks.on_episode_created
            config.callbacks(callbacks_class=multi_callback)
        else:
            config.callbacks(callbacks_class=callback_class)
    # endregion

    config.reporting(
        keep_per_episode_custom_metrics=True,  # If True calculate max min mean
        log_gradients=args["test"],  # Default is True
        # Will smooth metrics in the reports, e.g. tensorboard
        # NOTE This value will smooth over num_episodes, which might be from the past iterations.
        # But should be > 1 to smooth over episodes from the current iteration
        # We use the reset_episode_metrics_each_iteration, so that no bleed over happens,
        # but for all episodes to be considered we use a high value here
        # Should use ~50-60 for a higher batch_size
        metrics_num_episodes_for_smoothing=30,  # Default is 100
    )
    config.debugging(
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.debugging.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-debugging
        seed=args["seed"],
        log_sys_usage=False,
        # These loggers will log more metrics which are stored less-accessible in the ~/ray_results/logdir
        # Using these could be useful if no Tuner is used
        logger_config={"type": tune_logger.NoopLogger},
    )
    if new_api is not None:
        config.api_stack(enable_rl_module_and_learner=new_api, enable_env_runner_and_connector_v2=new_api)
    # Checks
    warn_about_larger_minibatch_size(
        minibatch_size=config.minibatch_size, train_batch_size_per_learner=config.train_batch_size_per_learner
    )
    warn_if_batch_size_not_divisible(
        batch_size=config.train_batch_size_per_learner,
        num_envs_per_env_runner=config.num_envs_per_env_runner,
    )
    warn_if_minibatch_size_not_divisible(
        minibatch_size=config.minibatch_size,
        num_envs_per_env_runner=config.num_envs_per_env_runner,
    )
    config.validate_train_batch_size_vs_rollout_fragment_length()
    return config, module_spec
