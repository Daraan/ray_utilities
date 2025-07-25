from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, TypeVar, cast

import gymnasium as gym
from gymnasium.envs.registration import VectorizeMode
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModule, RLModuleSpec
from ray.tune import logger as tune_logger

from ray_utilities.callbacks.algorithm.discrete_eval_callback import DiscreteEvalCallback
from ray_utilities.callbacks.algorithm.env_render_callback import make_render_callback
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback, make_seeded_env_callback
from ray_utilities.config import add_callbacks_to_config
from ray_utilities.learners import mix_learners
from ray_utilities.learners.leaner_with_debug_connector import LearnerWithDebugConnectors

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.core.learner import Learner
    from ray.rllib.core.models.catalog import Catalog

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
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
    config_class: type[_ConfigType] = PPOConfig,
    framework: Literal["torch", "tf2"],
    discrete_eval: bool = False,
) -> tuple[_ConfigType, RLModuleSpec]:
    """
    Creates a basic algorithm

    Args:
        args: Arguments for the algorithm
        env_type: Environment type
        env_seed: Environment seed, deprecated use `SeedEnvsCallback` instead.
        module_class: RLModule class used for the algorithm.
            Not recommended to be None must be updated in that case manually afterwards.
        catalog_class: Catalog used with the `module_class`
        model_config: Configuration dict describing the torch/tf implemented model, by the module_class
        config_class: Config class of the Algorithm, defaults to PPOConfig
        discrete_eval: Wether to add the DiscreteEvalCallback
    """
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
            gym_env_vectorize_mode=VectorizeMode.ASYNC,  # pyright: ignore[reportArgumentType]
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
        num_env_runners=2 if args["parallel"] else 0,
        num_cpus_per_env_runner=1,  # num_cpus_per_worker
        # How long an rollout episode lasts, for "auto" calculated from batch_size
        # total_train_batch_size / (num_envs_per_env_runner * num_env_runners)
        # rollout_fragment_length=1,  # Default: "auto"
        num_envs_per_env_runner=1,
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
        from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner

        learner_mix.insert(0, RemoveMaskedSamplesLearner)
    if False:  # NOTE: Must always be the first in the mix
        learner_mix.insert(0, LearnerWithDebugConnectors)
    if len(learner_mix) > 1:
        config.training(learner_class=mix_learners(learner_mix))
    config.training(
        gamma=0.99,
        # with a growing number of Learners and to increase the learning rate as follows:
        # lr = [original_lr] * ([num_learners] ** 0.5)
        lr=(
            1e-3
            if True
            # Shedule LR
            else [
                [0, 8e-3],  # <- initial value at timestep 0
                [100, 4e-3],
                [400, 1e-3],
                [800, 1e-4],
            ]
        ),
        # The total effective batch size is then
        # `num_learners` x `train_batch_size_per_learner` and you can
        # access it with the property `AlgorithmConfig.total_train_batch_size`.
        train_batch_size_per_learner=args["train_batch_size_per_learner"],
        grad_clip=0.5,
        learner_config_dict={
            "dynamic_buffer": args["dynamic_buffer"],
            "dynamic_batch": args["dynamic_batch"],
            "total_steps": args["total_steps"],
            "remove_masked_samples": not args["keep_masked_samples"],
            "min_dynamic_buffer_size": args["min_step_size"],
            "max_dynamic_buffer_size": args["max_step_size"],
            "accumulate_gradients_every": args["accumulate_gradients_every"],
        },
    )
    try:
        cast("PPOConfig", config).training(
            num_epochs=20,
            minibatch_size=args["minibatch_size"],
        )
    except TypeError:
        cast("PPOConfig", config).training(
            num_sgd_iter=20,
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
    config.rl_module(
        rl_module_spec=module_spec,
    )
    # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
    config.evaluation(
        evaluation_interval=10,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=2 if args["parallel"] else 0,
        # NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # results in the evaluation workers not using this optimal policy!
        evaluation_config=PPOConfig.overrides(
            explore=False,
        ),
    )
    # Stateless callbacks
    if not args["no_exact_sampling"]:
        add_callbacks_to_config(config, on_sample_end=exact_sampling_callback)
    # Statefull callbacks
    callbacks: list[type[DefaultCallbacks]] = []
    if discrete_eval:
        callbacks.append(DiscreteEvalCallback)
    if args["env_seeding_strategy"] == "sequential":
        # Must set this in the trainable with seed_environments_for_config(config, env_seed)
        logger.info(
            "Using sequential env seed strategy, "
            "Remember to call seed_environments_for_config(config, env_seed) with a seed acquired from the trial."
        )
    elif args["env_seeding_strategy"] == "same":
        make_seeded_env_callback(args["seed"])
    elif args["env_seeding_strategy"] == "constant":
        make_seeded_env_callback(SeedEnvsCallback.env_seed)
    if args["render_mode"]:
        callbacks.append(make_render_callback())

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

    config.reporting(
        keep_per_episode_custom_metrics=True,  # If True calculate max min mean
        log_gradients=False,  # Default is True
        # Will smooth metrics in the reports, e.g. tensorboard
        metrics_num_episodes_for_smoothing=1,  # Default is 100
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
    try:
        if config.train_batch_size_per_learner % config.minibatch_size != 0:  # pyright: ignore[reportOperatorIssue]
            logger.warning(
                "Train batch size (%s) is not divisible by minibatch size (%s).",
                config.train_batch_size_per_learner,
                config.minibatch_size,
            )
    except TypeError:
        logger.debug("Error encountered while checking train_batch_size_per_learner", exc_info=True)
    config.validate_train_batch_size_vs_rollout_fragment_length()
    return config, module_spec
