from __future__ import annotations

# pyright: reportOptionalMemberAccess=none
import argparse
import os
import tempfile
import time
import unittest
from inspect import isclass
from typing import TYPE_CHECKING, Any, cast

import tree
import typing_extensions as te
from ray import tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import (
    EVAL_METRIC_RETURN_MEAN,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.random import seed_everything
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger
from ray_utilities.testing_utils import DisableGUIBreakpoints, InitRay, SetupDefaults, patch_args
from ray_utilities.training.default_class import DefaultTrainable, TrainableBase

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


# os.environ["RAY_DEBUG"]="0"


class TestSetupClasses(SetupDefaults):
    def test_frozen_config(self):
        with patch_args():
            setup = AlgorithmSetup()
        self.assertTrue(isclass(setup.trainable))  # change test if fails
        assert isclass(setup.trainable)
        with self.assertRaisesRegex(AttributeError, "Cannot set attribute .* already frozen"):
            setup.config.training(num_epochs=3, minibatch_size=321)
        # unset trainable, unfreeze config
        self.assertIsNotNone(setup.trainable)
        setup.unset_trainable()
        self.assertIsNone(setup.trainable)
        setup.config.training(num_epochs=4, minibatch_size=222)
        setup.create_trainable()
        assert issubclass(setup.trainable, DefaultTrainable)
        trainable = setup.trainable({})
        self.assertEqual(trainable.algorithm_config.num_epochs, 4)
        self.assertEqual(trainable.algorithm_config.minibatch_size, second=222)
        self.set_max_diff(15000)
        self.assertIsNot(trainable.algorithm_config, setup.config)
        self.compare_configs(
            trainable.algorithm_config,
            setup.config,
            ignore=[
                # ignore callbacks that are created on Trainable.setup
                "callbacks_on_environment_created",
            ],
        )

    def test_basic(self):
        with patch_args():
            setup = AlgorithmSetup()
        self.assertIsNotNone(setup.config)
        self.assertIsNotNone(setup.args)
        self.assertIsNotNone(setup.create_tuner())
        self.assertIsNotNone(setup.create_config())
        self.assertIsNotNone(setup.create_param_space())
        self.assertIsNotNone(setup.create_parser())
        self.assertIsNotNone(setup.create_tags())
        self.assertIsNotNone(setup.create_trainable())

    def test_argument_usage(self):
        # Test warning and failure
        with patch_args("--batch_size", "1234"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 1234)
        with patch_args("--train_batch_size_per_learner", "456"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 456)

    def test_dynamic_param_spaces(self):
        # Test warning and failure
        with patch_args("--tune", "dynamic_buffer"):
            # error is deeper, need to catch outer SystemExit and check context/cause
            with self.assertRaises(SystemExit) as context:
                AlgorithmSetup().create_param_space()
            self.assertIsInstance(context.exception.__context__, argparse.ArgumentError)
            self.assertIn("invalid choice: 'dynamic_buffer'", context.exception.__context__.message)  # type: ignore
        with patch_args("--tune", "all", "rollout_size"):
            with self.assertRaisesRegex(ValueError, "Cannot use 'all' with other tune parameters"):
                AlgorithmSetup().create_param_space()
        with patch_args("--tune", "rollout_size", "rollout_size"):
            with self.assertLogs(logger, level="WARNING") as cm:
                AlgorithmSetup().create_param_space()
            self.assertIn("Unused dynamic tuning parameters: ['rollout_size']", cm.output[0])
        th = te.get_type_hints(DefaultArgumentParser)["tune"]
        self.assertIs(te.get_origin(th), te.Union)
        th_args = te.get_args(th)
        th_lists = [
            literal
            for li in [te.get_args(arg)[0] for arg in th_args if te.get_origin(arg) is list]
            for literal in te.get_args(li)
            if literal != "all"
        ]
        self.assertIn("rollout_size", th_lists)
        self.assertNotIn("all", th_lists)
        for param in th_lists:
            with (
                patch_args(
                    "--tune", param,
                    "--num_jobs", "1",
                    "--total_steps", "10",
                    "-it", "2",
                    "--num_samples", "16",
                )  # ,
                # self.assertNoLogs(logger, level="WARNING"),
            ):  # fmt: skip
                if param == "batch_size":  # shortcut name
                    param = "train_batch_size_per_learner"  # noqa: PLW2901
                setup = AlgorithmSetup()
                param_space = setup.create_param_space()
                self.assertIn(param, param_space)
                self.assertIsNotNone(param_space[param])  # dict with list
                if "grid_search" in param_space[param]:
                    grid = param_space[param]["grid_search"]
                else:
                    grid = []

                def fake_trainable(params, param=param):
                    return {
                        "current_step": 0,
                        "evaluation/env_runners/episode_return_mean": 42,
                        "param_value": params[param],
                    }

                setup.trainable = fake_trainable  # type: ignore
                tuner = setup.create_tuner()
                result_grid = tuner.fit()
                evaluated_params = [r.metrics["param_value"] for r in result_grid]  # pyright: ignore[reportOptionalSubscript]
                # Check that all grid search values were evaluated
                self.assertEqual(
                    len(set(evaluated_params)),
                    len(grid),
                    f"Evaluated params do not match grid: {evaluated_params} != {grid}",
                )

    def test_config_overrides_via_setup(self):
        with patch_args("--batch_size", "1234", "--minibatch_size", "444"):
            # test with init_trainable=False
            setup = AlgorithmSetup(init_trainable=False)
            setup.config.training(num_epochs=3, minibatch_size=321)
            Trainable = setup.create_trainable()
            assert isinstance(Trainable, type) and issubclass(Trainable, DefaultTrainable)
            trainable = Trainable()

        self.assertEqual(trainable.algorithm_config.num_epochs, 3)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 1234)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 321)

        self.set_max_diff(15000)
        self.assertIsNot(trainable.algorithm_config, setup.config)
        self.compare_configs(
            trainable.algorithm_config,
            setup.config,
            ignore=[
                # ignore callbacks that are created on Trainable.setup
                "callbacks_on_environment_created",
            ],
        )

    def test_config_overwrites_on_trainable(self):
        with patch_args("--batch_size", "1234", "--minibatch_size", "444"):
            # test with init_trainable=False
            setup = AlgorithmSetup(init_trainable=True)
            self.assertEqual(setup.config.minibatch_size, 444)
            self.assertEqual(setup.config.train_batch_size_per_learner, 1234)
            trainable = setup.trainable_class(
                overwrite_algorithm=AlgorithmConfig.overrides(
                    num_epochs=22,
                    minibatch_size=321,
                )
            )
        # Test that config is not changed
        self.assertEqual(setup.config.minibatch_size, 444)
        self.assertEqual(setup.config.train_batch_size_per_learner, 1234)

        self.assertEqual(trainable.algorithm_config.num_epochs, 22)  # changed
        self.assertEqual(trainable.algorithm_config.minibatch_size, 321)  # changed
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 1234)  # unchanged
        # Unchanged on setup
        self.assertNotEqual(trainable._setup.config.num_epochs, 22)  # changed from default (20)
        self.assertEqual(trainable._setup.config.minibatch_size, 444)  # changed
        self.assertEqual(trainable._setup.config.train_batch_size_per_learner, 1234)  # unchanged

        self.set_max_diff(15000)
        self.assertIsNot(trainable.algorithm_config, setup.config)
        self.compare_configs(
            trainable.algorithm_config,
            setup.config,
            ignore=[
                # ignore callbacks that are created on Trainable.setup
                "callbacks_on_environment_created",
                # ignore keys that differ via overrides
                "num_epochs",
                "minibatch_size",
            ],
        )


ENV_STEPS_PER_ITERATION = 10


class TestAlgorithm(InitRay, DisableGUIBreakpoints, SetupDefaults):
    def setUp(self):
        super().setUp()

    def test_evaluate(self):
        _result = self._DEFAULT_SETUP.build_algo().evaluate()

    def test_step(self):
        algo = self._DEFAULT_SETUP.build_algo()
        _result = algo.step()

        with self.subTest("test weights after step"):
            env_runner = cast("SingleAgentEnvRunner", algo.env_runner_group.local_env_runner)  # type: ignore[attr-defined]
            eval_env_runner = cast("SingleAgentEnvRunner", algo.eval_env_runner_group.local_env_runner)  # type: ignore[attr-defined]
            learner = algo.learner_group._learner
            assert learner
            learner_multi = learner.module

            runner_module = env_runner.module
            eval_module = eval_env_runner.module
            assert runner_module and eval_module
            learner_module = learner_multi["default_policy"]
            algo_module = algo.get_module()

            # Test weight sync
            # NOTE THESE ARE TRAIN STATES not arrays
            if runner_module is not algo_module:  # these are normally equal
                print("WARNING: modules are not the object")
                # identity
                self.assertDictEqual(
                    runner_module.get_state(),
                    algo_module.get_state(),
                    msg="runner_module vs algo_module",
                )
            self.set_max_diff(38000)
            state_on_learner = learner_module.get_state()
            state_on_algo = algo_module.get_state()
            # algo might not have critic states:
            critic_keys_learner = {k for k in state_on_learner.keys() if k.startswith(("encoder.critic_encoder", "vf"))}
            critic_keys_algo = {k for k in state_on_algo.keys() if k.startswith(("encoder.critic_encoder", "vf"))}
            # remove weights that are only on learner
            learner_weights = {
                k: v for k, v in state_on_learner.items() if k not in critic_keys_learner - critic_keys_algo
            }
            self.assertGreater(len(learner_weights), 0, "No weights found in learner state")
            self.compare_weights(
                learner_weights,
                state_on_algo,
                msg="learner_module vs algo_module",
            )
            algo.evaluate()
            # These are possibly not updated
            self.compare_weights(
                eval_module.get_state(),
                algo_module.get_state(),
            )

    # @unittest.skip("Fix first: Ray moves checkpoint need to load from different location")
    def test_stopper_with_checkpoint(self):
        from copy import deepcopy

        setup = deepcopy(self._DEFAULT_SETUP_LOW_RES)
        config = setup.config
        setup.args.num_jobs = 1
        setup.args.num_samples = 1

        def fake_trainable(params):
            algo = config.build_algo()
            for i in range(10):
                result = algo.train()
                logger.info(
                    "Iteration %s: Sampled Lifetime: %s", i, result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME]
                )
                # algo.save_checkpoint(tempdir1)
                log = {"evaluation/env_runners/episode_return_mean": i} | result
                algo.save_checkpoint(tempdir1)
                if (i + 1) % 2 == 0:
                    tune.report(
                        log,
                        checkpoint=tune.Checkpoint.from_directory(tempdir1),
                    )
                else:
                    tune.report(
                        log,
                        checkpoint=None,
                    )

        setup.trainable = fake_trainable  # type: ignore
        tuner = setup.create_tuner()
        # With stopper should only iterate 4 times:
        assert tuner._local_tuner
        # Define stopper
        tuner._local_tuner._run_config.stop = {ENV_RUNNER_RESULTS + "/" + NUM_ENV_STEPS_SAMPLED_LIFETIME: 512}
        with tempfile.TemporaryDirectory(prefix=".ckpt_") as tempdir1:
            result_grid = tuner.fit()
            checkpoint = result_grid[0].checkpoint
            assert checkpoint
            # Problem ray moves checkpoint to new location
            algo_restored = config.build_algo().from_checkpoint(checkpoint.path)
            assert algo_restored.metrics
            self.assertNotEqual(
                algo_restored.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)), 0
            )
            self.assertEqual(
                algo_restored.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)), 512
            )
            self.assertEqual(algo_restored.iteration, 4)
            self.assertEqual(result_grid[0].metrics["training_iteration"], 4)
            self.assertEqual(result_grid[0].metrics["evaluation/env_runners/episode_return_mean"], 512 // 128 - 1)


class TestMetricsRestored(InitRay, DisableGUIBreakpoints, SetupDefaults):
    def _test_checkpoint_values(self, result1: dict[str, Any], result2: dict[str, Any]):
        """Test NUM_ENV_STEPS_SAMPLED and NUM_ENV_STEPS_PASSED_TO_LEARNER values of two training results."""
        env_runner1 = result1
        env_runner2 = result2
        # This step - trivial tests
        # TODO: is this also checked with learner?
        self.assertEqual(env_runner1[NUM_ENV_STEPS_SAMPLED], env_runner2[NUM_ENV_STEPS_SAMPLED])
        self.assertEqual(env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER], env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER])

        # Lifetime stats:
        self.assertEqual(
            env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME], env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME]
        )
        self.assertEqual(env_runner1[NUM_ENV_STEPS_SAMPLED_LIFETIME], env_runner2[NUM_ENV_STEPS_SAMPLED_LIFETIME])
        # This would be amazing, but does not look possible:

    #        self.assertEqual(env_runner1[EPISODE_RETURN_MEAN], env_runner2[EPISODE_RETURN_MEAN])

    @unittest.skip("Needs to be fixed in ray first")
    def test_checkpointing_native(self):
        """
        NOTE: This test needs a patch in ray (very!) earliest coming with 2.47.2+
              However the custom metric NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME is still broken in 2.47
        """
        config = PPOConfig()
        config.training(
            train_batch_size_per_learner=ENV_STEPS_PER_ITERATION, num_epochs=1, minibatch_size=ENV_STEPS_PER_ITERATION
        )
        config.debugging(
            seed=11,
            log_sys_usage=False,
        )
        config.reporting(metrics_num_episodes_for_smoothing=1, keep_per_episode_custom_metrics=True)  # no smoothing
        config.environment(env="CartPole-v1")

        def log_custom_metric(metrics_logger: MetricsLogger, **kwargs):
            # Track env steps in a second metric
            metrics_logger.log_value(
                NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, metrics_logger.peek(NUM_ENV_STEPS_SAMPLED), reduce="sum"
            )

        config.callbacks(on_sample_end=log_custom_metric)
        algo_0_runner = config.env_runners(
            num_env_runners=0,
        ).build_algo()
        algo_1_runner = config.env_runners(
            num_env_runners=1,
        ).build_algo()
        # Metatest if local and remote env runner configs are correct
        with self.subTest("Trivial compare of config vs. env runner configs"):
            # compare with itself
            self.compare_env_runner_configs(algo_0_runner, algo_0_runner)
            self.compare_env_runner_configs(algo_1_runner, algo_1_runner)

        # Continue training and check new metris
        self._test_algo_checkpointing(
            algo_0_runner,
            algo_1_runner,
            metrics=[NUM_ENV_STEPS_SAMPLED_LIFETIME, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
        )

    def test_with_tuner(self):
        """Test if key stats are restored correctly - does not test further training and metrics"""
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", str(ENV_STEPS_PER_ITERATION),
            "--minibatch_size", str(ENV_STEPS_PER_ITERATION),
            "--iterations", "15",
            "--seed", "12",
        ):  # fmt: off
            frequency = 5
            # cannot make this deterministic on local vs remote
            seed_everything(None, 42)
            setup = AlgorithmSetup(init_trainable=False)
            setup.config.env_runners(num_env_runners=0)
            setup.config.training(minibatch_size=5)  # insert some noise
            setup.config.debugging(seed=42)
            setup.create_trainable()
            tuner_0 = setup.create_tuner()
            seed_everything(None, 42)
            setup = AlgorithmSetup(init_trainable=False)
            setup.config.env_runners(num_env_runners=1)
            setup.config.training(minibatch_size=5)  # insert some noise
            setup.config.debugging(seed=42)
            setup.create_trainable()
            tuner_1 = setup.create_tuner()
            tune_results = {}
            for num_env_runners, tuner in enumerate([tuner_0, tuner_1]):
                assert tuner._local_tuner
                tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                    checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                    checkpoint_score_order="max",
                    checkpoint_frequency=frequency,  # Save every iteration
                    # NOTE: num_keep does not appear to work here
                )
                seed_everything(None, 42)
                result = tuner.fit()
                self.assertEqual(result.num_errors, 0, "Encountered errors: " + str(result.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
                checkpoint_dir, checkpoints = self.get_checkpoint_dirs(result[0])
                self.assertEqual(
                    len(checkpoints),
                    3,
                    f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
                )
                # sort to get checkpoints in order 000, 001, 002
                tune_results[num_env_runners] = {
                    "checkpoint_dir": checkpoint_dir,
                    "checkpoints": sorted(checkpoints),
                    "trainables": [],
                }
                for step, checkpoint in enumerate(sorted(checkpoints), 1):
                    self.assertTrue(os.path.exists(checkpoint))

                    restored_trainable: DefaultTrainable[Any, AlgorithmConfig, Any] = DefaultTrainable.define(
                        setup
                    ).from_checkpoint(checkpoint)  # pyright: ignore[reportAssignmentType]
                    # restore is bad if algorithm_checkpoint_dir is a temp dir
                    self.assertEqual(restored_trainable.algorithm.iteration, step * frequency)
                    self.assertEqual(restored_trainable.algorithm_config.seed, 42)
                    self.assertEqual(restored_trainable._setup.args.seed, 12)
                    self.assertEqual(restored_trainable.algorithm_config.num_env_runners, num_env_runners)
                    self.assertEqual(restored_trainable._setup.config.num_env_runners, num_env_runners)
                    tune_results[num_env_runners]["trainables"].append(restored_trainable)
            self.assertGreater(len(tune_results[0]["trainables"]), 0)
            for step in range(len(tune_results[0]["trainables"])):
                with self.subTest(f"Compare trainables from step {(step + 1) * frequency}"):
                    trainable_0: DefaultTrainable = tune_results[0]["trainables"][step]
                    trainable_1: DefaultTrainable = tune_results[1]["trainables"][step]
                    assert trainable_0.algorithm.metrics and trainable_1.algorithm.metrics
                    metrics_0 = trainable_0.algorithm.metrics.reduce()
                    metrics_1 = trainable_1.algorithm.metrics.reduce()
                    self._test_checkpoint_values(
                        metrics_0[ENV_RUNNER_RESULTS],
                        metrics_1[ENV_RUNNER_RESULTS],
                    )
                    self.assertEqual(
                        metrics_0[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                        (step + 1) * frequency * ENV_STEPS_PER_ITERATION,
                        "steps count in learner it not equal",
                    )
                    self.assertEqual(
                        metrics_0[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                        metrics_1[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                        "steps count in learner it not equal",
                    )
                    self.assertEqual(
                        metrics_0[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                        (step + 1) * frequency * ENV_STEPS_PER_ITERATION,
                        "steps count in learner it not equal",
                    )
                    metrics_0_clean = self.clean_timer_logs(metrics_0)
                    metrics_1_clean = self.clean_timer_logs(metrics_1)
                    self.util_test_compare_env_runner_results(
                        metrics_0_clean[ENV_RUNNER_RESULTS],
                        metrics_1_clean[ENV_RUNNER_RESULTS],
                        "training",
                        strict=False,
                    )
                    if False:  # do not compare evaluation; no exact sampling and random differences does not align this
                        self.util_test_compare_env_runner_results(
                            metrics_0_clean["evaluation"][ENV_RUNNER_RESULTS],
                            metrics_1_clean["evaluation"][ENV_RUNNER_RESULTS],
                            "evaluation",
                            strict=False,
                        )

    @unittest.skip("Implementation Missing")
    def test_metrics_further_tuning(self): ...

    def _test_algo_checkpointing(
        self,
        algo_0_runner: Algorithm | TrainableBase,
        algo_1_runner: Algorithm | TrainableBase,
        metrics: list[str],
    ) -> dict[str, dict[int, dict[str, Any]]]:
        if isinstance(algo_0_runner, Algorithm) and isinstance(algo_1_runner, Algorithm):
            assert algo_0_runner.config and algo_1_runner.config and algo_0_runner.metrics and algo_1_runner.metrics
            algorithm_config_0 = algo_0_runner.config
            algorithm_config_1 = algo_1_runner.config
            BaseClass = Algorithm
        elif isinstance(algo_0_runner, TrainableBase) and isinstance(algo_1_runner, TrainableBase):
            assert (
                algo_0_runner._setup
                and algo_1_runner._setup
                and algo_0_runner.algorithm_config
                and algo_1_runner.algorithm_config
            )
            algorithm_config_0: AlgorithmConfig = algo_0_runner.algorithm_config
            algorithm_config_1: AlgorithmConfig = algo_1_runner.algorithm_config
            BaseClass = TrainableBase
        else:
            raise TypeError("Algo runners must be of type Algorithm or DefaultTrainable")

        self.assertEqual(algorithm_config_0.num_env_runners, 0)
        self.assertEqual(algorithm_config_1.num_env_runners, 1)
        # --- Step 1 ---
        result_algo_0_step1 = algo_0_runner.step()
        result_algo_1_step1 = algo_1_runner.step()
        with (
            tempfile.TemporaryDirectory(prefix=".ckpt_a0_") as checkpoint_0_step1,
            tempfile.TemporaryDirectory(prefix=".ckpt_a1_") as checkpoint_1_step1,
            tempfile.TemporaryDirectory(prefix=".ckpt_b0_") as checkpoint_0_step2,
            tempfile.TemporaryDirectory(prefix=".ckpt_b1_") as checkpoint_1_step2,
            tempfile.TemporaryDirectory(prefix=".ckpt_c0_") as checkpoint_0_step2_restored,
            tempfile.TemporaryDirectory(prefix=".ckpt_c1_") as checkpoint_1_step2_restored,
        ):
            # Save Step 1
            algo_0_runner.save_checkpoint(checkpoint_0_step1)
            algo_1_runner.save_checkpoint(checkpoint_1_step1)
            # --- Step 2 ---
            result_algo_0_step2 = algo_0_runner.step()
            result_algo_1_step2 = algo_1_runner.step()
            for metric in metrics:
                with self.subTest(f"Check {metric} after step 2", metric=metric):
                    self.assertEqual(
                        result_algo_0_step2[ENV_RUNNER_RESULTS][metric],
                        result_algo_1_step2[ENV_RUNNER_RESULTS][metric],
                    )
                    self.assertEqual(
                        result_algo_0_step2[ENV_RUNNER_RESULTS][metric],
                        ENV_STEPS_PER_ITERATION * 2,
                    )
            # Save Step 2
            algo_0_runner.save_checkpoint(checkpoint_0_step2)
            algo_1_runner.save_checkpoint(checkpoint_1_step2)

            # --- Step 3 ---
            result_algo_0_step3 = algo_0_runner.step()
            result_algo_1_step3 = algo_1_runner.step()
            for metric in metrics:
                with self.subTest(f"Check {metric} after step 3", metric=metric):
                    self.assertEqual(
                        result_algo_0_step3[ENV_RUNNER_RESULTS][metric],
                        result_algo_1_step3[ENV_RUNNER_RESULTS][metric],
                    )
                    self.assertEqual(
                        result_algo_0_step3[ENV_RUNNER_RESULTS][metric],
                        ENV_STEPS_PER_ITERATION * 3,
                    )
            # Load Step 1
            algo_0_runner_restored: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_0_step1)  # pyright: ignore[reportAssignmentType]
            algo_1_runner_restored: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_1_step1)  # pyright: ignore[reportAssignmentType]
            if isinstance(algo_0_runner_restored, Algorithm):
                metrics_0_restored = algo_0_runner_restored.metrics
            else:
                metrics_0_restored = algo_0_runner_restored.algorithm.metrics

            if isinstance(algo_1_runner_restored, Algorithm):
                metrics_1_restored = algo_1_runner_restored.metrics
            else:
                metrics_1_restored = algo_1_runner_restored.algorithm.metrics
            # Check loaded metric
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after restored step 1", metric=metric):
                    self.assertEqual(
                        metrics_0_restored.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION,
                    )
                    self.assertEqual(
                        metrics_1_restored.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION,
                    )
            tree.assert_same_structure(metrics_0_restored, metrics_1_restored)

            # --- Step 2 from restored & checkpoint ---
            result_algo0_step2_restored = algo_0_runner_restored.step()
            result_algo1_step2_restored = algo_1_runner_restored.step()
            # Check if metric was updated
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after restored step 2", metric=metric):
                    self.assertEqual(
                        metrics_0_restored.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                    )
                    self.assertEqual(
                        metrics_1_restored.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                    )
            algo_0_runner_restored.save_checkpoint(checkpoint_0_step2_restored)
            algo_1_runner_restored.save_checkpoint(checkpoint_1_step2_restored)

            # Check results
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after step 2", metric=metric):
                    with self.subTest("From checkpoint: env_runners=0 - Step 2"):
                        self.assertEqual(
                            result_algo_0_step2[ENV_RUNNER_RESULTS][metric],
                            result_algo0_step2_restored[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=0: {metric} does not match expected value "
                            f"{ENV_STEPS_PER_ITERATION * 2}",
                        )
                    with self.subTest("From checkpoint: env_runners=1 - Step 2"):
                        self.assertEqual(
                            result_algo_1_step2[ENV_RUNNER_RESULTS][metric],
                            result_algo1_step2_restored[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=1: {metric} does not match expected value "
                            f"{ENV_STEPS_PER_ITERATION * 2}",
                        )
            # Test after restoring a second time
            # Load Restored from step 2
            algo_0_restored_x2: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_0_step2_restored)  # pyright: ignore[reportAssignmentType]
            algo_1_restored_x2: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_1_step2_restored)  # pyright: ignore[reportAssignmentType]
            if isinstance(algo_0_restored_x2, Algorithm):
                metrics_0_restored_x2 = algo_0_restored_x2.metrics
            else:
                metrics_0_restored_x2 = algo_0_restored_x2.algorithm.metrics

            if isinstance(algo_1_restored_x2, Algorithm):
                metrics_1_restored_x2 = algo_1_restored_x2.metrics
            else:
                metrics_1_restored_x2 = algo_1_restored_x2.algorithm.metrics
            for metric in metrics:
                with self.subTest(f"(Checkpointed x2) Check {metric} after step 2", metric=metric):
                    self.assertEqual(
                        metrics_0_restored_x2.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                        f"Restored x2 num_env_runners=0: {metric} does not match expected value "
                        f"{ENV_STEPS_PER_ITERATION * 2}",
                    )
                    self.assertEqual(
                        metrics_1_restored_x2.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                        f"Restored x2 num_env_runners=1: {metric} does not match expected value "
                        f"{ENV_STEPS_PER_ITERATION * 2}",
                    )
            # Step 3 from restored
            result_algo0_step3_restored = algo_0_runner_restored.step()
            result_algo1_step3_restored = algo_1_runner_restored.step()
            result_algo0_step3_restored_x2 = algo_0_restored_x2.step()
            result_algo1_step3_restored_x2 = algo_1_restored_x2.step()

            # Test that all results after step 3 are have 300 steps
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after step 3", metric=metric):
                    with self.subTest("From checkpoint: env_runners=0 - Step 3"):
                        self.assertEqual(
                            result_algo_0_step3[ENV_RUNNER_RESULTS][metric],
                            result_algo0_step3_restored[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=0: {metric} does not match {ENV_STEPS_PER_ITERATION * 3}",
                        )
                    with self.subTest("From checkpoint: env_runners=0 - Step 3 (restored x2)"):
                        self.assertEqual(
                            result_algo_0_step3[ENV_RUNNER_RESULTS][metric],
                            result_algo0_step3_restored_x2[ENV_RUNNER_RESULTS][metric],
                            f"Restored x2 num_env_runners=0: {metric} does not match {ENV_STEPS_PER_ITERATION * 3}",
                        )
                    with self.subTest("From checkpoint: env_runners=1 - Step 3"):
                        self.assertEqual(
                            result_algo_1_step3[ENV_RUNNER_RESULTS][metric],
                            result_algo1_step3_restored[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=1: {metric} does not match {ENV_STEPS_PER_ITERATION * 3}",
                        )
                    with self.subTest("From checkpoint: env_runners=1 - Step 3 (restored x2)"):
                        self.assertEqual(
                            result_algo_1_step3[ENV_RUNNER_RESULTS][metric],
                            result_algo1_step3_restored_x2[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=1: {metric} does not match expected value "
                            f"{ENV_STEPS_PER_ITERATION * 3}",
                        )
        return {
            "env_runners": {
                0: {
                    "step_1": result_algo_0_step1[ENV_RUNNER_RESULTS],
                    "step_2": result_algo_0_step2[ENV_RUNNER_RESULTS],
                    "step_3": result_algo_0_step3[ENV_RUNNER_RESULTS],
                },
                1: {
                    "step_1": result_algo_1_step1[ENV_RUNNER_RESULTS],
                    "step_2": result_algo_1_step2[ENV_RUNNER_RESULTS],
                    "step_3": result_algo_1_step3[ENV_RUNNER_RESULTS],
                },
            }
        }

    @unittest.skip("Needs to be fixed in ray first")
    def test_trainable_checkpointing(self):
        """Test if trainable can be checkpointed and restored."""
        with patch_args(
            "--batch_size",
            str(ENV_STEPS_PER_ITERATION),
        ):
            setup = AlgorithmSetup(init_trainable=False)
            config = setup.config
            config.debugging(seed=11)
            config.environment(env="CartPole-v1")
            config.training(
                train_batch_size_per_learner=ENV_STEPS_PER_ITERATION,
                num_epochs=2,
                minibatch_size=ENV_STEPS_PER_ITERATION // 2,
            )
            config.env_runners(num_env_runners=0)
            trainable0 = setup.create_trainable()
            setup = AlgorithmSetup(init_trainable=False)
            config = setup.config
            config.debugging(seed=11)
            config.environment(env="CartPole-v1")
            config.training(
                train_batch_size_per_learner=ENV_STEPS_PER_ITERATION,
                num_epochs=2,
                minibatch_size=ENV_STEPS_PER_ITERATION // 2,
            )
            config.env_runners(num_env_runners=1)
            trainable1 = setup.create_trainable()
        results = self._test_algo_checkpointing(
            trainable0(),
            trainable1(),
            metrics=[
                NUM_ENV_STEPS_SAMPLED_LIFETIME,
                NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
            ],
        )

        self._test_checkpoint_values(
            results["env_runners"][0]["step_3"],
            results["env_runners"][1]["step_3"],
        )

    @unittest.skip("Needs to be fixed in ray first")
    def test_algorithm_checkpointing(self):
        # similar to test_trainable_checkpointing, but pure algorithms
        with patch_args(
            "--batch_size",
            str(ENV_STEPS_PER_ITERATION),
        ):
            setup = AlgorithmSetup(init_trainable=False)
        config = setup.config
        config.debugging(seed=11)
        config.environment(env="CartPole-v1")
        config.training(
            train_batch_size_per_learner=ENV_STEPS_PER_ITERATION,
            num_epochs=2,
            minibatch_size=ENV_STEPS_PER_ITERATION // 2,
        )
        algo1 = config.env_runners(num_env_runners=1).build_algo()
        algo0 = config.env_runners(num_env_runners=0).build_algo()
        results = self._test_algo_checkpointing(
            algo0,
            algo1,
            metrics=[
                # NUM_ENV_STEPS_SAMPLED_LIFETIME,
                NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
            ],
        )

        self._test_checkpoint_values(
            results["env_runners"][0]["step_3"],
            results["env_runners"][1]["step_3"],
        )
        # because _throughput values results are not equal in structure (only after 2 steps)
        # self.set_max_diff(40000)

        # breakpoint()
        # self.assertDictEqual(results["env_runners"][0]["step_3"], results["env_runners"][1]["step_3"])


if __name__ == "__main__":
    import unittest

    unittest.main(defaultTest="TestAlgorithm.test_checkpointing_native2")
