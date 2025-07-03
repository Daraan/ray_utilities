from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import TYPE_CHECKING

import ray
import tree
import typing_extensions as te
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger

try:
    from .utils import DisableBreakpointsForGUI, SetupDefaults, patch_args
except ImportError:
    if not TYPE_CHECKING:
        from utils import DisableBreakpointsForGUI, SetupDefaults, patch_args  # type: ignore
    else:
        from .utils import DisableBreakpointsForGUI, SetupDefaults, patch_args


if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


# os.environ["RAY_DEBUG"]="0"


class TestSetupClasses(SetupDefaults):
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
                    return {"evaluation/env_runners/episode_return_mean": 42, "param_value": params[param]}

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


ENV_STEPS_PER_ITERATION = 10


class TestAlgorithm(DisableBreakpointsForGUI, SetupDefaults):
    @classmethod
    def setUpClass(cls):
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        ray.shutdown()


    def _test_checkpoint_values(self, result1, result2):
        env_runner1 = result1
        env_runner2 = result2
        # This step - trivial tests
        self.assertEqual(env_runner1[NUM_ENV_STEPS_SAMPLED], env_runner2[NUM_ENV_STEPS_SAMPLED])
        self.assertEqual(env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER], env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER])

        # Lifetime stats:
        self.assertEqual(
            env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME], env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME]
        )
        self.assertEqual(env_runner1[NUM_ENV_STEPS_SAMPLED_LIFETIME], env_runner2[NUM_ENV_STEPS_SAMPLED_LIFETIME])
        # This would be amazing, but does not look possible:

    #        self.assertEqual(env_runner1[EPISODE_RETURN_MEAN], env_runner2[EPISODE_RETURN_MEAN])

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
        self._test_algo_checkpointing(
            algo_0_runner,
            algo_1_runner,
            metrics=[NUM_ENV_STEPS_SAMPLED_LIFETIME, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
        )

    def _test_algo_checkpointing(self, algo_0_runner: Algorithm, algo_1_runner: Algorithm, metrics: list[str]):
        assert algo_0_runner.config and algo_1_runner.config and algo_0_runner.metrics and algo_1_runner.metrics
        self.assertEqual(algo_0_runner.config.num_env_runners, 0)
        self.assertEqual(algo_1_runner.config.num_env_runners, 1)
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
            algo_0_runner_restored = PPO.from_checkpoint(checkpoint_0_step1)
            algo_1_runner_restored = PPO.from_checkpoint(checkpoint_1_step1)
            assert algo_0_runner_restored.metrics and algo_1_runner_restored.metrics
            # Check loaded metric
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after restored step 1", metric=metric):
                    self.assertEqual(
                        algo_0_runner_restored.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION,
                    )
                    self.assertEqual(
                        algo_1_runner_restored.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION,
                    )
            tree.assert_same_structure(algo_0_runner_restored.metrics, algo_1_runner_restored.metrics)

            # --- Step 2 from restored & checkpoint ---
            result_algo0_step2_restored = algo_0_runner_restored.step()
            result_algo1_step2_restored = algo_1_runner_restored.step()
            # Check if metric was updated
            for metric in metrics:
                with self.subTest(f"(Checkpointed) Check {metric} after restored step 2", metric=metric):
                    self.assertEqual(
                        algo_0_runner_restored.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                    )
                    self.assertEqual(
                        algo_1_runner_restored.metrics.peek((ENV_RUNNER_RESULTS, metric)),
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
                            f"Restored num_env_runners=0: {metric} does not match expected value {ENV_STEPS_PER_ITERATION * 2}",
                        )
                    with self.subTest("From checkpoint: env_runners=1 - Step 2"):
                        self.assertEqual(
                            result_algo_1_step2[ENV_RUNNER_RESULTS][metric],
                            result_algo1_step2_restored[ENV_RUNNER_RESULTS][metric],
                            f"Restored num_env_runners=1: {metric} does not match expected value {ENV_STEPS_PER_ITERATION * 2}",
                        )
            # Test after restoring a second time
            # Load Restored from step 2
            algo_0_restored_x2 = PPO.from_checkpoint(checkpoint_0_step2_restored)
            algo_1_restored_x2 = PPO.from_checkpoint(checkpoint_1_step2_restored)
            assert algo_0_restored_x2.metrics and algo_1_restored_x2.metrics
            for metric in metrics:
                with self.subTest(f"(Checkpointed x2) Check {metric} after step 2", metric=metric):
                    self.assertEqual(
                        algo_0_restored_x2.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                        f"Restored x2 num_env_runners=0: {metric} does not match expected value {ENV_STEPS_PER_ITERATION * 2}",
                    )
                    self.assertEqual(
                        algo_1_restored_x2.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                        f"Restored x2 num_env_runners=1: {metric} does not match expected value {ENV_STEPS_PER_ITERATION * 2}",
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
                            f"Restored num_env_runners=1: {metric} does not match expected value {ENV_STEPS_PER_ITERATION * 3}",
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

    def test_checkpointing(self):
        print("start")
        path = os.path.dirname(__file__)
        print("path is", path)
        with patch_args(
            "--batch_size",
            str(ENV_STEPS_PER_ITERATION),
        ):
            setup = AlgorithmSetup()
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
        # self.assertDictEqual(results["env_runners"][0]["step_3"], results["env_runners"][1]["step_3"])


if __name__ == "__main__":
    import unittest

    unittest.main(defaultTest="TestAlgorithm.test_checkpointing_native2")
