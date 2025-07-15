from __future__ import annotations

# pyright: reportOptionalMemberAccess=none
import argparse
import os
import tempfile
import time
import unittest
from typing import TYPE_CHECKING, Any, cast
from inspect import isclass

import tree
import typing_extensions as te
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger
from ray_utilities.testing_utils import DisableGUIBreakpoints, InitRay, SetupDefaults, patch_args
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.algorithms import Algorithm
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
        self.maxDiff = 15000
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

    def test_config_overrides(self):
        with patch_args("--batch_size", "1234", "--minibatch_size", "444"):
            # test with init_trainable=False
            setup = AlgorithmSetup(init_trainable=False)
            setup.config.training(num_epochs=3, minibatch_size=321)
            Trainable = setup.create_trainable()
            assert isinstance(Trainable, type) and issubclass(Trainable, DefaultTrainable)
            trainable = Trainable({})

        self.assertEqual(trainable.algorithm_config.num_epochs, 3)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 1234)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 321)

        self.maxDiff = 15000
        self.assertIsNot(trainable.algorithm_config, setup.config)
        self.compare_configs(
            trainable.algorithm_config,
            setup.config,
            ignore=[
                # ignore callbacks that are created on Trainable.setup
                "callbacks_on_environment_created",
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
            self.maxDiff = 38000
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

    @unittest.skip("Fix first: Ray moves checkpoint need to load from different location")
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
                print("iteration", i, "Sampled Lifetime", result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME])
                time.sleep(0.2)
                # algo.save_checkpoint(tempdir1)
                if i == 2:
                    tune.report(
                        {"evaluation/env_runners/episode_return_mean": i} | result,
                        checkpoint=tune.Checkpoint.from_directory(tempdir1),
                    )
                else:
                    tune.report(
                        {"evaluation/env_runners/episode_return_mean": i} | result,
                        checkpoint=None,
                    )

        setup.trainable = fake_trainable  # type: ignore
        tuner = setup.create_tuner()
        # With stopper should only iterate 4 times:
        assert tuner._local_tuner
        tuner._local_tuner._run_config.stop = {NUM_ENV_STEPS_SAMPLED_LIFETIME: 512}
        with tempfile.TemporaryDirectory(prefix=".ckpt_") as tempdir1:
            result_grid = tuner.fit()
            time.sleep(1)
            print("Training ended.")
            checkpoint = result_grid[0].checkpoint
            assert checkpoint
            # Problem ray moves checkpoint to new location
            tempdir = checkpoint.to_directory(tempdir1)
            algo_restored = config.build_algo().from_checkpoint(tempdir)
            assert algo_restored.metrics
            self.assertNotEqual(
                algo_restored.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)), 0
            )

    def clean_timer_logs(self, result: dict):
        """
        Cleans the timer logs from the env_runners_dict.
        This is useful to compare results without the timer logs.
        """
        env_runners_dict = result["env_runners"]
        for key in list(result.keys()):
            if key.startswith("timers/"):
                del env_runners_dict[key]
        del env_runners_dict["module_to_env_connector"]
        del env_runners_dict["env_to_module_connector"]
        del env_runners_dict["episode_duration_sec_mean"]
        del result["env_runners"]["env_reset_timer"]
        del result["env_runners"]["env_step_timer"]
        del result["env_runners"]["rlmodule_inference_timer"]
        del result["env_runners"]["sample"]
        env_runners_dict.pop("num_env_steps_sampled_lifetime_throughput", None)
        # result["env_runners"]["time_between_sampling"]
        return result

    def _test_checkpoint_values(self, result1: dict[str, Any], result2: dict[str, Any]):
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
            self.compare_env_runner_configs(algo_0_runner, algo_0_runner)
            self.compare_env_runner_configs(algo_1_runner, algo_1_runner)

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
            algo_0_restored_x2 = PPO.from_checkpoint(checkpoint_0_step2_restored)
            algo_1_restored_x2 = PPO.from_checkpoint(checkpoint_1_step2_restored)
            assert algo_0_restored_x2.metrics and algo_1_restored_x2.metrics
            for metric in metrics:
                with self.subTest(f"(Checkpointed x2) Check {metric} after step 2", metric=metric):
                    self.assertEqual(
                        algo_0_restored_x2.metrics.peek((ENV_RUNNER_RESULTS, metric)),
                        ENV_STEPS_PER_ITERATION * 2,
                        f"Restored x2 num_env_runners=0: {metric} does not match expected value "
                        f"{ENV_STEPS_PER_ITERATION * 2}",
                    )
                    self.assertEqual(
                        algo_1_restored_x2.metrics.peek((ENV_RUNNER_RESULTS, metric)),
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
        # self.maxDiff = 40_000
        # breakpoint()
        # self.assertDictEqual(results["env_runners"][0]["step_3"], results["env_runners"][1]["step_3"])


if __name__ == "__main__":
    import unittest

    unittest.main(defaultTest="TestAlgorithm.test_checkpointing_native2")
