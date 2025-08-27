# pyright: reportOptionalMemberAccess=information
from __future__ import annotations

import argparse
import io
import os
import subprocess
from ast import literal_eval

import pyarrow as pa
import pytest

from ray_utilities.callbacks.algorithm.seeded_env_callback import NUM_ENV_RUNNERS_0_1_EQUAL
from ray_utilities.config.model_config_parsers import MLPConfigParser
from ray_utilities.misc import raise_tune_errors
from ray_utilities.runfiles import run_tune
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

os.environ["RAY_DEBUG"] = "legacy"
# os.environ["RAY_DEBUG"]="0"

import tempfile
import unittest
import unittest.mock
from copy import deepcopy
from inspect import isclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, Optional, cast

import cloudpickle
import tree
import typing_extensions as te
from ray import tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.config import logger as parser_logger
from ray_utilities.constants import (
    EVAL_METRIC_RETURN_MEAN,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.random import seed_everything
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger
from ray_utilities.testing_utils import (
    TWO_ENV_RUNNER_CASES,
    Cases,
    DisableGUIBreakpoints,
    InitRay,
    SetupDefaults,
    SetupWithCheck,
    TrainableWithChecks,
    iter_cases,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable, TrainableBase

if TYPE_CHECKING:
    import numpy as np
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict


class TestSetupClasses(InitRay, SetupDefaults, num_cpus=4):
    @pytest.mark.basic
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

    @pytest.mark.basic
    def test_argument_usage(self):
        # Test warning and failure
        with patch_args("--batch_size", "1234"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 1234)
        with patch_args("--train_batch_size_per_learner", "456"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 456)

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
        trainable = setup.trainable()
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

    def test_context_manager(self):
        with patch_args():
            for setup in [
                AlgorithmSetup(init_trainable=False, init_param_space=False),
                AlgorithmSetup(init_trainable=True, init_param_space=True),
            ]:
                # Check unset
                with setup:
                    self.assertTrue(not hasattr(setup, "trainable") or setup.trainable is None)
                    self.assertTrue(not hasattr(setup, "param_space") or setup.param_space is None)
                    setup.config.training(num_epochs=4, minibatch_size=222)
                self.assertIsNotNone(setup.trainable)
                self.assertIsNotNone(setup.param_space)
                if isclass(setup.trainable):
                    trainable = setup.trainable_class()
                    self.assertEqual(trainable.algorithm_config.num_epochs, 4)
                    self.assertEqual(trainable.algorithm_config.minibatch_size, 222)

    def test_project_name_substitution(self):
        setup = AlgorithmSetup(init_trainable=False, init_param_space=False, init_config=False)
        setup.PROJECT = "Test-<agent_type>-<env_type>"
        self.assertEqual(setup.project_name.rstrip("-v0123456789"), "Test-mlp-CartPole")

    @pytest.mark.tuner
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
        type_hints = te.get_type_hints(DefaultArgumentParser)["tune"]
        self.assertIs(te.get_origin(type_hints), te.Union)
        th_args = te.get_args(type_hints)
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
                    "--num_jobs", "4",
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

    def test_dynamic_param_space_with_trainable(self):
        """Check the --tune parameters"""
        type_hints = te.get_type_hints(DefaultArgumentParser)["tune"]
        self.assertIs(te.get_origin(type_hints), te.Union)
        th_args = te.get_args(type_hints)
        th_lists = [
            literal
            for li in [te.get_args(arg)[0] for arg in th_args if te.get_origin(arg) is list]
            for literal in te.get_args(li)
            if literal != "all"
        ]
        self.assertIn("rollout_size", th_lists)
        self.assertNotIn("all", th_lists)

        for param in th_lists:
            # run 3 jobs in parallel
            with (
                patch_args(
                    "--tune", param,
                    "--num_jobs", "3",
                    "-it", "2",
                    "--num_samples", "3",
                    "--use_exact_total_steps",
                    "--env_seeding_strategy", "same"
                )  # ,
                # self.assertNoLogs(logger, level="WARNING"),
            ):  # fmt: skip
                if param == "batch_size":  # shortcut name
                    param = "train_batch_size_per_learner"  # noqa: PLW2901

                class TrainableWithChecksB(TrainableWithChecks):
                    debug_step = False
                    debug_setup = False
                    _param_name = param

                    def setup_check(self, config: dict[str, Any], algorithm_overrides=None):
                        self._param_to_check = config[self._param_name]
                        if hasattr(self.algorithm_config, self._param_name):
                            assert getattr(self.algorithm_config, self._param_name) == self._param_to_check, (
                                f"Expected {self._param_to_check}, "
                                f"but got {getattr(self.algorithm_config, self._param_name)}"
                            )

                    def step_post_check(self, result: StrictAlgorithmReturnData, metrics: LogMetricsDict, rewards: Any):
                        if self._param_name == "train_batch_size_per_learner":
                            assert (
                                result[ENV_RUNNER_RESULTS].get(NUM_ENV_STEPS_PASSED_TO_LEARNER, None)
                                == self._param_to_check
                            )
                            assert (
                                result["learners"]["__all_modules__"].get(NUM_ENV_STEPS_PASSED_TO_LEARNER, None)
                                == self._param_to_check
                            )
                        metrics["param_name"] = self._param_name  # pyright: ignore[reportGeneralTypeIssues]
                        metrics["param_value"] = self._param_to_check  # pyright: ignore[reportGeneralTypeIssues]

                Setup = SetupWithCheck(TrainableWithChecksB)
                # Limit for performance
                Setup.batch_size_sample_space = {"grid_search": [16, 64, 128]}
                Setup.rollout_size_sample_space = {"grid_search": [16, 64, 128]}

                with Setup() as setup:
                    setup.config.minibatch_size = 8  # set to small value to prevent ValueErrors
                param_space = setup.param_space
                self.assertIn(param, param_space)
                self.assertIsNotNone(param_space[param])  # dict with list
                if "grid_search" in param_space[param]:
                    grid = param_space[param]["grid_search"]
                else:
                    grid = []
                tuner = setup.create_tuner()
                tuner._local_tuner.get_run_config().checkpoint_config.checkpoint_at_end = False  # pyright: ignore[reportOptionalMemberAccess]
                results = tuner.fit()
                raise_tune_errors(results)
                self.check_tune_result(results)
                assert results[0].metrics
                self.assertIn(
                    "_checking_class_",
                    results[0].metrics,
                    "Metrics should contain '_checking_class_'. Custom class was likely not used",
                )
                self.assertEqual(results[0].metrics["param_name"], param)
                choices = {r.metrics["param_value"] for r in results}  # pyright: ignore[reportOptionalSubscript]
                self.assertEqual(len(choices), results.num_terminated)
                if grid:
                    self.assertLessEqual(choices, set(grid), f"Choices {choices} not in grid {grid}")

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
                algorithm_overrides=AlgorithmConfig.overrides(
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

    def test_args_config_after_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_size1: Final[int] = 40
            default_timeout: Final[float] = self._DEFAULT_SETUP.config.evaluation_sample_timeout_s  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
            eval_unit_default = self._DEFAULT_SETUP.config.evaluation_duration_unit
            eval_unit_other = "episodes" if eval_unit_default == "timesteps" else "timesteps"
            assert eval_unit_default != eval_unit_other
            with patch_args(
                "--total_steps", "80",
                "--batch_size", batch_size1,
                "--use_exact_total_steps",  # Do not adjust total_steps
                "--minibatch_size", "20",
                "--comment", "A",
                "--extra", "abc",  # nargs
                "--env_seeding_strategy", "constant",
                "--wandb", "offline",  # possibly do not restore
            ):  # fmt: skip
                with AlgorithmSetup(init_trainable=False) as setup:
                    # These depend on args and CANNOT be restored!
                    setup.config.training(minibatch_size=10, num_epochs=2)
                    # Use non-default value
                    setup.config.evaluation(
                        evaluation_sample_timeout_s=default_timeout + 120,
                        evaluation_duration_unit=eval_unit_other,  # check that this is restored from config
                    )

                self.assertEqual(setup.config.train_batch_size_per_learner, setup.args.train_batch_size_per_learner)
                self.assertEqual(setup.config.train_batch_size_per_learner, 40)  # batch_size1

                self.assertNotEqual(setup.config.minibatch_size, setup.args.minibatch_size)
                self.assertEqual(setup.config.minibatch_size, 10)
                self.assertEqual(setup.args.total_steps, 80)
                self.assertEqual(setup.args.comment, "A")
                self.assertEqual(setup.args.extra, ["abc"])  # nargs
                self.assertEqual(setup.args.env_seeding_strategy, "constant")
                self.assertEqual(setup.args.wandb, "offline")

                setup_state = setup.get_state()
                # Check saved state
                self.assertEqual(setup_state["config"].minibatch_size, 10)
                self.assertDictEqual(
                    setup_state["config_overrides"],
                    {
                        "num_epochs": 2,
                        "evaluation_duration_unit": eval_unit_other,
                        "evaluation_sample_timeout_s": default_timeout + 120,
                        "minibatch_size": 10,
                    },
                )
                # save state:
                filename = Path(tmpdir) / "state.pkl"
                filesystem, path = pa.fs.FileSystem.from_uri(tmpdir)  # pyright: ignore[reportAttributeAccessIssue]
                with filesystem.open_output_stream(filename.as_posix()) as f:
                    state = {"setup": setup_state}
                    cloudpickle.dump(state, f)
            del setup  # avoid mistakes

            with patch_args(
                "--total_steps", 80 + 120,
                "--batch_size", "60",
                "--use_exact_total_steps",  # Do not adjust total_steps
                "--comment", "B",
                "--from_checkpoint", tmpdir,
                "--env_seeding_strategy", "sequential",  # default value
            ):  # fmt: skip
                # Test if setup1.args are merged with setup2.args
                loaded_config, *_ = AlgorithmSetup._config_from_checkpoint(tmpdir)
                self.assertEqual(loaded_config.train_batch_size_per_learner, batch_size1)
                self.assertEqual(loaded_config.minibatch_size, 10)
                with AlgorithmSetup(init_trainable=False) as setup2:
                    setup2.config.training(num_epochs=5)
                    # Use default value
                    setup2.config.evaluation(evaluation_sample_timeout_s=default_timeout)
                self.assertDictEqual(
                    setup2.get_state()["config_overrides"],
                    {
                        # NOT restored override
                        # "minibatch_size": 10,
                        # "evaluation_duration_unit": eval_unit_other,
                        # new overrides
                        "evaluation_sample_timeout_s": default_timeout,
                        "num_epochs": 5,
                    },
                )
                # Changed values
                self.assertEqual(setup2.config.train_batch_size_per_learner, setup2.args.train_batch_size_per_learner)
                self.assertEqual(setup2.config.train_batch_size_per_learner, 60)
                self.assertEqual(setup2.args.comment, "B")
                self.assertEqual(setup2.args.total_steps, 80 + 120)
                self.assertEqual(setup2.args.env_seeding_strategy, "sequential")
                self.assertEqual(setup2.config.num_epochs, 5)
                # restored from 1st
                self.assertEqual(setup2.args.extra, ["abc"])
                # NeverRestore; wandb is the defaul value again
                self.assertEqual(setup2.args.wandb, DefaultArgumentParser.wandb)
                # Changed manually
                self.assertEqual(setup2.config.evaluation_sample_timeout_s, default_timeout)

                # Attention: These are NOT the overrides as values are explicitly overwritten by config_from_args
                self.assertEqual(setup2.config.minibatch_size, 20)
                self.assertNotEqual(setup2.config.evaluation_duration_unit, eval_unit_other)
                self.assertEqual(setup2.config.evaluation_duration_unit, eval_unit_default)

    def test_parser_restore_annotations(self):
        with patch_args(
            "--batch_size", "1234",
            "--num_jobs", DefaultArgumentParser.num_jobs + 2,  # NeverRestore
            "--log_level", "DEBUG",  # NeverRestore
            "--env_type", "cart",  # AlwaysRestore
            "--actor_type", "mlp",
        ):  # fmt: skip
            setup = AlgorithmSetup()
            self.assertEqual(setup.args.log_level, "DEBUG")
            Trainable = setup.create_trainable()
        assert isclass(Trainable)
        trainable = Trainable()
        assert trainable.algorithm_config.minibatch_size is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            trainable.save_to_path(tmpdir)
            with patch_args(
                "--from_checkpoint", tmpdir,
                "-a", "a new value",
            ):  # fmt: skip
                with self.assertLogs(
                    parser_logger,
                    "WARNING",
                ) as context:
                    AlgorithmSetup()
                self.assertIn(
                    "Restoring AlwaysRestore argument 'agent_type' from checkpoint: "
                    "replacing a new value (explicitly passed) with mlp",
                    context.output[0],
                    msg=context.output,
                )
            with patch_args(
                "--from_checkpoint", tmpdir,
                "--minibatch_size", trainable.algorithm_config.minibatch_size * 2,
                "--seed", (trainable._setup.config.seed or 1234) * 2
            ):  # fmt: skip
                setup2 = AlgorithmSetup()
                # Not annotated value, restored
                self.assertEqual(setup2.config.train_batch_size_per_learner, 1234)
                # Never restored
                self.assertNotEqual(setup2.args.log_level, "DEBUG")
                self.assertEqual(setup2.args.num_jobs, DefaultArgumentParser.num_jobs)

    @unittest.mock.patch.object(subprocess, "Popen", autospec=True)
    def test_wandb_upload(self, mock_run: unittest.mock.MagicMock):
        # NOTE: This test is flaky, there are instances of no wandb folder copied
        # Does the actor die silently?
        self.no_pbar_updates()

        class MockPopen(unittest.mock.MagicMock):
            returncode = 1
            stdout: IO[bytes] = io.BytesIO(b"MOCK: wandb: Syncing files...")
            stderr: IO[bytes] | None = io.BytesIO(b"MOCK: stderr - its expected you see this message")

            def poll(self) -> None:
                return None

        mocked_popen = MockPopen()
        mock_run.return_value = mocked_popen
        # use more iterations here for it to be more likely that the files are synced.
        with patch_args("--wandb", "offline+upload", "--num_jobs", 1, "--iterations", 5, "--batch_size", 32):
            setup = AlgorithmSetup()
            _results = run_tune(setup, raise_errors=True)
        mocked_popen.wait.assert_called_once()
        self.assertDictEqual(mock_run.call_args.kwargs, {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT})

    def test_seeded_env(self):
        with patch_args("--seed", "1234", "--num_env_runners", 2), AlgorithmSetup(init_trainable=False) as setup:
            # NOTE: if async the np_random generator is changed my gymnasium
            setup.config.env_runners(gym_env_vectorize_mode="SYNC")
        trainable = setup.trainable_class({"env_seed": 2222})
        assert trainable.algorithm_config.num_envs_per_env_runner is not None
        num_envs = trainable.algorithm_config.num_envs_per_env_runner

        def check_np_random_seed(runner: SingleAgentEnvRunner | Any):
            return runner.env.np_random_seed == (-1,) * num_envs

        def check_np_random_generator(runner: SingleAgentEnvRunner | Any):
            rngs: list[np.random.Generator] = runner.env.np_random
            return all(
                rng.bit_generator.seed_seq.spawn_key[:-1]  # pyright: ignore[reportAttributeAccessIssue]
                == (1 if runner.worker_index == 0 and NUM_ENV_RUNNERS_0_1_EQUAL else runner.worker_index, 0, False)
                for rng in rngs
            ) and all(rng.bit_generator.seed_seq.entropy == 2222 for rng in rngs)  # pyright: ignore[reportAttributeAccessIssue]

        if setup.config.num_env_runners == 0:
            self.assertTrue(check_np_random_seed(trainable.algorithm.env_runner))
            # when async these are not equal to the ones from the callback, but still based on them
            self.assertTrue(check_np_random_generator(trainable.algorithm.env_runner))
            logged_seed = trainable.algorithm.env_runner.metrics.peek(
                ("environments", "seeds", "seed_sequence"), compile=False
            )
        else:
            # Cannot pickle generators => cannot pickle envs
            assert trainable.algorithm.env_runner_group
            self.assertTrue(
                all(
                    trainable.algorithm.env_runner_group.foreach_env_runner(
                        check_np_random_seed, local_env_runner=False
                    )
                )
            )
            self.assertTrue(
                all(
                    trainable.algorithm.env_runner_group.foreach_env_runner(
                        check_np_random_generator, local_env_runner=False
                    )
                )
            )
            logged_seeds = trainable.algorithm.env_runner_group.foreach_env_runner(
                lambda r: r.metrics.peek(("environments", "seeds", "seed_sequence")), local_env_runner=False
            )
            self.assertEqual(len(logged_seeds), setup.config.num_env_runners)
            # Assert that the deques in logged_seeds are pairwise different
            for i in range(len(logged_seeds)):
                for j in range(i + 1, len(logged_seeds)):
                    self.assertNotEqual(
                        logged_seeds[i], logged_seeds[j], f"Deque at index {i} is equal to deque at index {j}"
                    )


class TestPPOMLPSetup(InitRay, num_cpus=4):
    def test_basic(self):
        with patch_args():
            setup = PPOMLPSetup()
        self.assertIsNotNone(setup.config)
        self.assertIsNotNone(setup.args)
        self.assertIsNotNone(setup.create_tuner())
        self.assertIsNotNone(setup.create_config())
        self.assertIsNotNone(setup.create_param_space())
        self.assertIsNotNone(setup.create_parser())
        self.assertIsNotNone(setup.create_tags())
        self.assertIsNotNone(setup.create_trainable())

    def test_model_config_dict(self):
        with patch_args():
            setup = PPOMLPSetup()
        model_config = setup._model_config_from_args(setup.args)
        assert model_config, f"Not truthy {model_config}"
        for k, v in MLPConfigParser().parse_args([]).as_dict().items():
            self.assertIn(k, model_config)
            self.assertEqual(v, model_config[k])

    @Cases([[], ["--fcnet_hiddens", "[16, 32, 64]"]])
    def test_layers(self, cases):
        import torch  # noqa: PLC0415  # import lazy

        for args in iter_cases(cases):
            with patch_args(*args):
                setup = PPOMLPSetup()
            algo = setup.build_algo()
            module: DefaultPPOTorchRLModule = algo.get_module()  # pyright: ignore[reportAssignmentType]
            mlp_encoder: torch.nn.Sequential = module.encoder.encoder.net.mlp
            # Use default for empty args
            expected_layers = MLPConfigParser.fcnet_hiddens if not args else literal_eval(args[-1])
            self.assertEqual(len(mlp_encoder), 2 * len(expected_layers))
            size_iter = iter(expected_layers)
            for layer in mlp_encoder:
                if isinstance(layer, torch.nn.Linear):
                    self.assertEqual(layer.out_features, next(size_iter))


ENV_STEPS_PER_ITERATION = 20 * max(1, DefaultArgumentParser.num_envs_per_env_runner)


class TestAlgorithm(InitRay, DisableGUIBreakpoints, SetupDefaults, num_cpus=4):
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

    def test_train(self):
        self._DEFAULT_SETUP_LOW_RES.unset_trainable()  # we do not use Trainable here
        self._DEFAULT_SETUP_LOW_RES.config.evaluation(evaluation_interval=1)
        algo = self._DEFAULT_SETUP_LOW_RES.build_algo()
        algo.train()

    @pytest.mark.tuner
    def test_stopper_with_checkpoint(self):
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
            assert result_grid[0].metrics
            self.assertEqual(result_grid[0].metrics["training_iteration"], 4)
            self.assertEqual(result_grid[0].metrics["evaluation/env_runners/episode_return_mean"], 512 // 128 - 1)


class TestMetricsRestored(InitRay, DisableGUIBreakpoints, SetupDefaults, num_cpus=4):
    def _test_checkpoint_values(self, result1: dict[str, Any], result2: dict[str, Any], msg: Optional[str] = None):
        """Test NUM_ENV_STEPS_SAMPLED and NUM_ENV_STEPS_PASSED_TO_LEARNER values of two training results."""
        env_runner1 = result1
        env_runner2 = result2
        # This step - trivial tests
        # TODO: is this also checked with learner?
        self.assertEqual(
            env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER], env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER], msg
        )

        # Lifetime stats:
        self.assertEqual(
            env_runner1[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
            env_runner2[NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
            msg,
        )
        # self.assertEqual(env_runner1[NUM_ENV_STEPS_SAMPLED_LIFETIME], env_runner2[NUM_ENV_STEPS_SAMPLED_LIFETIME], msg)
        # This would be amazing, but does not look possible:
        # self.assertEqual(env_runner1[EPISODE_RETURN_MEAN], env_runner2[EPISODE_RETURN_MEAN])

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

    @Cases(TWO_ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    @pytest.mark.length("medium")
    def test_with_tuner(self, cases):
        """Test if key stats are restored correctly - does not test further training and metrics"""
        self.no_pbar_updates()
        frequency = 5
        num_checkpoints = 1
        setup_seed = 42
        cli_seed = 36
        # for both num_env_runners equal the results do match, but for (0, 1) ...
        # FIXME: For a low ENV_STEPS_PER_ITERATION, i.e. not completed episodes the comparison holds
        # for a higher amount of steps were episodes are completed training results DO NOT match anymore
        # maybe because of Async envs?
        expected_minibatch_size = ENV_STEPS_PER_ITERATION
        OTHER_MINI_BATCH_SIZE = 5
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", ENV_STEPS_PER_ITERATION,
            "--minibatch_size", expected_minibatch_size,
            "--iterations", str(frequency * num_checkpoints),
            "--seed", str(cli_seed),
        ):  # fmt: skip
            assert OTHER_MINI_BATCH_SIZE != ENV_STEPS_PER_ITERATION

            for num_env_runners_a, num_env_runners_b in iter_cases(cases):
                msg_prefix = f"num_env_runners=({num_env_runners_a} & {num_env_runners_b}) :"
                with self.subTest(
                    num_env_runners_a=num_env_runners_a, num_env_runners_b=num_env_runners_b, msg=msg_prefix
                ):
                    # cannot make this deterministic on local vs remote
                    seed_everything(None, setup_seed)
                    with AlgorithmSetup(init_trainable=False) as setup:
                        setup.config.env_runners(num_env_runners=num_env_runners_a)
                        setup.config.training(minibatch_size=OTHER_MINI_BATCH_SIZE)  # insert some noise
                        setup.config.debugging(seed=setup_seed)
                    setup.trainable_class.use_pbar = False
                    tuner_0 = setup.create_tuner()
                    seed_everything(None, setup_seed)
                    with AlgorithmSetup(init_trainable=False) as setup:
                        setup.config.env_runners(num_env_runners=num_env_runners_b)
                        setup.config.training(minibatch_size=OTHER_MINI_BATCH_SIZE)  # insert some noise
                        setup.config.debugging(seed=setup_seed)
                    setup.trainable_class.use_pbar = False
                    compare_dict = {"num_env_runners": num_env_runners_b, "minibatch_size": OTHER_MINI_BATCH_SIZE}
                    if num_env_runners_a == num_env_runners_b == DefaultArgumentParser.num_env_runners:
                        compare_dict.pop("num_env_runners")
                    self.assertDictEqual(
                        setup.config_overrides(),
                        compare_dict
                        | (
                            {"seed": setup_seed}
                            if setup_seed != cli_seed  # pyright: ignore[reportUnnecessaryComparison]
                            else {}
                        ),
                        msg_prefix,
                    )
                    self.assertEqual(setup.config.minibatch_size, 5)
                    tuner_1 = setup.create_tuner()
                    tune_results = {}
                    for num_env_runners, tuner in zip((num_env_runners_a, num_env_runners_b), [tuner_0, tuner_1]):
                        assert tuner._local_tuner
                        tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                            checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                            checkpoint_score_order="max",
                            checkpoint_frequency=frequency,  # Save every iteration
                            # NOTE: num_keep does not appear to work here
                        )
                        seed_everything(None, setup_seed)
                        result = tuner.fit()
                        self.check_tune_result(result)
                        checkpoint_dir, checkpoints = self.get_checkpoint_dirs(result[0])
                        self.assertEqual(
                            len(checkpoints),
                            num_checkpoints,
                            msg_prefix
                            + f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
                        )
                        # sort to get checkpoints in order 000, 001, 002
                        tune_results[num_env_runners] = {
                            "checkpoint_dir": checkpoint_dir,
                            "checkpoints": sorted(checkpoints),
                            "trainables": [],
                        }
                        for step, checkpoint in enumerate(sorted(checkpoints), 1):
                            self.assertTrue(os.path.exists(checkpoint))

                            Cls = DefaultTrainable.define(setup)
                            restored_trainable: DefaultTrainable[Any, AlgorithmConfig, Any] = Cls.from_checkpoint(
                                checkpoint
                            )  # pyright: ignore[reportAssignmentType]  # noqa: E501
                            # restore is bad if algorithm_checkpoint_dir is a temp dir
                            self.assertEqual(restored_trainable.algorithm.iteration, step * frequency)
                            self.assertEqual(restored_trainable.algorithm_config.seed, setup_seed)  # run seed
                            self.assertEqual(restored_trainable._setup.args.seed, cli_seed)

                            self.assertEqual(restored_trainable.algorithm_config.num_env_runners, num_env_runners)
                            self.assertEqual(restored_trainable.algorithm_config.minibatch_size, 5)

                            # NOTE: config overrides may not be applied to the setup with favors get_config_from_args!
                            # Adjust the tests if changing this behavior
                            self.assertEqual(restored_trainable._setup.config.minibatch_size, expected_minibatch_size)
                            self.assertEqual(restored_trainable._setup.config.num_env_runners, 0)  # not updated

                            tune_results[num_env_runners]["trainables"].append(restored_trainable)
                    self.assertGreater(len(tune_results[num_env_runners_a]["trainables"]), 0)
                    try:
                        for step in range(len(tune_results[num_env_runners_a]["trainables"])):
                            trainable_0: DefaultTrainable = tune_results[num_env_runners_a]["trainables"][step]
                            trainable_1: DefaultTrainable = tune_results[num_env_runners_b]["trainables"][step]
                            assert trainable_0.algorithm.metrics and trainable_1.algorithm.metrics
                            metrics_0 = trainable_0.algorithm.metrics.reduce()
                            metrics_1 = trainable_1.algorithm.metrics.reduce()
                            self._test_checkpoint_values(
                                metrics_0[ENV_RUNNER_RESULTS], metrics_1[ENV_RUNNER_RESULTS], msg_prefix
                            )
                            self.assertEqual(
                                metrics_0[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                                (step + 1) * frequency * ENV_STEPS_PER_ITERATION,
                                msg_prefix + "steps count in learner it not equal",
                            )
                            self.assertEqual(
                                metrics_0[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                                metrics_1[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                                msg_prefix + "steps count in learner it not equal",
                            )
                            self.assertEqual(
                                metrics_0[LEARNER_RESULTS][ALL_MODULES][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                                (step + 1) * frequency * ENV_STEPS_PER_ITERATION,
                                msg_prefix + "steps count in learner it not equal",
                            )
                            metrics_0_clean = self.clean_timer_logs(metrics_0)
                            metrics_1_clean = self.clean_timer_logs(metrics_1)
                            self.compare_env_runner_results(
                                metrics_0_clean[ENV_RUNNER_RESULTS],
                                metrics_1_clean[ENV_RUNNER_RESULTS],
                                msg_prefix + f"training results do not match at step {(step + 1) * frequency}",
                                strict=False,
                                # TODO: Need False but why, especially for (0, 1) why are the results not equal?
                                compare_results=False,  # worked for step == 0 on small total_steps;  will fail on higher restores, why? Runner resets env > 1?
                                compare_steps_sampled=False,
                                seed_subset_ok=(  # 1 vs 2 will be A vs A B in the seed sequence
                                    max(num_env_runners_a, num_env_runners_b) > 1
                                    and num_env_runners_a != num_env_runners_b
                                ),
                            )
                            # Results differ greatly for evaluation
                            # same sampling not enforced, might be the reason?
                        if False:
                            with self.subTest("Compare evaluation results at step", step=step + 1):  # pyright: ignore[reportPossiblyUnboundVariable]
                                self.compare_env_runner_results(
                                    metrics_0_clean[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],  # pyright: ignore[reportPossiblyUnboundVariable]
                                    metrics_1_clean[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],  # pyright: ignore[reportPossiblyUnboundVariable]
                                    f"evaluation results do not match at from step {(step + 1) * frequency}",  # pyright: ignore[reportPossiblyUnboundVariable]
                                    strict=False,
                                    compare_results=False,
                                )
                    finally:
                        for step in range(len(tune_results[num_env_runners_a]["trainables"])):
                            tune_results[num_env_runners_a]["trainables"][step].stop()
                            tune_results[num_env_runners_b]["trainables"][step].stop()

    def _test_algo_checkpointing(
        self,
        algo_0_runner: Algorithm | TrainableBase,
        algo_1_runner: Algorithm | TrainableBase,
        metrics: list[str],
        num_env_runners_expected: tuple[int, int] = (0, 1),
        msg: Optional[str] = None,
        *,
        stop_trainable: bool = True,
    ) -> dict[str, dict[int, dict[str, Any]]]:
        eval_metrics = [m for m in metrics if "learner" not in m]  # strip learner metrics
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

        self.assertEqual(algorithm_config_0.num_env_runners, num_env_runners_expected[0])
        self.assertEqual(algorithm_config_1.num_env_runners, num_env_runners_expected[1])
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
            # Train until step 3 - get results of uninterrupted training, check if expected
            # Save Step 1
            algo_0_runner.save_checkpoint(checkpoint_0_step1)
            algo_1_runner.save_checkpoint(checkpoint_1_step1)
            # --- Step 2 ---
            result_algo_0_step2 = algo_0_runner.step()
            result_algo_1_step2 = algo_1_runner.step()
            self.compare_metrics_in_results(
                result_algo_0_step2[ENV_RUNNER_RESULTS],
                result_algo_1_step2[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2,
                metrics,
                "Check {} after step 2",
            )
            self.compare_metrics_in_results(
                result_algo_0_step2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_1_step2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2 * 10,
                eval_metrics,
                "Evaluation: Check {} after step 2",
            )
            # Save Step 2
            algo_0_runner.save_checkpoint(checkpoint_0_step2)
            algo_1_runner.save_checkpoint(checkpoint_1_step2)

            # --- Step 3 ---
            result_algo_0_step3 = algo_0_runner.step()
            result_algo_1_step3 = algo_1_runner.step()
            self.compare_metrics_in_results(
                result_algo_0_step3[ENV_RUNNER_RESULTS],
                result_algo_1_step3[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3,
                metrics,
                "Check {} after step 3",
            )
            self.compare_metrics_in_results(
                result_algo_0_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_1_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3 * 10,
                eval_metrics,
                "Evaluation: Check {} after step 3",
            )
            if stop_trainable:
                # Stop trainable to free resources
                algo_0_runner.stop()
                algo_1_runner.stop()

            # Load and train from checkpoint

            # Load Step 1
            algo_0_runner_restored: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_0_step1)  # pyright: ignore[reportAssignmentType]
            algo_1_runner_restored: Algorithm | TrainableBase = BaseClass.from_checkpoint(checkpoint_1_step1)  # pyright: ignore[reportAssignmentType]
            if isinstance(algo_0_runner_restored, Algorithm):
                metrics_0_restored = algo_0_runner_restored.metrics
            else:
                metrics_0_restored = algo_0_runner_restored.algorithm.metrics
            assert metrics_0_restored is not None

            if isinstance(algo_1_runner_restored, Algorithm):
                metrics_1_restored = algo_1_runner_restored.metrics
            else:
                metrics_1_restored = algo_1_runner_restored.algorithm.metrics
            # Check loaded metric
            for metric in metrics:
                self.assertIn(metric, metrics_0_restored.stats[ENV_RUNNER_RESULTS])
                self.assertIn(metric, metrics_1_restored.stats[ENV_RUNNER_RESULTS])
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
            # Check that metrics was updated
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

            # Check results, compare with original results
            self.compare_metrics_in_results(
                result_algo_0_step2[ENV_RUNNER_RESULTS],
                result_algo0_step2_restored[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2,
                metrics,
                "(Checkpointed) Check {} after step 2, env_runners=0",
            )
            self.compare_metrics_in_results(
                result_algo_1_step2[ENV_RUNNER_RESULTS],
                result_algo1_step2_restored[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2,
                metrics,
                "(Checkpointed) Check {} after step 2, env_runners=1",
            )
            # Advanced: Compare all results # NOTE: This superseeds weaker metrics-only check
            self.compare_env_runner_results(
                result_algo_0_step2[ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
                result_algo0_step2_restored[ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
                compare_results=False,
                compare_steps_sampled=False,  # do not compare when num samples is
                msg=msg,
            )
            # Evaluation
            self.compare_metrics_in_results(
                result_algo_0_step2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo0_step2_restored[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed) Check {} after step 2, env_runners=0",
            )
            self.compare_metrics_in_results(
                result_algo_1_step2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo1_step2_restored[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 2 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed) Check {} after step 2, env_runners=1",
            )
            # Some evaluation metrics like: 'num_agent_steps_sampled_lifetime': {'default_agent': 200}
            # do not align, or num_env_steps_sampled_lifetime can be missing
            # This might still be a bug in ray that evaluation metrics are not restored
            # self.compare_env_runner_results(
            #    result_algo_0_step2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
            #    result_algo0_step2_restored[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
            #    compare_results=True,
            # )

            # Step 3 from restored
            result_algo_0_step3_restored = algo_0_runner_restored.step()
            result_algo_1_step3_restored = algo_1_runner_restored.step()
            algo_0_runner_restored.stop()
            algo_1_runner_restored.stop()

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
                self.assertIn(metric, metrics_0_restored_x2.stats[ENV_RUNNER_RESULTS])
                self.assertIn(metric, metrics_1_restored_x2.stats[ENV_RUNNER_RESULTS])

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
            # Step 3 from restored x2
            result_algo_0_step3_restored_x2 = algo_0_restored_x2.step()
            result_algo_1_step3_restored_x2 = algo_1_restored_x2.step()
            algo_0_restored_x2.stop()
            algo_1_restored_x2.stop()

            # Test that all results after step 3 are have 3x steps
            #  -- num_env_runners=0 --
            self.compare_metrics_in_results(
                result_algo_0_step3[ENV_RUNNER_RESULTS],
                result_algo_0_step3_restored[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3,
                metrics,
                "(Checkpointed) Check {} after step 3, env_runners=0",
            )
            self.compare_metrics_in_results(
                result_algo_0_step3[ENV_RUNNER_RESULTS],
                result_algo_0_step3_restored_x2[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3,
                metrics,
                "(Checkpointed x2) Check {} after step 3, env_runners=0",
            )
            # Evaluation
            self.compare_metrics_in_results(
                result_algo_0_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_0_step3_restored[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed) Check {} after step 3, env_runners=0",
            )
            self.compare_metrics_in_results(
                result_algo_0_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_0_step3_restored_x2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed x2) Check {} after step 3, env_runners=0",
            )
            #  -- num_env_runners=1 --
            self.compare_metrics_in_results(
                result_algo_1_step3[ENV_RUNNER_RESULTS],
                result_algo_1_step3_restored[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3,
                metrics,
                "(Checkpointed) Check {} after step 3, env_runners=1",
            )
            self.compare_metrics_in_results(
                result_algo_1_step3[ENV_RUNNER_RESULTS],
                result_algo_1_step3_restored_x2[ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3,
                metrics,
                "(Checkpointed x2) Check {} after step 3, env_runners=1",
            )
            # Evaluation
            self.compare_metrics_in_results(
                result_algo_1_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_1_step3_restored[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed) Check {} after step 3, env_runners=0",
            )
            self.compare_metrics_in_results(
                result_algo_1_step3[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                result_algo_1_step3_restored_x2[EVALUATION_RESULTS][ENV_RUNNER_RESULTS],
                ENV_STEPS_PER_ITERATION * 3 * 10,
                eval_metrics,
                "Evaluation: (Checkpointed x2) Check {} after step 3, env_runners=0",
            )

        return {
            "env_runners": {
                num_env_runners_expected[0]: {
                    "step_1": result_algo_0_step1[ENV_RUNNER_RESULTS],
                    "step_2": result_algo_0_step2[ENV_RUNNER_RESULTS],
                    "step_3": result_algo_0_step3[ENV_RUNNER_RESULTS],
                },
                num_env_runners_expected[1]: {
                    "step_1": result_algo_1_step1[ENV_RUNNER_RESULTS],
                    "step_2": result_algo_1_step2[ENV_RUNNER_RESULTS],
                    "step_3": result_algo_1_step3[ENV_RUNNER_RESULTS],
                },
            }
        }

    @Cases(TWO_ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    @pytest.mark.length("medium")
    @pytest.mark.timeout(method="thread")
    def test_trainable_checkpointing(self, cases):
        """Test if trainable can be checkpointed and restored."""
        for num_env_runners_a, num_env_runners_b in iter_cases(cases):
            with patch_args(
                "--batch_size", str(ENV_STEPS_PER_ITERATION),
                "--minibatch_size", str(ENV_STEPS_PER_ITERATION // 2),
                "--log_stats",  "most",  # increase log stats to assure necessary keys are present
                "--env_seeding_strategy", "same",
            ):  # fmt: skip
                setup = AlgorithmSetup(init_trainable=False)
                config = setup.config
                config.debugging(seed=11)
                config.environment(env="CartPole-v1")
                config.training(
                    train_batch_size_per_learner=ENV_STEPS_PER_ITERATION,
                    num_epochs=2,
                    minibatch_size=ENV_STEPS_PER_ITERATION // 2,
                )
                config.env_runners(num_env_runners=num_env_runners_a)
                config.evaluation(
                    evaluation_interval=1,
                    evaluation_duration=100,
                    evaluation_duration_unit="timesteps",
                    evaluation_num_env_runners=0,
                )
                Trainable0 = setup.create_trainable()
                setup = AlgorithmSetup(init_trainable=False)
                config = setup.config
                config.debugging(seed=11)
                config.environment(env="CartPole-v1")
                config.training(
                    train_batch_size_per_learner=ENV_STEPS_PER_ITERATION,
                    num_epochs=2,
                    minibatch_size=ENV_STEPS_PER_ITERATION // 2,
                )
                config.env_runners(num_env_runners=num_env_runners_b)
                config.evaluation(
                    evaluation_interval=1,
                    evaluation_duration=100,
                    evaluation_duration_unit="timesteps",
                    evaluation_num_env_runners=0,
                )

                Trainable1 = setup.create_trainable()
            assert isclass(Trainable0) and isclass(Trainable1)
            trainable0 = Trainable0()
            trainable1 = Trainable1()
            self.assertEqual(trainable0.algorithm_config.num_env_runners, num_env_runners_a)
            self.assertEqual(trainable1.algorithm_config.num_env_runners, num_env_runners_b)

            results = self._test_algo_checkpointing(
                trainable0,
                trainable1,
                num_env_runners_expected=(num_env_runners_a, num_env_runners_b),
                metrics=[
                    # NUM_ENV_STEPS_SAMPLED_LIFETIME, # not exact when using multiple envs per runner
                    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
                ],
            )

            self._test_checkpoint_values(
                results["env_runners"][num_env_runners_a]["step_3"],
                results["env_runners"][num_env_runners_b]["step_3"],
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
                NUM_ENV_STEPS_SAMPLED_LIFETIME,
                NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
            ],
        )

        self._test_checkpoint_values(
            results["env_runners"][0]["step_3"],
            results["env_runners"][1]["step_3"],
        )
        # because _throughput values results are not equal in structure (only after 2 steps)
        # self.set_max_diff(40000)

        # self.assertDictEqual(results["env_runners"][0]["step_3"], results["env_runners"][1]["step_3"])


if __name__ == "__main__":
    import unittest

    os.environ["RAY_DEBUG"] = "legacy"

    unittest.main(defaultTest="TestMetricsRestored.test_with_tuner")
