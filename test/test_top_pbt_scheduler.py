"""Unit tests for the TopTrialScheduler."""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
import time
import unittest
from typing import TYPE_CHECKING, cast
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import _PBTTrialState

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser
from ray_utilities.constants import PERTURBED_HPARAMS
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelpers, patch_args
from ray_utilities.tune.scheduler.top_pbt_scheduler import (
    SAVE_ALL_CHECKPOINTS,
    TopPBTTrialScheduler,
    _debug_dump_new_config,
    _grid_search_sample_function,
)

if TYPE_CHECKING:
    from ray_utilities.tune.scheduler.top_pbt_scheduler import _PBTTrialState2

logger = logging.getLogger(__name__)


class TestPBTParser(unittest.TestCase):
    @unittest.skip("not implemented")
    def test_hyperparam_mutations_parsing(self): ...

    @patch_args("pbt")
    def test_default_overrides(self):
        for parser in (PopulationBasedTrainingParser(), DefaultArgumentParser()):
            args = parser.parse_args(known_only=True)
            if isinstance(args, DefaultArgumentParser):
                assert args.command is not None
                args = args.command
            self.assertEqual(args.time_attr, "current_step")
            self.assertEqual(args.quantile_fraction, 0.1)
            self.assertEqual(args.perturbation_interval, PopulationBasedTrainingParser.perturbation_interval)
            self.assertEqual(args.resample_probability, 1.0)
            self.assertEqual(args.mode, "max")


class TestGridSearchSampleFunction(DisableLoggers, TestHelpers):
    """Tests for the grid search sample function utility."""

    def test_grid_search_sample_repeat(self):
        """Test grid search sampler with repeat enabled."""
        values = [1, 2, 3]
        sampler = _grid_search_sample_function(values, repeat=True)

        # Sample more than the length of the list to test cycling
        results = [sampler() for _ in range(7)]
        expected = [1, 2, 3, 1, 2, 3, 1]

        self.assertEqual(results, expected)

    def test_grid_search_sample_no_repeat(self):
        """Test grid search sampler without repeat."""
        values = [1, 2, 3]
        sampler = _grid_search_sample_function(values, repeat=False)

        # Should return each value once
        self.assertEqual(sampler(), 1)
        self.assertEqual(sampler(), 2)
        self.assertEqual(sampler(), 3)

        # Should raise StopIteration after exhausting the list
        with self.assertRaises(StopIteration):
            sampler()


class TestDebugDumpNewConfig(DisableLoggers, TestHelpers):
    """Tests for the debug dump config function."""

    def test_debug_dump_new_config(self):
        """Test that perturbed hyperparameters are recorded in the config."""
        config = {"lr": 0.001, "batch_size": 64}
        mutation_keys = ["lr", "batch_size"]

        result = _debug_dump_new_config(config, mutation_keys)

        self.assertIn(PERTURBED_HPARAMS, result)
        self.assertEqual(result[PERTURBED_HPARAMS], {"lr": 0.001, "batch_size": 64})


class TestTopTrialScheduler(DisableLoggers, TestHelpers):
    """Tests for the TopTrialScheduler class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Create a scheduler with default settings
        self.scheduler = TopPBTTrialScheduler(
            metric="reward",
            mode="max",
            perturbation_interval=10,
            hyperparam_mutations={
                "lr": {"grid_search": [0.001, 0.01, 0.1]},
                "batch_size": {"grid_search": [32, 64, 128]},
            },
            quantile_fraction=0.2,
        )

        # Mock trials and trial states
        self.trials = []
        self.trial_scores = []

        # Create 10 mock trials with different scores
        for i in range(10):
            trial = MagicMock(spec=Trial)
            trial.trial_id = f"trial_{i}"
            trial.is_finished.return_value = False

            state = MagicMock(spec=_PBTTrialState)
            score = random.random() * 100  # Random score between 0 and 100
            state.last_score = score
            state.last_checkpoint = None
            state.last_perturbation_time = 0
            state.last_result = {"reward": score}
            state.last_train_time = 0

            self.trials.append(trial)
            self.trial_scores.append(score)
            self.scheduler._trial_state[trial] = state

    def test_initialization(self):
        """Test scheduler initialization."""
        # Verify default values
        self.assertEqual(self.scheduler._time_attr, "current_step")
        self.assertEqual(self.scheduler._metric, "reward")
        self.assertEqual(self.scheduler._mode, "max")
        self.assertEqual(self.scheduler._quantile_fraction, 0.2)

        # Verify grid search mutation conversion
        for fn in self.scheduler._hyperparam_mutations.values():
            self.assertTrue(callable(fn))

    def test_quantiles_max_mode(self):
        """Test quantile calculation with max mode."""
        lower, upper = self.scheduler._quantiles()

        # Since quantile_fraction is 0.2, we should have 2 trials in upper quantile (20% of 10)
        # and 2 trials in lower quantile as well.
        self.assertEqual(len(upper), 2)
        self.assertEqual(len(lower), 8)

        # Upper quantile should have the highest scores
        upper_scores = [self.scheduler._trial_state[t].last_score for t in upper]
        lower_scores = [self.scheduler._trial_state[t].last_score for t in lower]
        lower_scores_filtered = [s for s in lower_scores if s is not None]

        # Verify all upper scores are higher than any lower score
        self.assertTrue(all(u > max(lower_scores_filtered) for u in upper_scores))  # pyright: ignore[reportOptionalOperand]

    def test_quantiles_min_mode(self):
        """Test quantile calculation with min mode."""
        # Create a new scheduler with min mode and dummy hyperparam_mutations
        scheduler = TopPBTTrialScheduler(
            mode="min",
            perturbation_interval=10,
            quantile_fraction=0.2,
            hyperparam_mutations={"dummy": [1, 2]},
        )

        # Copy trial states to the new scheduler
        for trial, state in self.scheduler._trial_state.items():
            scheduler._trial_state[trial] = state

        lower, upper = scheduler._quantiles()

        # Upper quantile should have the lowest scores in min mode
        upper_scores = [scheduler._trial_state[t].last_score for t in upper]
        lower_scores = [scheduler._trial_state[t].last_score for t in lower]
        # Filter out None values for min()
        lower_scores_filtered = [s for s in lower_scores if s is not None]
        # Verify all upper scores are greater than any lower score. Remember for min mode the sign is switched.
        self.assertTrue(all(u > max(lower_scores_filtered) for u in upper_scores))  # pyright: ignore[reportOptionalOperand]

    def test_distribute_exploitation(self):
        """Test exploitation distribution among trials."""
        lower, upper = self.scheduler._quantiles()

        assignments = self.scheduler._distribute_exploitation(lower, upper)

        # Each lower trial should be assigned to an upper trial
        self.assertEqual(len(assignments), len(lower))

        # Count how many times each upper trial is assigned
        assignment_counts = {}
        for upper_trial in assignments.values():
            if upper_trial.trial_id in assignment_counts:
                assignment_counts[upper_trial.trial_id] += 1
            else:
                assignment_counts[upper_trial.trial_id] = 1

        # Assignments should be balanced - each upper trial should have
        # roughly the same number of lower trials assigned to it
        counts = list(assignment_counts.values())
        self.assertTrue(max(counts) - min(counts) <= 1)

    def test_reset_exploitations(self):
        """Test resetting exploitations."""
        # Set up current assignments
        lower, upper = self.scheduler._quantiles()
        self.scheduler._current_assignments = self.scheduler._distribute_exploitation(lower, upper)

        # Verify assignments exist
        self.assertIsNotNone(self.scheduler._current_assignments)

        # Reset and verify
        self.scheduler.reset_exploitations()
        self.assertIsNone(self.scheduler._current_assignments)

    @patch("ray.tune.execution.tune_controller.TuneController")
    @patch("pathlib.Path.open", new=MagicMock())
    def test_on_trial_complete(self, mock_controller):
        """Test on_trial_complete resets exploitation assignments."""
        # Setup current assignments
        lower, upper = self.scheduler._quantiles()
        self.scheduler._current_assignments = self.scheduler._distribute_exploitation(lower, upper)

        # Call on_trial_complete
        trial = self.trials[0]
        result = {"reward": 100, "training_iteration": 1}

        # Mock parent class method
        self.scheduler.on_trial_complete(mock_controller, trial, result)

        self.assertIsNone(self.scheduler._current_assignments)


class TestTopTrialSchedulerIntegration(DisableLoggers, TestHelpers):
    """Integration tests for TopTrialScheduler."""

    @patch("ray.tune.execution.tune_controller.TuneController")
    @patch("ray.tune.schedulers.pbt.PopulationBasedTraining.on_trial_add")
    @patch("pathlib.Path.open", new=MagicMock())
    def test_checkpoint_or_exploit(self, mock_controller, _on_trial_mock):
        """Test the checkpoint_or_exploit method."""
        scheduler = TopPBTTrialScheduler(
            metric="reward",
            perturbation_interval=10,
            quantile_fraction=0.3,
            hyperparam_mutations={"dummy": [1, 2]},
            reseed=True,
        )

        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        mock_controller.experiment_path = temp_dir

        # Create trials and states
        trials = []
        for i in range(10):
            trial = MagicMock(spec=Trial)
            trial.trial_id = f"trial_{i}"
            trial.is_finished.return_value = False
            trial.status = Trial.RUNNING
            trial.config = {"dummy": -i, "env_seed": 0}
            trial.storage = None

            # For testing the temporary_state for paused trials
            if i < 3:  # Make first 3 trials be in upper quantile
                score = 90 + i
            else:
                score = 10 + i

            state = cast("_PBTTrialState2", MagicMock())
            state.last_score = score
            state.last_checkpoint = None
            state.last_perturbation_time = 1
            state.last_train_time = 0
            state.last_result = {"reward": score, "training_iteration": 1}

            trials.append(trial)
            scheduler._trial_state[trial] = state
            scheduler.on_trial_add(mock_controller, trial)
            # this is set to 0 on_trial_add
            state.current_env_steps = 12
            state.last_training_iteration = 1

        # Mock controller methods
        mock_controller._schedule_trial_save.return_value = "checkpoint_path"

        # Mock _exploit to verify it's called correctly
        with patch.object(scheduler, "_exploit") as mock_exploit:
            # Calculate quantiles
            lower, upper = scheduler._quantiles()

            # Test with a trial in the upper quantile
            upper_trial = upper[0]
            scheduler._checkpoint_or_exploit(upper_trial, mock_controller, upper, lower)

            self.assertEqual(upper_trial.config["env_seed"], (0, 12))

            # Verify upper trial gets a checkpoint but doesn't exploit
            mock_controller._schedule_trial_save.assert_called_with(
                upper_trial, result=scheduler._trial_state[upper_trial].last_result
            )
            self.assertEqual(scheduler._trial_state[upper_trial].last_checkpoint, "checkpoint_path")
            mock_exploit.assert_not_called()

            # Reset mock
            mock_controller._schedule_trial_save.reset_mock()
            mock_exploit.reset_mock()

            # Test with a trial in the lower quantile
            lower_trial = lower[0]

            # Need to have checkpoints for upper trials
            for trial in upper:
                scheduler._trial_state[trial].last_checkpoint = "upper_checkpoint"  # pyright: ignore[reportAttributeAccessIssue]

            scheduler._checkpoint_or_exploit(lower_trial, mock_controller, upper, lower)

            self.assertEqual(upper_trial.config["env_seed"], (0, 12))

            if SAVE_ALL_CHECKPOINTS:
                # Verify lower trial gets a checkpoint and exploits if we care about it
                mock_controller._schedule_trial_save.assert_called_with(
                    lower_trial, result=scheduler._trial_state[lower_trial].last_result
                )
                self.assertIsNone(scheduler._trial_state[lower_trial].last_checkpoint)

            # Should exploit one of the upper trials
            mock_exploit.assert_called_once()
            exploit_args = mock_exploit.call_args[0]
            self.assertEqual(exploit_args[0], mock_controller)
            self.assertEqual(exploit_args[1], lower_trial)
            self.assertIn(exploit_args[2], upper)


class TestGroupedTopPBTTrialScheduler(DisableLoggers, TestHelpers):
    """Tests for the GroupedTopPBTTrialScheduler class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler

        # Create a grouped scheduler
        self.scheduler = GroupedTopPBTTrialScheduler(
            metric="reward",
            mode="max",
            perturbation_interval=100,
            quantile_fraction=0.25,
            num_samples=3,
        )

    def test_group_trials_by_config(self):
        """Test that trials are correctly grouped by config."""
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler

        scheduler = GroupedTopPBTTrialScheduler(metric="reward", mode="max", num_samples=3)

        # Create mock trials with different seeds but same config
        trials = []
        for i in range(6):
            trial = MagicMock(spec=Trial)
            trial.trial_id = f"trial_{i}"
            trial.config = {
                "lr": 0.001 if i < 3 else 0.01,  # Two different configs
                "env_seed": i,  # Different seeds
                "batch_size": 64,
            }
            trial.is_finished.return_value = False
            trials.append(trial)
            scheduler._trial_state[trial] = MagicMock(spec=_PBTTrialState)
            scheduler._trial_state[trial].last_score = float(i)

        # Group trials
        groups = scheduler._group_trials_by_config(trials)

        # Should have 2 groups (different lr values)
        self.assertEqual(len(groups), 2)

        # Each group should have 3 trials
        for group_trials in groups.values():
            self.assertEqual(len(group_trials), 3)

    def test_quantiles_with_groups(self):
        """Test that quantiles are computed based on group averages."""
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler

        scheduler = GroupedTopPBTTrialScheduler(
            metric="reward",
            mode="max",
            perturbation_interval=100,
            quantile_fraction=0.5,  # Keep top 50% of groups
            num_samples=2,
        )

        # Create 4 groups of 2 trials each
        trials = []
        for group_idx in range(4):
            for trial_idx in range(2):
                trial = MagicMock(spec=Trial)
                trial.trial_id = f"trial_g{group_idx}_t{trial_idx}"
                trial.config = {
                    "lr": 0.001 * (group_idx + 1),  # Different config per group
                    "env_seed": trial_idx,
                }
                trial.is_finished.return_value = False
                trials.append(trial)

                state = MagicMock(spec=_PBTTrialState)
                # Group 0: avg 0.5, Group 1: avg 1.5, Group 2: avg 2.5, Group 3: avg 3.5
                state.last_score = float(group_idx) + (trial_idx * 0.5)
                scheduler._trial_state[trial] = state

        lower, upper = scheduler._quantiles()

        # With quantile_fraction=0.5 and 4 groups, should have 2 top groups
        # That's 4 trials in upper (groups 2 and 3)
        self.assertEqual(len(upper), 4)
        # And 4 trials in lower (groups 0 and 1)
        self.assertEqual(len(lower), 4)

    def test_distribute_exploitation_group_matching(self):
        """Test that exploitation is distributed with 1:1 group matching."""
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler

        scheduler = GroupedTopPBTTrialScheduler(metric="reward", mode="max", num_samples=2)

        # Create 2 lower groups and 2 upper groups
        lower_trials = []
        upper_trials = []

        for is_lower in [True, False]:
            trial_list = lower_trials if is_lower else upper_trials
            for group_idx in range(2):
                for trial_idx in range(2):
                    trial = MagicMock(spec=Trial)
                    prefix = "lower" if is_lower else "upper"
                    trial.trial_id = f"{prefix}_g{group_idx}_t{trial_idx}"
                    trial.config = {
                        "lr": (0.001 if is_lower else 0.01) * (group_idx + 1),
                        "env_seed": trial_idx,
                    }
                    trial_list.append(trial)
                    scheduler._trial_state[trial] = MagicMock(spec=_PBTTrialState)

        assignments = scheduler._distribute_exploitation(lower_trials, upper_trials)

        # Should have assignments for all lower trials
        self.assertEqual(len(assignments), 4)

        # Verify each lower trial is assigned to an upper trial
        for lower_trial, upper_trial in assignments.items():
            self.assertIn(lower_trial, lower_trials)
            self.assertIn(upper_trial, upper_trials)


class TestGroupedTopPBTIntegration(InitRay, TestHelpers, DisableLoggers):
    """Integration test for GroupedTopPBTTrialScheduler using run_tune.

    Note: This test requires Ray to be initialized. Run with InitRay if needed.
    """

    @pytest.mark.length(speed="medium")
    @mock.patch("wandb.Api", new=MagicMock())
    @mock.patch("ray_utilities.callbacks.wandb.wandb_api", new=MagicMock())
    def test_run_tune_with_grouped_top_pbt_scheduler(self):
        """Test GroupedTopPBTTrialScheduler with run_tune using grouped trials."""
        # Need to import here to avoid circular imports
        import ray
        from ray import tune
        from ray.rllib.utils.metrics import (
            ENV_RUNNER_RESULTS,
            EPISODE_RETURN_MEAN,
            EVALUATION_RESULTS,
            NUM_ENV_STEPS_SAMPLED_LIFETIME,
        )
        from ray.tune.schedulers.pbt import logger as ray_pbt_logger

        from ray_utilities.constants import (
            EPISODE_RETURN_MEAN_EMA,
            NUM_ENV_STEPS_PASSED_TO_LEARNER,
            NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
        )
        from ray_utilities.misc import is_pbar, raise_tune_errors
        from ray_utilities.runfiles import run_tune
        from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
        from ray_utilities.testing_utils import (
            SetupWithCheck,
            TrainableWithChecks,
            patch_args,
        )
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler
        from ray_utilities.tune.scheduler.top_pbt_scheduler import CyclicMutation, KeepMutation

        # Skip if Ray not initialized
        if not ray.is_initialized():
            pytest.skip("Ray not initialized")

        original_exploit = GroupedTopPBTTrialScheduler._exploit
        perturbation_interval = 100
        best_group_idx = 1  # Group 1 (with lr=0.01) will be best

        # Create 3 learning rates, each with 2 seeds = 6 trials total
        # 3 groups of 2 trials each
        learning_rates = (0.001, 0.01, 0.005)  # Group 1 (lr=0.01) will have highest scores
        num_seeds = 2

        num_exploits = 0
        group_exploit_counts = dict.fromkeys(learning_rates, 0)

        # Fake results: scores depend on learning rate and step
        # Group with lr=0.01 will consistently perform best
        fake_results: dict[float, dict[int, float]] = {
            learning_rates[0]: {  # lr=0.001: scores 1, 2, 3, ...
                v: v // perturbation_interval for v in range(perturbation_interval, 401, perturbation_interval)
            },
            learning_rates[1]: {  # lr=0.01: scores 21, 22, 23, ... (BEST)
                v: v // perturbation_interval + 20 for v in range(perturbation_interval, 401, perturbation_interval)
            },
            learning_rates[2]: {  # lr=0.005: scores 6, 7, 8, ...
                v: v // perturbation_interval + 5 for v in range(perturbation_interval, 401, perturbation_interval)
            },
        }

        race_conditions = 0

        def test_exploit_function(self: GroupedTopPBTTrialScheduler, tune_controller, trial, trial_to_clone) -> None:
            """Verify group-based exploitation logic."""
            nonlocal num_exploits, race_conditions
            num_exploits += 1

            trial_lr = trial.config.get("lr", trial.config.get("cli_args", {}).get("lr"))
            clone_lr = trial_to_clone.config.get("lr", trial_to_clone.config.get("cli_args", {}).get("lr"))

            logger.info(
                "Exploit #%d: trial lr=%s → clone lr=%s at step %s",
                num_exploits,
                trial_lr,
                clone_lr,
                self._trial_state[trial].last_train_time,
            )

            if self._trial_state[trial].last_perturbation_time % perturbation_interval != 0:
                race_conditions += 1
                logger.warning(
                    "Exploit at step %s not at perturbation interval (race condition)",
                    self._trial_state[trial].last_perturbation_time,
                )
            else:
                # Verify that lower-performing trial exploits higher-performing group
                # The best group (lr=0.01) should be cloned
                assert clone_lr == learning_rates[best_group_idx], (
                    f"Expected clone from best group lr={learning_rates[best_group_idx]}, got lr={clone_lr}"
                )
                # Trial being exploited should NOT be from best group
                assert trial_lr != learning_rates[best_group_idx], (
                    f"Trial with lr={trial_lr} should not be in best group"
                )

                group_exploit_counts[trial_lr] += 1

            # Call original exploit function
            original_exploit(self, tune_controller, trial, trial_to_clone)

        GroupedTopPBTTrialScheduler._exploit = test_exploit_function

        class CheckTrainableForGroupedPBT(TrainableWithChecks):
            """Custom trainable that returns predetermined scores based on learning rate."""

            debug_step = False
            use_pbar = False

            def step(self):  # pyright: ignore[reportIncompatibleMethodOverride]
                """Return fake results based on learning rate."""
                # Get learning rate from config
                lr = self.algorithm_config.lr or self.config.get("cli_args", {}).get("lr", 0.001)

                self._current_step += self.algorithm_config.train_batch_size_per_learner
                result = {ENV_RUNNER_RESULTS: {}, EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {}}}
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME] = self._current_step
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER] = (
                    self.algorithm_config.train_batch_size_per_learner
                )
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME] = self._current_step + 2

                # from ray_utilities.testing_utils import remote_breakpoint
                # remote_breakpoint()
                # Return score from fake_results
                result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] = fake_results[lr][
                    self._current_step
                ]
                result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN_EMA] = result[EVALUATION_RESULTS][
                    ENV_RUNNER_RESULTS
                ][EPISODE_RETURN_MEAN]
                result["_checking_class_"] = "CheckTrainableForGroupedPBT"

                logger.info(
                    "LR: %s, step %s, result: %s",
                    lr,
                    self._current_step,
                    result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
                )
                result["current_step"] = self._current_step

                if is_pbar(self._pbar):
                    self._pbar.update(1)
                    self._pbar.set_description(
                        f"Step: {self._current_step} lr={lr} "
                        f"result={result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}"
                    )

                time.sleep(2)  # Simulate some work
                return result

        ray_pbt_logger.setLevel(logging.DEBUG)

        with patch_args(
            # Main experiment args
            "--tune", "lr",
            # Meta arguments
            "--num_samples", num_seeds,  # 2 seeds per learning rate
            "--num_jobs", len(learning_rates) * num_seeds,  # 6 total trials
            "--batch_size", perturbation_interval,
            "--minibatch_size",
            perturbation_interval,
            "--total_steps", perturbation_interval * 3,
            "--use_exact_total_steps",
            # TODO: We want all groups to have the same seed sequence
            "--env_seeding_strategy", "sequential",  # Different seeds per trial
            # Constant
            "--seed", "42",
            "--log_level", "DEBUG",
            "--log_stats", "minimal",
            "--no_dynamic_eval_interval",
            "--fcnet_hiddens", "[4]",
            "--test",
            "--num_envs_per_env_runner", 5,
            "pbt",
            "--quantile_fraction", "0.34",  # Top 1/3 of groups (1 out of 3)
            "--perturbation_interval", perturbation_interval,
        ):  # fmt: skip
            Setup = SetupWithCheck(CheckTrainableForGroupedPBT, PPOMLPWithPBTSetup)
            setup = Setup(config_files=["experiments/models/mlp/default.cfg"])

            # Use grid search for learning rate
            setup.param_space["lr"] = tune.grid_search(learning_rates)

            # Set mutations
            assert setup.args.command
            setup.args.command.set_hyperparam_mutations(
                {
                    "lr": CyclicMutation(learning_rates),
                    "fcnet_hiddens": KeepMutation([4]),
                }
            )

            # Create custom scheduler
            setup.args.command.to_scheduler = lambda *args, **kwargs: GroupedTopPBTTrialScheduler(
                # metric="episode_reward_mean",
                # mode="max",
                perturbation_interval=perturbation_interval,
                quantile_fraction=0.34,  # Top 1 out of 3 groups
                hyperparam_mutations=setup.args.command.hyperparam_mutations,
                num_samples=num_seeds,  # Expected trials per group
                synch=True,
            )

            results = run_tune(setup)
            raise_tune_errors(results)

            # Verify all results are from our custom trainable
            self.assertTrue(
                all(result.metrics["_checking_class_"] == "CheckTrainableForGroupedPBT" for result in results)
            )

            # Expected exploitations:
            # - 3 steps total (at 100, 200, 300)
            # - At each perturbation interval, lower quantile groups exploit upper quantile
            # - With 3 groups and quantile_fraction=0.34, we have 1 upper group (best) and 2 lower groups
            # - Each lower group has 2 trials, so 4 total exploitations per interval
            # - But at step 300, training ends, so only 2 intervals have full exploitation
            expected_exploits = 2 * 2 * num_seeds  # 2 lower groups x 2 trials x 2 intervals

            logger.info("Total exploits: %d, Expected: %d", num_exploits, expected_exploits)
            self.assertGreaterEqual(num_exploits, expected_exploits - 2)  # Allow some race conditions
            self.assertLessEqual(num_exploits, expected_exploits + 2)

            # Check that at most a few race conditions happened
            self.assertLessEqual(race_conditions, 2)

            # Verify all final configs have fcnet_hiddens=[4]
            self.assertTrue(all(r.config["fcnet_hiddens"] == [4] for r in results))

            logger.info("Group exploit counts: %s", group_exploit_counts)

        # Restore original exploit function
        GroupedTopPBTTrialScheduler._exploit = original_exploit


if __name__ == "__main__":
    unittest.main()
