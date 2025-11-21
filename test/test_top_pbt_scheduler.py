"""Unit tests for the TopTrialScheduler."""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import _PBTTrialState

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser
from ray_utilities.constants import PERTURBED_HPARAMS
from ray_utilities.testing_utils import DisableLoggers, TestHelpers, patch_args
from ray_utilities.tune.scheduler.top_pbt_scheduler import (
    SAVE_ALL_CHECKPOINTS,
    TopPBTTrialScheduler,
    _debug_dump_new_config,
    _grid_search_sample_function,
)

if TYPE_CHECKING:
    from ray_utilities.tune.scheduler.top_pbt_scheduler import _PBTTrialState2


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


if __name__ == "__main__":
    unittest.main()
