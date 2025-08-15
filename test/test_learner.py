from typing import TYPE_CHECKING

from ray_utilities.learners import mix_learners
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelpers, patch_args

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class TestLearners(InitRay, TestHelpers, DisableLoggers):
    @patch_args("-a", "mlp", "--accumulate_gradients_every", "2")
    def test_ppo_torch_learner_with_gradient_accumulation(self):
        setup = AlgorithmSetup(init_trainable=False)
        # NOTE: Need RemoveMaskedSamplesLearner to assure only one epoch is done
        setup.config.training(
            learner_class=mix_learners([PPOTorchLearnerWithGradientAccumulation, RemoveMaskedSamplesLearner]),
            minibatch_size=64,
            train_batch_size_per_learner=64,
            num_epochs=1,
        )
        setup.create_trainable()
        algo_setup = setup.config.build_algo()
        for algo in (algo_setup, setup.trainable_class().algorithm_config.build_algo()):
            with self.subTest("setup.config" if algo is algo_setup else "trainable.algorithm_config"):
                learner: PPOTorchLearnerWithGradientAccumulation = (
                    algo.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
                )
                self.assertEqual(
                    algo.config.learner_config_dict["accumulate_gradients_every"],  # pyright: ignore[reportOptionalMemberAccess]
                    2,
                )
                self.assertEqual(
                    learner.config.learner_config_dict["accumulate_gradients_every"],
                    2,
                )
                module: DefaultPPOTorchRLModule = learner.module["default_policy"]  # type: ignore
                state0 = module.get_state()
                algo.step()
                self.assertEqual(learner._step_count, 1)
                state1 = module.get_state()
                self.util_test_tree_equivalence(state0, state1)
                algo.step()
                self.assertEqual(learner._step_count, 2)
                state2 = module.get_state()
                with self.assertRaisesRegex(AssertionError, "(weight|bias).* not equal"):
                    self.util_test_tree_equivalence(state1, state2, use_subtests=False)
