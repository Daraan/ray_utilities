from ray_utilities.learners import mix_learners
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from .utils import patch_args, SetupDefaults

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class TestLearners(SetupDefaults):
    @patch_args("-a", "mlp", "--accumulate_gradients_every", "2", "--smooth_accumulated_gradients")
    def test_ppo_torch_learner_with_gradient_accumulation(self):
        setup = AlgorithmSetup()
        # NOTE: Need RemoveMaskedSamplesLearner to assure only one epoch is done
        setup.config.training(
            learner_class=mix_learners([PPOTorchLearnerWithGradientAccumulation, RemoveMaskedSamplesLearner]),
            minibatch_size=64,
            train_batch_size_per_learner=64,
            num_epochs=1,
        )
        algo = setup.config.build()
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
