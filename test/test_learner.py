from collections import Counter
from typing import TYPE_CHECKING

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelpers, patch_args

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class TestLearners(InitRay, TestHelpers, DisableLoggers):
    # NOTE keep minibatch_size == batch size to keep update on step, otherwise update will happen mid-step
    @patch_args("-a", "mlp", "--accumulate_gradients_every", "2", "--batch_size", 64, "--minibatch_size", 64)
    def test_ppo_torch_learner_with_gradient_accumulation(self):
        setup = AlgorithmSetup(init_trainable=False)
        self.assertEqual(setup.config.train_batch_size_per_learner, 64)
        # NOTE: Need RemoveMaskedSamplesLearner to assure only one epoch is done
        setup.config.training(
            num_epochs=1,
        )
        self.assertTrue(issubclass(setup.config.learner_class, PPOTorchLearnerWithGradientAccumulation))
        setup.create_trainable()
        algorithm = setup.config.build_algo()
        assert algorithm.config
        self.assertTrue(issubclass(algorithm.config.learner_class, PPOTorchLearnerWithGradientAccumulation))
        for algo in (algorithm, setup.trainable_class().algorithm_config.build_algo()):
            with self.subTest("setup.config" if algo is algorithm else "trainable.algorithm_config"):
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

    @patch_args(
        "-a", "mlp",
        "--dynamic_batch",
        "--batch_size", 32,  # is fixed
        "--minibatch_size", 32,
        "--total_steps", 384,  # 1x128 + 2x64 + 4x32 = 3x128, 4+2*2+4 = 12 iterations
        "--min_step_size", 32,
        "--max_step_size", 128,
    )  # fmt: skip
    def test_gradient_accumulation_by_dynamic_batch(self):
        with AlgorithmSetup(init_trainable=False) as setup:
            setup.config.training(
                num_epochs=1,
            )
        print("iterations", setup.args.iterations)
        self.assertTrue(issubclass(setup.config.learner_class, PPOTorchLearnerWithGradientAccumulation))

        trainable = setup.trainable_class()
        self.assertTrue(self.is_algorithm_callback_added(trainable.algorithm_config, DynamicGradientAccumulation))
        learner: PPOTorchLearnerWithGradientAccumulation = trainable.algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        self.assertEqual(
            learner.config.learner_config_dict["accumulate_gradients_every"],
            1,
        )
        module: DefaultPPOTorchRLModule = learner.module["default_policy"]  # type: ignore
        state0 = module.get_state()

        counter = Counter()
        self.assertEqual(setup.args.total_steps, 384)
        self.assertEqual(setup.args.iterations, 12)
        for iteration in range(setup.args.iterations):  # pyright: ignore[reportArgumentType]
            # do before on_train_result update
            accumulate_value = learner.config.learner_config_dict["accumulate_gradients_every"]
            result = trainable.train()
            self.assertEqual(result["training_iteration"], iteration + 1)
            self.assertEqual(result["current_step"], (iteration + 1) * 32)
            self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 32)
            counter.update(
                {
                    "learner_step_count": 1,
                    "gradient_updated": learner._last_gradient_update_step == learner._step_count,
                    f"accumulate_every={accumulate_value}": 1,
                }
            )
        self.assertEqual(learner._step_count, 12)
        self.assertEqual(learner._gradient_updates, 4 + 2 + 1)
        self.assertEqual(counter["learner_step_count"], learner._step_count)
        self.assertEqual(counter["gradient_updated"], learner._gradient_updates)
        self.assertEqual(counter["accumulate_every=1"], 4)  # 4x32
        self.assertEqual(counter["accumulate_every=2"], 4)  # 2x64 = 4x32
        self.assertEqual(counter["accumulate_every=4"], 4)  # 1x128 = 4x32
