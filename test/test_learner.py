from collections import Counter
from typing import TYPE_CHECKING

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelpers, patch_args
from ray_utilities.training.helpers import make_divisible

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class TestLearners(InitRay, TestHelpers, DisableLoggers):
    # NOTE keep minibatch_size == batch size to keep update on step, otherwise update will happen mid-step
    def test_ppo_torch_learner_with_gradient_accumulation(self):
        batch_size = make_divisible(64, DefaultArgumentParser.num_envs_per_env_runner)
        with patch_args(
            "-a", "mlp", "--accumulate_gradients_every", "2", "--batch_size", batch_size, "--minibatch_size", batch_size
        ):
            setup = AlgorithmSetup(init_trainable=False)
        self.assertEqual(setup.config.train_batch_size_per_learner, batch_size)
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

    def test_gradient_accumulation_by_dynamic_batch(self):
        # TODO: Why does exact sampling not trim down the steps when they are not divisible by num_envs_per_env_runner?
        # A: Episodes are trimmed to at most 1, as those are kept we might end up with more steps.
        # When there are many short episodes (high num_envs_per_env_runner) this can easily happen.
        min_step_size = 32  # step sizes should be base x 1, x2, x4
        step_sizes = [
            make_divisible(min_step_size, DefaultArgumentParser.num_envs_per_env_runner) * 2**i for i in range(3)
        ]
        lower_step_size = step_sizes[0]
        total_steps = sum(s * 2**i for s, i in zip(step_sizes, range(len(step_sizes) - 1, -1, -1)))

        self.assertEqual(DefaultArgumentParser.accumulate_gradients_every, 1)
        with (patch_args(
            "-a", "mlp",
            "--dynamic_batch",
            "--batch_size", lower_step_size,  # is fixed
            "--minibatch_size", lower_step_size,
            "--total_steps", total_steps,  # 1x128 + 2x64 + 4x32 = 3x128, 4+2*2+4 = 12 iterations
            "--min_step_size", lower_step_size,
            "--max_step_size", step_sizes[-1],
            ),
            AlgorithmSetup(init_trainable=False) as setup,
        ):  # fmt: skip
            setup.config.training(
                num_epochs=1,
            )
        self.assertTrue(issubclass(setup.config.learner_class, PPOTorchLearnerWithGradientAccumulation))

        self.assertEqual(setup.config.learner_config_dict["accumulate_gradients_every"], 1)
        trainable = setup.trainable_class()
        self.assertTrue(self.is_algorithm_callback_added(trainable.algorithm_config, DynamicGradientAccumulation))
        learner: PPOTorchLearnerWithGradientAccumulation = trainable.algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        self.assertEqual(  # at start not accumulation, then 2, 4, ...
            learner.config.learner_config_dict["accumulate_gradients_every"],
            1,
        )
        module: DefaultPPOTorchRLModule = learner.module["default_policy"]  # type: ignore
        state0 = module.get_state()

        counter = Counter()
        self.assertEqual(setup.args.total_steps, total_steps, f"steps sizes are {step_sizes}")
        self.assertEqual(setup.args.iterations, 12)
        for iteration in range(setup.args.iterations):  # pyright: ignore[reportArgumentType]
            # do before on_train_result update
            accumulate_value = learner.config.learner_config_dict["accumulate_gradients_every"]
            result = trainable.train()
            self.assertEqual(result["training_iteration"], iteration + 1)
            self.assertEqual(result["current_step"], (iteration + 1) * step_sizes[0])
            self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, step_sizes[0])
            # count how often each accumulate value occured, + 1 step, + 0|1 gradient update
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
