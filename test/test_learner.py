from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from typing import TYPE_CHECKING, NamedTuple

import torch

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.learners import mix_learners
from ray_utilities.learners.dqn_torch_learner_with_gradient_accumulation import (
    DQNTorchLearnerWithGradientAccumulation,
)
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.ppo_mlp_setup import DQNMLPSetup
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelpers, no_parallel_envs, patch_args
from ray_utilities.training.helpers import is_algorithm_callback_added, make_divisible

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class _ApplyCall(NamedTuple):
    step_count: int
    applied_gradients: bool
    gradients: dict | None = None


class TestLearners(InitRay, TestHelpers, DisableLoggers, num_cpus=4):
    # NOTE keep minibatch_size == batch size to keep update on step, otherwise update will happen mid-step
    def test_ppo_torch_learner_with_gradient_accumulation(self):
        batch_size = make_divisible(64, DefaultArgumentParser.num_envs_per_env_runner)
        with patch_args(
            "-a",
            "mlp",
            "--accumulate_gradients_every",
            "2",
            "--batch_size",
            batch_size,
            "--minibatch_size",
            batch_size,
            "--fcnet_hiddens",
            "[8, 8]",
        ):
            setup = MLPSetup(init_trainable=False)
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

        def mock_apply_gradients(gradients_dict):
            apply_calls.append(
                _ApplyCall(
                    step_count=learner._step_count,
                    applied_gradients=len(gradients_dict) > 0,
                )
            )
            return original_apply_gradients(gradients_dict)

        for algo in (algorithm, setup.trainable_class().algorithm_config.build_algo()):
            with self.subTest("setup.config" if algo is algorithm else "trainable.algorithm_config"):
                apply_calls: list[_ApplyCall] = []
                assert algo.learner_group is not None
                learner: PPOTorchLearnerWithGradientAccumulation = algo.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
                # Track gradient application calls
                original_apply_gradients = learner.apply_gradients

                learner.apply_gradients = mock_apply_gradients
                learner: PPOTorchLearnerWithGradientAccumulation = (
                    algo.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
                )
                assert algo.config is not None
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
                self.assertEqual(len(apply_calls), 1)
                self.assertFalse(apply_calls[0].applied_gradients, "First step should not apply gradients")
                state1 = module.get_state()
                self.util_test_tree_equivalence(state0, state1)
                algo.step()
                self.assertEqual(learner._step_count, 2)
                self.assertEqual(len(apply_calls), 2)
                self.assertTrue(apply_calls[1].applied_gradients, "Second step should apply gradients")
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
        with (
            patch_args(
                "-a", "mlp",
                "--dynamic_batch",
                "--batch_size", lower_step_size,  # is fixed
                "--minibatch_size", lower_step_size,
                "--total_steps", total_steps,  # 1x128 + 2x64 + 4x32 = 3x128, 4+2*2+4 = 12 iterations
                "--min_step_size", lower_step_size,
                "--max_step_size", step_sizes[-1],
                "--fcnet_hiddens", "[8, 8]",
                "--num_envs_per_env_runner", "4",
            ),
            MLPSetup(init_trainable=False) as setup,
        ):  # fmt: skip
            setup.config.training(
                num_epochs=1,
            )
        self.assertTrue(issubclass(setup.config.learner_class, PPOTorchLearnerWithGradientAccumulation))

        self.assertEqual(setup.config.learner_config_dict["accumulate_gradients_every"], 1)
        trainable = setup.trainable_class()
        self.assertTrue(is_algorithm_callback_added(trainable.algorithm_config, DynamicGradientAccumulation))
        assert trainable.algorithm.learner_group is not None
        learner: PPOTorchLearnerWithGradientAccumulation = trainable.algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        self.assertEqual(  # at start not accumulation, then 2, 4, ...
            learner.config.learner_config_dict["accumulate_gradients_every"],
            1,
        )
        module: DefaultPPOTorchRLModule = learner.module["default_policy"]  # type: ignore
        _state0 = module.get_state()

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

    def test_gradient_accumulation_behavior(self):
        """Test that gradients are accumulated correctly and applied only when expected."""
        batch_size = make_divisible(32, 6)
        accumulate_every = 3

        with patch_args(
            "-a", "mlp",
            "--accumulate_gradients_every", str(accumulate_every),
            "--batch_size", batch_size,
            "--minibatch_size", batch_size,
            "--fcnet_hiddens", "[8, 8]",
            "--num_envs_per_env_runner", "6",
        ):  # fmt: skip
            setup = MLPSetup(init_trainable=False)

        setup.config.training(num_epochs=1)
        algorithm = setup.config.build_algo()
        learner: PPOTorchLearnerWithGradientAccumulation = (
            algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        )

        # Track gradient application calls
        original_apply_gradients = learner.apply_gradients

        apply_calls: list[_ApplyCall] = []

        def mock_apply_gradients(gradients_dict):
            apply_calls.append(
                _ApplyCall(
                    step_count=learner._step_count,
                    applied_gradients=len(gradients_dict) > 0,
                )
            )
            return original_apply_gradients(gradients_dict)

        learner.apply_gradients = mock_apply_gradients

        # Track parameter changes
        module = learner.module["default_policy"]
        initial_state = module.get_state()

        # Step 1: First step should zero gradients and accumulate, but not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 1)
        self.assertEqual(learner._gradient_updates, 0)
        self.assertIsNone(learner._last_gradient_update_step)
        self.assertEqual(len(apply_calls), 1)
        self.assertFalse(apply_calls[0].applied_gradients, "First step should not apply gradients")

        # Parameters should not have changed
        state_after_step1 = module.get_state()
        self.util_test_tree_equivalence(initial_state, state_after_step1)

        # Step 2: Second step should accumulate, but not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 2)
        self.assertEqual(learner._gradient_updates, 0)
        self.assertIsNone(learner._last_gradient_update_step)
        self.assertEqual(len(apply_calls), 2)
        self.assertFalse(apply_calls[1].applied_gradients, "Second step should not apply gradients")

        # Parameters should still not have changed
        state_after_step2 = module.get_state()
        self.util_test_tree_equivalence(initial_state, state_after_step2)

        # Step 3: Third step should accumulate and apply gradients
        algorithm.step()
        self.assertEqual(learner._step_count, 3)
        self.assertEqual(learner._gradient_updates, 1)
        self.assertEqual(learner._last_gradient_update_step, 3)
        self.assertEqual(len(apply_calls), 3)
        self.assertTrue(apply_calls[2].applied_gradients, "Third step should apply gradients")

        # Parameters should have changed now
        state_after_step3 = module.get_state()
        with self.assertRaisesRegex(AssertionError, "(weight|bias).* not equal"):
            self.util_test_tree_equivalence(initial_state, state_after_step3, use_subtests=False)

        # Step 4: Fourth step should start new accumulation cycle
        algorithm.step()
        self.assertEqual(learner._step_count, 4)
        self.assertEqual(learner._gradient_updates, 1)  # Should still be 1
        self.assertEqual(learner._last_gradient_update_step, 3)  # Should still be 3
        self.assertEqual(len(apply_calls), 4)
        self.assertFalse(apply_calls[3].applied_gradients, "Fourth step should not apply gradients (new cycle)")

        # Parameters should not have changed from after the previous update
        state_after_step4 = module.get_state()
        self.util_test_tree_equivalence(state_after_step3, state_after_step4)

        # Step 5: Fifth step should accumulate
        algorithm.step()
        self.assertEqual(learner._step_count, 5)
        self.assertEqual(learner._gradient_updates, 1)

        # Step 6: Sixth step should apply gradients again
        algorithm.step()
        self.assertEqual(learner._step_count, 6)
        self.assertEqual(learner._gradient_updates, 2)
        self.assertEqual(learner._last_gradient_update_step, 6)
        self.assertTrue(apply_calls[5].applied_gradients, "Sixth step should apply gradients")

        # Test that the pattern continues correctly
        expected_apply_steps = {3, 6, 9, 12}
        for i in range(7, 13):
            algorithm.step()
            expected_applies = len([s for s in expected_apply_steps if s <= i])
            self.assertEqual(learner._gradient_updates, expected_applies, f"Wrong gradient updates at step {i}")
            if i in expected_apply_steps:
                self.assertEqual(learner._last_gradient_update_step, i, f"Wrong last update step at step {i}")

    def test_ppo_torch_learner_accumulation_sums_gradients(self):
        """
        Test that gradients are summed up during accumulation and applied correctly.

        NOTE:
            When apply_gradients is called, the .grad attribute is replaced by grad / accumulate_gradients_every
            via in-place division and further processed by postprocessing.
        """
        batch_size = 32
        accumulate_every = 2

        with patch_args(
            "--accumulate_gradients_every", accumulate_every,
            "--batch_size", batch_size,
            "--minibatch_size", batch_size,
            "--fcnet_hiddens", "[8, 8]",
        ):  # fmt: skip
            setup = MLPSetup(init_trainable=False)

        setup.config.training(num_epochs=1)
        if accumulate_every == 1:  # pyright: ignore[reportUnnecessaryComparison]
            setup.config.training(
                learner_class=mix_learners([PPOTorchLearnerWithGradientAccumulation, RemoveMaskedSamplesLearner])
            )
        algorithm = setup.config.build_algo()
        learner: PPOTorchLearnerWithGradientAccumulation = (
            algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        )

        # Get a reference to a parameter to track its gradients
        param_ref, param_tensor = next(iter(learner._params.items()))

        # Track raw gradients for each step
        raw_gradients = []
        unaccumulated_grads = []
        original_compute_gradients = learner.compute_gradients

        updated_gradients = []

        def mock_compute_gradients(loss_per_module: dict[str, torch.Tensor], **kwargs):
            grads_backup = {
                ref: param.grad.clone() if param.grad is not None else None for ref, param in learner._params.items()
            }
            for optim in learner._optimizer_parameters:
                optim.zero_grad(set_to_none=True)
            for loss in loss_per_module.values():
                loss.backward(retain_graph=True)
            unaccumulated_grads.append(
                param_tensor.grad.clone()  # pyright: ignore[reportOptionalMemberAccess]
            )  # pyright: ignore[reportOptionalMemberAccess] # This is only accumulated for one step
            for ref, grad in grads_backup.items():
                learner._params[ref].grad = grad
            result = original_compute_gradients(loss_per_module, **kwargs)  # pyright: ignore[reportArgumentType]
            # After compute_gradients, grad has been accumulated and possibly divided (in-place)
            assert param_tensor.grad is not None
            raw_gradients.append(param_tensor.grad.detach().clone())
            # unaccumulated and raw_gradients (accumulated) should never match, except when set to None one step later
            with (
                self.assertRaises(AssertionError, msg=f"Gradients matched at step {learner._step_count}")
                if accumulate_every != 1  # pyright: ignore[reportUnnecessaryComparison]
                and learner._step_count % accumulate_every != 1
                else nullcontext()
            ):
                torch.testing.assert_close(
                    unaccumulated_grads[-1], raw_gradients[-1], msg=f"Gradients matched at step {learner._step_count}"
                )

            # grad already divided:
            updated_gradients.append(
                {param_ref: result[param_ref].detach().clone()}  # pyright: ignore[reportAttributeAccessIssue]
                if len(result) > 0
                else {}
            )
            return result

        learner.compute_gradients = mock_compute_gradients  # pyright: ignore[reportAttributeAccessIssue]
        # Track gradient application calls
        original_apply_gradients = learner.apply_gradients

        apply_calls: list[_ApplyCall] = []

        def mock_apply_gradients(gradients_dict):
            apply_calls.append(
                _ApplyCall(
                    step_count=learner._step_count,
                    applied_gradients=len(gradients_dict) > 0,
                    gradients={param_ref: gradients_dict[param_ref].clone()} if len(gradients_dict) > 0 else {},
                )
            )
            return original_apply_gradients(gradients_dict)

        learner.apply_gradients = mock_apply_gradients
        # learner.postprocess_gradients = lambda g: g  # No postprocessing for this test

        # Step 1: accumulate gradients
        algorithm.step()
        self.assertEqual(learner._step_count, 1)
        self.assertEqual(learner._gradient_updates, 0)

        self.assertEqual(learner._step_count % setup.config.learner_config_dict["accumulate_gradients_every"], 1)
        self.assertEqual(updated_gradients[0], {}, "Gradients should not be applied in step 1")
        self.assertIs(param_tensor, learner._params[param_ref], "Parameter reference should match")
        grad_step1 = unaccumulated_grads[0]

        # Step 2: accumulate and apply gradients
        algorithm.step()
        self.assertEqual(learner._step_count, 2)
        self.assertEqual(learner._gradient_updates, 1)
        self.assertEqual(learner._step_count % setup.config.learner_config_dict["accumulate_gradients_every"], 0)
        self.assertNotEqual(updated_gradients[1], {}, "Gradients should have been applied in step 2")
        self.assertIs(param_tensor, learner._params[param_ref], "Parameter reference should match")

        grad_step2 = unaccumulated_grads[1]  # grad is accumulated

        # With in-place division, raw_gradients[1] now contains the already-divided gradient
        # So it should equal (grad_step1 + grad_step2) / accumulate_every
        manual_sum = grad_step1 + grad_step2
        manual_mean = manual_sum / accumulate_every

        torch.testing.assert_close(
            raw_gradients[1],
            manual_mean,
            msg="Raw gradient should be the mean (sum/accumulate_every) after in-place division",
        )

        # The updated_gradients should match the raw_gradients since they're the same reference
        torch.testing.assert_close(
            updated_gradients[1][param_ref],
            manual_mean,
            msg="Applied gradient should be the mean of accumulated gradients",
        )


class TestDQNGradientAccumulation(TestHelpers, DisableLoggers):
    def test_dqn_torch_learner_with_gradient_accumulation(self):
        """
        Test that DQNTorchLearnerWithGradientAccumulation accumulates gradients and applies them only at the correct steps.
        """
        batch_size = 32
        accumulate_every = 3
        with patch_args(
            "--algorithm",
            "dqn",
            "-a",
            "mlp",
            "--accumulate_gradients_every",
            str(accumulate_every),
            "--batch_size",
            str(batch_size),
            "--minibatch_size",
            str(batch_size),
            "--fcnet_hiddens",
            "[8, 8]",
        ):
            setup = DQNMLPSetup(init_trainable=False)
        setup.config.training(num_epochs=1)
        algorithm = setup.config.build_algo()
        learner: DQNTorchLearnerWithGradientAccumulation = (
            algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        )
        self.assertIsInstance(learner, DQNTorchLearnerWithGradientAccumulation)
        # Track gradient application calls
        original_apply_gradients = learner.apply_gradients
        apply_calls: list[dict] = []

        def mock_apply_gradients(gradients_dict):
            apply_calls.append(
                {
                    "step_count": learner._step_count,
                    "applied_gradients": len(gradients_dict) > 0,
                }
            )
            return original_apply_gradients(gradients_dict)

        learner.apply_gradients = mock_apply_gradients
        # Track parameter changes
        module = learner.module["default_policy"]
        initial_state = module.get_state()
        # Step 1: accumulate, not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 1)
        self.assertEqual(learner._gradient_updates, 0)
        self.assertIsNone(learner._last_gradient_update_step)
        self.assertEqual(len(apply_calls), 1)
        self.assertFalse(apply_calls[0]["applied_gradients"])
        state_after_step1 = module.get_state()
        self.util_test_tree_equivalence(initial_state, state_after_step1)
        # Step 2: accumulate, not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 2)
        self.assertEqual(learner._gradient_updates, 0)
        self.assertIsNone(learner._last_gradient_update_step)
        self.assertEqual(len(apply_calls), 2)
        self.assertFalse(apply_calls[1]["applied_gradients"])
        state_after_step2 = module.get_state()
        self.util_test_tree_equivalence(initial_state, state_after_step2)
        # Step 3: accumulate and apply
        algorithm.step()
        self.assertEqual(learner._step_count, 3)
        self.assertEqual(learner._gradient_updates, 1)
        self.assertEqual(learner._last_gradient_update_step, 3)
        self.assertEqual(len(apply_calls), 3)
        self.assertTrue(apply_calls[2]["applied_gradients"])
        state_after_step3 = module.get_state()
        with self.assertRaisesRegex(AssertionError, "(weight|bias).* not equal"):
            self.util_test_tree_equivalence(initial_state, state_after_step3, use_subtests=False)
        # Step 4: new cycle, not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 4)
        self.assertEqual(learner._gradient_updates, 1)
        self.assertEqual(learner._last_gradient_update_step, 3)
        self.assertEqual(len(apply_calls), 4)
        self.assertFalse(apply_calls[3]["applied_gradients"])
        state_after_step4 = module.get_state()
        self.util_test_tree_equivalence(state_after_step3, state_after_step4)
        # Step 5: accumulate, not apply
        algorithm.step()
        self.assertEqual(learner._step_count, 5)
        self.assertEqual(learner._gradient_updates, 1)
        # Step 6: apply again
        algorithm.step()
        self.assertEqual(learner._step_count, 6)
        self.assertEqual(learner._gradient_updates, 2)
        self.assertEqual(learner._last_gradient_update_step, 6)
        self.assertTrue(apply_calls[5]["applied_gradients"])
        # Continue pattern
        expected_apply_steps = {3, 6, 9, 12}
        for i in range(7, 13):
            algorithm.step()
            expected_applies = len([s for s in expected_apply_steps if s <= i])
            self.assertEqual(learner._gradient_updates, expected_applies, f"Wrong gradient updates at step {i}")
            if i in expected_apply_steps:
                self.assertEqual(learner._last_gradient_update_step, i, f"Wrong last update step at step {i}")

    def test_dqn_torch_learner_accumulation_sums_gradients(self):
        """
        Test that DQNTorchLearnerWithGradientAccumulation sums gradients over steps and applies the mean at the correct time.
        """
        batch_size = 32
        accumulate_every = 2
        with patch_args(
            "--algorithm",
            "dqn",
            "-a",
            "mlp",
            "--accumulate_gradients_every",
            str(accumulate_every),
            "--batch_size",
            str(batch_size),
            "--minibatch_size",
            str(batch_size),
            "--fcnet_hiddens",
            "[8, 8]",
        ):
            setup = DQNMLPSetup(init_trainable=False)
        setup.config.training(num_epochs=1)
        algorithm = setup.config.build_algo()
        learner: DQNTorchLearnerWithGradientAccumulation = (
            algorithm.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
        )
        param_ref, param_tensor = next(iter(learner._params.items()))
        raw_gradients = []
        unaccumulated_grads = []
        original_compute_gradients = learner.compute_gradients
        updated_gradients = []

        def mock_compute_gradients(loss_per_module: dict, **kwargs):
            grads_backup = {
                ref: param.grad.clone() if param.grad is not None else None for ref, param in learner._params.items()
            }
            for optim in learner._optimizer_parameters:
                optim.zero_grad(set_to_none=True)
            for loss in loss_per_module.values():
                loss.backward(retain_graph=True)
            unaccumulated_grads.append(param_tensor.grad.clone())
            for ref, grad in grads_backup.items():
                learner._params[ref].grad = grad
            result = original_compute_gradients(loss_per_module, **kwargs)
            raw_gradients.append(param_tensor.grad.detach().clone())
            # unaccumulated and raw_gradients (accumulated) should never match, except when set to None one step later
            if accumulate_every != 1 and learner._step_count % accumulate_every != 1:
                with self.assertRaises(AssertionError, msg=f"Gradients matched at step {learner._step_count}"):
                    torch.testing.assert_close(
                        unaccumulated_grads[-1],
                        raw_gradients[-1],
                        msg=f"Gradients matched at step {learner._step_count}",
                    )
            updated_gradients.append({param_ref: result[param_ref].detach().clone()} if len(result) > 0 else {})
            return result

        learner.compute_gradients = mock_compute_gradients
        original_apply_gradients = learner.apply_gradients
        apply_calls: list[dict] = []

        def mock_apply_gradients(gradients_dict):
            apply_calls.append(
                {
                    "step_count": learner._step_count,
                    "applied_gradients": len(gradients_dict) > 0,
                    "gradients": {param_ref: gradients_dict[param_ref].clone()} if len(gradients_dict) > 0 else {},
                }
            )
            return original_apply_gradients(gradients_dict)

        learner.apply_gradients = mock_apply_gradients
        # Step 1: accumulate
        algorithm.step()
        self.assertEqual(learner._step_count, 1)
        self.assertEqual(learner._gradient_updates, 0)
        self.assertEqual(learner._step_count % setup.config.learner_config_dict["accumulate_gradients_every"], 1)
        self.assertEqual(updated_gradients[0], {}, "Gradients should not be applied in step 1")
        self.assertIs(param_tensor, learner._params[param_ref], "Parameter reference should match")
        grad_step1 = unaccumulated_grads[0]
        # Step 2: accumulate and apply
        algorithm.step()
        self.assertEqual(learner._step_count, 2)
        self.assertEqual(learner._gradient_updates, 1)
        self.assertEqual(learner._step_count % setup.config.learner_config_dict["accumulate_gradients_every"], 0)
        self.assertNotEqual(updated_gradients[1], {}, "Gradients should have been applied in step 2")
        self.assertIs(param_tensor, learner._params[param_ref], "Parameter reference should match")
        grad_step2 = unaccumulated_grads[1]
        manual_sum = grad_step1 + grad_step2
        torch.testing.assert_close(
            raw_gradients[1], manual_sum, msg="Raw accumulated gradient should be the sum of step gradients"
        )
        torch.testing.assert_close(
            updated_gradients[1][param_ref],
            manual_sum / accumulate_every,
            msg="Applied gradient should be the mean of accumulated gradients",
        )
