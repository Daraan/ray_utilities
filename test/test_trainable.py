from functools import partial
import tempfile

import ray
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.utils import validate_save_restore
from ray.tune.search.sample import Domain, Categorical

from ray_utilities.setup.algorithm_setup import PPOSetup
from ray_utilities.testing_utils import DisableGUIBreakpoints, DisableLoggers, TestHelpers, patch_args
from ray_utilities.training.default_class import DefaultTrainable
import pickle
import io


class TestTraining(TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    @patch_args()
    def test_trainable_function(self):
        # with self.subTest("No parameters"):
        #    _result = trainable({})
        with self.subTest("With parameters"):
            setup = PPOSetup(init_param_space=True)
            setup.config.evaluation(evaluation_interval=1)
            setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
            trainable = setup.create_trainable()
            params = setup.sample_params()
            _result = trainable(params)

    @patch_args()
    def test_trainable_class(self):
        Trainable = DefaultTrainable.define(PPOSetup.typed())
        trainable = Trainable()
        trainable.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
        _result = trainable.step()
        trainable.cleanup()

    def test_trainable_class_save_checkpoint(self):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        with patch_args("--iterations", "5", "--total_steps", "320", "--batch_size", "64"):
            Trainable = DefaultTrainable.define(PPOSetup.typed())
            trainable = Trainable()
        trainable.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)

        _result1 = trainable.step()
        with tempfile.TemporaryDirectory() as tmpdir:
            # NOTE This loads some parts by identity!
            saved_ckpt = trainable.save_checkpoint(tmpdir)
            from copy import deepcopy
            saved_ckpt = deepcopy(saved_ckpt)  # make sure we do not modify the original
            # pickle and load
            if False:
                # fails because of:  AttributeError: Can't pickle local object 'mix_learners.<locals>.MixedLearner'
                # Serialize saved_ckpt to a BytesIO object
                buf = io.BytesIO()
                pickle.dump(saved_ckpt, buf)
                buf.seek(0)
                # Deserialize from BytesIO
                saved_ckpt = pickle.load(buf)
            with patch_args():  # make sure that args do not influence the restore
                trainable2 = Trainable()
                trainable2.load_checkpoint(saved_ckpt)
        self.maxDiff = 60_000
        self.assertDictEqual(trainable2.args, trainable.args)
        self.assertEqual(trainable.config.minibatch_size, 32)
        self.assertEqual(trainable2.config.minibatch_size, trainable.config.minibatch_size)
        self.assertEqual(trainable2._iteration, trainable._iteration)

        self.assertDictEqual(trainable2.config.to_dict(), trainable.config.to_dict())
        setup_data1 = trainable._setup.save()  # does not compare setup itself
        setup_data2 = trainable2._setup.save()
        # check all keys
        self.assertEqual(setup_data1.keys(),  setup_data2.keys())
        keys = set(setup_data1.keys())
        self.assertDictEqual(vars(setup_data1["args"]), vars(setup_data2["args"]))  # SimpleNamespace
        keys.remove("args")
        self.assertIs(setup_data1["class"], setup_data2["class"])
        keys.remove("class")
        self.assertEqual(setup_data1["config"].to_dict(), setup_data2["config"].to_dict()) # AlgorithmConfig
        keys.remove("config")
        param_space1 = setup_data1["param_space"]
        param_space2 = setup_data2["param_space"]
        keys.remove("param_space")
        self.assertEqual(len(keys), 0) # checked all params
        self.assertCountEqual(param_space1, param_space2)
        self.assertDictEqual(param_space1["cli_args"], param_space2["cli_args"])
        for key in param_space1.keys() | param_space2.keys():
            value1 = param_space1[key]
            value2 = param_space2[key]
            if isinstance(value1, Domain) or isinstance(value2, Domain):
                # Domain is not hashable, so we cannot compare them directly
                self.assertIs(type(value1), type(value2))
                if isinstance(value1, Categorical):
                    assert isinstance(value2, Categorical)
                    self.assertListEqual(value1.categories, value2.categories)
                else:
                    # This will likely fail, need to compare attributes
                    self.assertEqual(value1, value2, f"Domain {key} differs: {value1} != {value2}")
            else:
                self.assertEqual(value1, value2, f"Parameter {key} differs: {value1} != {value2}")

        # Compare attrs
        self.assertIsNot(trainable2._reward_updaters, trainable._reward_updaters)
        for key in trainable2._reward_updaters.keys() | trainable._reward_updaters.keys():
            updater1 = trainable._reward_updaters[key]
            updater2 = trainable2._reward_updaters[key]
            self.assertIsNot(updater1, updater2)
            assert isinstance(updater1, partial) and isinstance(updater2, partial)
            self.assertDictEqual(updater1.keywords, updater2.keywords)
            self.assertIsNot(updater1.keywords["reward_array"], updater2.keywords["reward_array"])

        self.assertIsNot(trainable2._pbar, trainable._pbar)
        self.assertEqual(trainable2._pbar._get_state(), trainable._pbar._get_state())

        # Step 2
        result2 = trainable.step()
        result2_restored = trainable2.step()
        self.assertEqual(result2[TRAINING_ITERATION], result2_restored[TRAINING_ITERATION])
        self.assertEqual(result2[TRAINING_ITERATION], 2)

    def test_validate_save_restore(self):
        """Basically test if TRAINING_ITERATION is set correctly."""
        ray.init(include_dashboard=False, ignore_reinit_error=True)

        with patch_args("--iterations", "5", "--total_steps", "320", "--batch_size", "64"):
            # Need to fix argv for remote
            PPOTrainable = DefaultTrainable.define(PPOSetup.typed(), fix_argv=True)
            trainable = PPOTrainable()
            self.assertEqual(trainable.args["iterations"], 5)
            self.assertEqual(trainable.args["total_steps"], 320)
            validate_save_restore(PPOTrainable)
        ray.shutdown()
