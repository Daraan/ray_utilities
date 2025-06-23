from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger
import typing_extensions as te
from .utils import SetupDefaults, patch_args
from ray_utilities.config import DefaultArgumentParser


class TestSetupClasses(SetupDefaults):
    @patch_args()
    def test_basic(self):
        setup = AlgorithmSetup()
        self.assertIsNotNone(setup.config)
        self.assertIsNotNone(setup.args)
        self.assertIsNotNone(setup.create_tuner())
        self.assertIsNotNone(setup.create_config())
        self.assertIsNotNone(setup.create_param_space())
        self.assertIsNotNone(setup.create_parser())
        self.assertIsNotNone(setup.create_tags())

    def test_argument_usage(self):
        # Test warning and failure
        with patch_args("--batch_size", "1234"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 1234)
        with patch_args("--train_batch_size_per_learner", "456"):
            self.assertEqual(AlgorithmSetup().config.train_batch_size_per_learner, 456)

    def test_dynamic_param_spaces(self):
        # Test warning and failure
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
            with patch_args("--tune", param), self.assertNoLogs(logger, level="WARNING"):
                if param == "batch_size":  # shortcut name
                    param = "train_batch_size_per_learner"  # noqa: PLW2901
                setup = AlgorithmSetup()
                param_space = setup.create_param_space()
                self.assertIn(param, param_space)
                self.assertIsNotNone(param_space[param])  # dict with list

                def fake_trainable(params):
                    return params

                setup.create_trainable = lambda _self: fake_trainable  # type: ignore
                tuner = setup.create_tuner()
                tuner.fit()
