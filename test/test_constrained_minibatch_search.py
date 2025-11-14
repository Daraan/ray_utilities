import pytest
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Categorical, Integer, Float
from ray_utilities.tune.searcher.constrained_minibatch_search import constrained_minibatch_search


class DummyArgs:
    def __init__(
        self,
        train_batch_size_per_learner=64,
        num_samples=1,
        not_parallel=False,  # noqa: FBT002
        num_jobs=2,
    ):
        self.train_batch_size_per_learner = train_batch_size_per_learner
        self.num_samples = num_samples
        self.not_parallel = not_parallel
        self.num_jobs = num_jobs


class DummySetup:
    def __init__(self, param_space, args):
        self.param_space = param_space
        self.args = args


@pytest.mark.parametrize(
    "minibatch_grid,train_batch_size",
    [
        ([16, 32, 64, 128], 64),
        ([8, 16, 32], 16),
    ],
)
def test_minibatch_grid_dict_with_fixed_train_batch_size(minibatch_grid, train_batch_size):
    param_space = {
        "minibatch_size": {"grid_search": minibatch_grid},
    }
    args = DummyArgs(train_batch_size_per_learner=train_batch_size, num_samples=2)
    setup = DummySetup(param_space, args)
    gen = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    expected = [v for v in minibatch_grid if v <= train_batch_size]
    assert isinstance(gen, BasicVariantGenerator)
    points = gen._points_to_evaluate
    assert all(d["minibatch_size"] in expected for d in points)
    assert len(points) == len(expected) * args.num_samples


def test_minibatch_grid_dict_missing_grid_search(caplog):
    param_space = {
        "minibatch_size": {"not_grid_search": [1, 2, 3]},
    }
    args = DummyArgs()
    setup = DummySetup(param_space, args)
    with caplog.at_level("WARNING"):
        result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    assert "minibatch_size param space is a dict but does not contain grid_search" in caplog.text


def test_minibatch_categorical_with_fixed_train_batch_size():
    param_space = {
        "minibatch_size": Categorical([8, 16, 32, 64, 128]),
    }
    args = DummyArgs(train_batch_size_per_learner=32)
    setup = DummySetup(param_space, args)
    result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    assert param_space["minibatch_size"].categories == [8, 16, 32]


def test_minibatch_integer_with_fixed_train_batch_size():
    param_space = {
        "minibatch_size": Integer(8, 128),
    }
    args = DummyArgs(train_batch_size_per_learner=32)
    setup = DummySetup(param_space, args)
    result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    assert param_space["minibatch_size"].upper == 32


def test_both_grid_search_dicts():
    param_space = {
        "minibatch_size": {"grid_search": [8, 16, 32, 64]},
        "train_batch_size_per_learner": {"grid_search": [16, 32]},
    }
    args = DummyArgs(num_samples=1)
    setup = DummySetup(param_space, args)
    gen = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert isinstance(gen, BasicVariantGenerator)
    expected = [
        {"minibatch_size": v, "train_batch_size_per_learner": bs} for bs in [16, 32] for v in [8, 16, 32, 64] if v <= bs
    ]
    assert all(d in expected for d in gen._points_to_evaluate)
    assert len(gen._points_to_evaluate) == len(expected)


def test_train_batch_size_dict_missing_grid_search(caplog):
    param_space = {
        "minibatch_size": {"grid_search": [8, 16]},
        "train_batch_size_per_learner": {"not_grid_search": [16, 32]},
    }
    args = DummyArgs()
    setup = DummySetup(param_space, args)
    with caplog.at_level("WARNING"):
        result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    assert "train_batch_size_per_learner param space is a dict but does not contain grid_search" in caplog.text


def test_train_batch_size_categorical_and_minibatch_categorical():
    param_space = {
        "minibatch_size": Categorical([8, 16, 32, 64]),
        "train_batch_size_per_learner": Categorical([16, 32]),
    }
    args = DummyArgs()
    setup = DummySetup(param_space, args)
    result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    # Should be limited to <= max(train_batch_size_per_learner)
    assert param_space["minibatch_size"].categories == [8, 16, 32]


def test_train_batch_size_integer_and_minibatch_float():
    param_space = {
        "minibatch_size": Float(8, 128),
        "train_batch_size_per_learner": Integer(16, 64),
    }
    args = DummyArgs()
    setup = DummySetup(param_space, args)
    result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    # Should be limited to upper bound of train_batch_size_per_learner
    assert param_space["minibatch_size"].upper == 64


def test_unknown_train_batch_size_type(caplog):
    class DummyDomain:
        pass

    param_space = {
        "minibatch_size": {"grid_search": [8, 16]},
        "train_batch_size_per_learner": DummyDomain(),
    }
    args = DummyArgs()
    setup = DummySetup(param_space, args)
    with caplog.at_level("WARNING"):
        result = constrained_minibatch_search(setup)  # pyright: ignore[reportArgumentType]
    assert result is None
    assert "train_batch_size_per_learner param space is of unknown type" in caplog.text
