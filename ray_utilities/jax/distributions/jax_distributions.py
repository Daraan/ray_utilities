from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple, Union, cast

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tree
from distrax import Distribution as DistraxDistribution
from ray.rllib.core.distribution.distribution import Distribution as RllibDistribution
from tensorflow_probability.substrates import jax as tfp
from typing_extensions import Self

if TYPE_CHECKING:
    import chex
    import gymnasium as gym
    from ray.rllib.utils.typing import TensorType

    IntLike = Union[int, np.int16, np.int32, np.int64]
    from tensorflow_probability.substrates.jax import distributions as __tfd

logger = logging.getLogger(__name__)


def _get_tfp_distributions():
    """Type hint support for return value"""
    if TYPE_CHECKING:
        return __tfd
    return tfp.distributions


class RLlibToJaxDistribution(RllibDistribution, distrax.Distribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist = self._get_jax_distribution(*args, **kwargs)

    @abstractmethod
    def _get_jax_distribution(self, *args, **kwargs) -> distrax.Distribution: ...

    def sample(
        self,
        *,
        seed: IntLike | chex.PRNGKey,  # XXX <-- not RLlib interface
        sample_shape: Union[IntLike, Sequence[IntLike]] = (),
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, tuple[TensorType, TensorType]]:
        if not return_logp:
            return self._dist.sample(seed=seed, sample_shape=sample_shape, **kwargs)
        return self._dist.sample_and_log_prob(seed=seed, sample_shape=sample_shape, **kwargs)

    def logp(self, value: TensorType, **kwargs) -> TensorType:
        return self._dist.log_prob(value, **kwargs)

    def kl(self, other: "RLlibToJaxDistribution | distrax.DistributionLike", **kwargs) -> TensorType:
        if isinstance(other, RLlibToJaxDistribution):
            other = other._dist
        return self._dist.kl_divergence(other, **kwargs)

    @abstractmethod
    def to_deterministic(self) -> Deterministic: ...


class SupportsLogitsMixin(RllibDistribution):
    @classmethod
    def from_logits(cls, logits: TensorType, **kwargs):
        return cls(logits=logits, **kwargs)  # type: ignore[call-arg]


class _EntropyArgCorrector(RllibDistribution):
    def entropy(self, **kwargs) -> TensorType:
        return cast(
            "DistraxDistribution",
            self._dist,  # type: ignore[attr-defined]
        ).entropy(**kwargs)


class Normal(_EntropyArgCorrector, distrax.Normal, RLlibToJaxDistribution):
    def __init__(self, loc: chex.Numeric, scale: chex.Numeric):
        super().__init__(loc, scale)
        self._dist: distrax.Normal

    def _get_jax_distribution(self, *args, **kwargs) -> distrax.Distribution:
        return distrax.Normal(*args, **kwargs)

    def rsample(
        self,
        *,
        seed: IntLike | chex.PRNGKey,  # XXX <-- not RLlib interface
        sample_shape: Tuple[int, ...] = (),
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        eps = jax.random.normal(seed, sample_shape, **kwargs)
        if not return_logp:
            return self._dist.loc + eps * self._dist.scale
        # possibly use _sample_from_std_normal
        raise NotImplementedError("rsample with return_logp not implemented")

    @staticmethod
    def required_input_dim(space: gym.Space | gym.spaces.Box, **kwargs) -> int:  # noqa: ARG004
        assert space.shape is not None
        return int(np.prod(space.shape, dtype=np.int32))

    def to_deterministic(self) -> Deterministic:
        return Deterministic(loc=self.loc)


if TYPE_CHECKING:
    Normal(1, 0)


class Categorical(_EntropyArgCorrector, SupportsLogitsMixin, RLlibToJaxDistribution, distrax.Categorical):
    def __init__(
        self,
        probs: Optional[chex.Array] = None,
        logits: Optional[chex.Array] = None,
        dtype: Union[jnp.dtype, type[Any]] = int,
    ):
        # Problem super call call to distrax.Categorical does not forward to RLlibDistribution
        super().__init__(probs=probs, logits=logits, dtype=dtype)
        self.one_hot = _get_tfp_distributions().OneHotCategorical(logits=logits, probs=probs)

    def _get_jax_distribution(
        self,
        probs: Optional[chex.Array] = None,
        logits: Optional[chex.Array] = None,
        dtype: Union[jnp.dtype, type[Any]] = int,
    ) -> distrax.Distribution:
        self._dist: distrax.Categorical
        return distrax.Categorical(probs=probs, logits=logits, dtype=dtype)

    def rsample(self, *, sample_shape=(), return_logp: bool = False, seed=None, **kwargs):
        if return_logp:
            one_hot_sample, logp = self.one_hot.experimental_sample_and_log_prob(sample_shape, seed=seed, **kwargs)
            return jax.lax.stop_gradient(one_hot_sample - self.probs) + self.probs, logp
        one_hot_sample = self.one_hot.sample(sample_shape, seed=seed, **kwargs)
        return jax.lax.stop_gradient(one_hot_sample - self.probs) + self.probs

    def logp(self, value: TensorType, **kwargs) -> TensorType:
        return self._dist.log_prob(value, **kwargs)

    def kl(self, other: distrax.DistributionLike, **kwargs) -> TensorType:
        if isinstance(other, RLlibToJaxDistribution):
            other = other._dist
        return self._dist.kl_divergence(other, **kwargs)

    @staticmethod
    def required_input_dim(space: gym.Space | gym.spaces.Discrete, **kwargs) -> int:  # noqa: ARG004
        return int(space.n)  # type: ignore[attr-defined]§

    def to_deterministic(self) -> Deterministic:
        if self.probs is not None:
            probs_or_logits = self.probs
        else:
            probs_or_logits = self.logits

        return Deterministic(loc=jnp.argmax(probs_or_logits, axis=-1))


if TYPE_CHECKING:
    Categorical(None, None)


class Deterministic(_EntropyArgCorrector, RLlibToJaxDistribution, distrax.Deterministic):
    def __init__(self, loc: chex.Array, atol: Optional[chex.Numeric] = None, rtol: Optional[chex.Numeric] = None):
        super().__init__(loc=loc, atol=atol, rtol=rtol)

    def _get_jax_distribution(
        self, loc: chex.Array, atol: Optional[chex.Numeric] = None, rtol: Optional[chex.Numeric] = None
    ) -> distrax.Distribution:
        self._dist: distrax.Deterministic
        return distrax.Deterministic(loc=loc, atol=atol, rtol=rtol)

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:  # noqa: ARG004
        assert space.shape is not None
        return int(np.prod(space.shape, dtype=np.int32))

    def rsample(self, *, sample_shape=(), return_logp: bool = False, seed=None, **kwargs):
        raise NotImplementedError

    def to_deterministic(self) -> Self:
        return self


if TYPE_CHECKING:
    Deterministic(jnp.array(1))


# Note: Torch distribution works better with various structures
class MultiCategorical(RLlibToJaxDistribution):
    def __init__(
        self,
        categoricals: list[Categorical],
    ):
        RLlibToJaxDistribution.__init__(self)
        self._cats = categoricals

    def _get_jax_distribution(self) -> distrax.Distribution:
        self._dist: None
        return None

    def sample(self, *, seed, **kwargs) -> TensorType:
        if kwargs.get("return_logp", False):
            raise NotImplementedError("return_logp not implemented")
        arr: list[chex.Array] = [cat.sample(seed=seed) for cat in self._cats]  # pyright: ignore[reportAssignmentType]
        sample_ = jnp.stack(arr, axis=-1)
        return sample_

    def rsample(self, sample_shape=(), *seed, **kwargs):  # noqa: ARG002  # no torch/tf class uses sample shape here
        arr: list[chex.Array] = [cat.rsample(seed=seed) for cat in self._cats]  # type: ignore[attr-defined]
        sample_ = jnp.stack(arr, axis=-1)
        return sample_

    def logp(self, value: chex.Array, **kwargs) -> TensorType:  # noqa: ARG002
        actions = jnp.unstack(jnp.astype(value, jnp.int32), axis=-1)
        logps = jnp.stack([cat.logp(act) for cat, act in zip(self._cats, actions)])
        return jnp.sum(logps, axis=0)

    def entropy(self, **kwargs) -> TensorType:  # noqa: ARG002
        return jnp.sum(jnp.stack([cat.entropy() for cat in self._cats], axis=-1), axis=-1)

    def kl(self, other: MultiCategorical, **kwargs) -> TensorType:  # noqa: ARG002  # type: ignore[override]
        kls = jnp.stack([cat.kl(oth_cat) for cat, oth_cat in zip(self._cats, other._cats)], axis=-1)
        return jnp.sum(kls, axis=-1)

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:  # noqa: ARG004
        # assert isinstance(space, gym.spaces.MultiDiscrete)
        return int(np.sum(space.nvec))  # type: ignore[attr-defined]

    @classmethod
    def from_logits(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        logits: chex.Array,
        input_lens: list[int],
        **kwargs,  # noqa: ARG003
    ) -> "MultiCategorical":
        """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by logit_lens.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list of integers that indicate the length of the logits
                vectors to be passed into each child distribution.
            **kwargs: Forward compatibility kwargs.
        """
        categoricals = [Categorical(logits=logits) for logits in jnp.split(logits, input_lens, axis=-1)]

        return cls(categoricals=categoricals)

    def to_deterministic(self) -> "MultiDistribution":  # type: ignore
        return MultiDistribution([cat.to_deterministic() for cat in self._cats])

    def log_prob(self, value):
        return self.logp(value)

    def _sample_n(self, key: jax.Array, n: int) -> Any:
        raise NotImplementedError("use sample method instead")

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return ()


if TYPE_CHECKING:
    MultiCategorical([Categorical(None, None), Categorical(None, None)])


class MultiDistribution(RLlibToJaxDistribution):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(
        self,
        child_distribution_struct: Union[tuple, list, dict],
    ):
        """Initializes a TfMultiDistribution object.

        Args:
            child_distribution_struct: Any struct
                that contains the child distribution classes to use to
                instantiate the child distributions from `logits`.
        """
        super().__init__()
        self._original_struct = child_distribution_struct
        self._flat_child_distributions: Sequence[RLlibToJaxDistribution] = tree.flatten(child_distribution_struct)

    def _get_jax_distribution(self, *args, **kwargs) -> None:  # noqa: ARG002
        self._dist: None
        return None  # noqa

    def rsample(
        self,
        *,
        sample_shape: Optional[tuple[int, ...]] = None,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        rsamples = []
        for dist in self._flat_child_distributions:
            rsample = dist.rsample(sample_shape=sample_shape, **kwargs)  # pyright: ignore[reportArgumentType] super has implicit optional
            rsamples.append(rsample)

        rsamples = tree.unflatten_as(self._original_struct, rsamples)
        return rsamples

    def logp(self, value, **kwargs) -> TensorType:  # noqa: ARG002
        # Single tensor input (all merged).
        if isinstance(value, (jax.Array, np.ndarray)):
            split_indices = []
            for dist in self._flat_child_distributions:
                if isinstance(dist, Categorical):
                    split_indices.append(1)
                elif isinstance(dist, MultiCategorical):
                    split_indices.append(len(dist._cats))
                else:
                    sample: chex.Array = dist.sample(seed=4423)  # XXX
                    # Cover Box(shape=()) case.
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(jnp.shape(sample)[1])
            split_value = jnp.split(value, split_indices, axis=1)
        # Structured or flattened (by single action component) input.
        else:
            split_value = tree.flatten(value)

        def map_(val: chex.Array, dist: RLlibToJaxDistribution):
            # Remove extra dimension if present.
            if isinstance(dist, Categorical) and len(val.shape) > 1 and val.shape[-1] == 1:
                val = jnp.squeeze(val, axis=-1)

            return dist.logp(val)

        # Remove extra categorical dimension and take the logp of each
        # component.
        flat_logps = tree.map_structure(map_, split_value, self._flat_child_distributions)

        return sum(flat_logps)  # type: ignore

    def kl(self, other: MultiDistribution, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        kl_list = [d.kl(o, **kwargs) for d, o in zip(self._flat_child_distributions, other._flat_child_distributions)]
        return sum(kl_list)

    def entropy(self, **kwargs):
        entropy_list = [d.entropy(**kwargs) for d in self._flat_child_distributions]
        return sum(entropy_list)

    def sample(self, *, seed, **kwargs):
        child_distributions_struct = tree.unflatten_as(self._original_struct, self._flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(seed=seed, **kwargs), child_distributions_struct)

    @staticmethod
    def required_input_dim(space: gym.Space, input_lens: list[int], **kwargs) -> int:  # noqa: ARG004  # pyright: ignore[reportIncompatibleMethodOverride]
        return sum(input_lens)

    @classmethod
    def from_logits(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        logits: chex.Array,
        child_distribution_cls_struct: Union[dict, Iterable],
        input_lens: Union[dict, list[int]],
        **kwargs,  # noqa: ARG003
    ) -> "MultiDistribution":
        """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by `input_lens`.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list or dict of integers that indicate the length of each
                logit. If this is given as a dict, the structure should match the
                structure of child_distribution_cls_struct.
            space: The possibly nested output space.
            **kwargs: Forward compatibility kwargs.

        Returns:
            A TfMultiDistribution object.
        """
        logit_lens = tree.flatten(input_lens)
        child_distribution_cls_list: Sequence[RllibDistribution] = tree.flatten(child_distribution_cls_struct)
        split_logits = jnp.split(logits, logit_lens, axis=1)

        child_distribution_list = tree.map_structure(
            lambda dist, input_: dist.from_logits(input_),
            child_distribution_cls_list,
            list(split_logits),
        )

        child_distribution_struct: Any | dict | list = tree.unflatten_as(
            child_distribution_cls_struct, child_distribution_list
        )

        return MultiDistribution(
            child_distribution_struct=child_distribution_struct,
        )

    def to_deterministic(self) -> "MultiDistribution":  # pyright: ignore[reportIncompatibleMethodOverride]
        flat_deterministic_dists = [dist.to_deterministic for dist in self._flat_child_distributions]
        deterministic_dists: Any | dict | list = tree.unflatten_as(self._original_struct, flat_deterministic_dists)
        return MultiDistribution(deterministic_dists)

    def log_prob(self, value):
        return self.logp(value)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        logger.warning("Event Shape Accessed")
        return ()

    def _sample_n(self, key: jax.Array, n: int) -> Any:
        raise NotImplementedError("use sample method instead")
