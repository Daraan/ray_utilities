from ast import literal_eval
from typing import Optional

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from tap import Tap
import tree

from ray_utilities.config import DefaultArgumentParser

try:
    from frozenlist import FrozenList  # pyright: ignore[reportAssignmentType]
except ImportError:
    from copy import deepcopy

    class FrozenList(list):
        def freeze(self):
            pass

        def append(self, *args, **kwargs) -> None:  # noqa: ARG002
            raise RuntimeError("Cannot modify frozen list.")

        insert = pop = append
        frozen: bool = True

        def __deepcopy__(self, memo):
            # Return a new FrozenList with the same elements
            copied_list = FrozenList((deepcopy(item, memo) for item in self))
            if self.frozen:
                copied_list.freeze()
            return copied_list

    _fcnet_hiddens_default = FrozenList([256, 256])
    _head_fcnet_hiddens_default = FrozenList([])
else:
    _fcnet_hiddens_default = FrozenList([256, 256])
    _fcnet_hiddens_default.freeze()
    _head_fcnet_hiddens_default = FrozenList([])
    _head_fcnet_hiddens_default.freeze()

__all__ = ["MLPArgumentParser"]


class SimpleMLPParser(Tap):
    """Keys must align with RLlib's DefaultModelConfig"""

    # Documentation taken from ray's DefaultModelConfig

    fcnet_hiddens: list[int] = _fcnet_hiddens_default
    """
    List containing the sizes (number of nodes) of a fully connected (MLP) stack.
    Note that in an encoder-based default architecture with a policy head (and
    possible value head), this setting only affects the encoder component. To set the
    policy (and value) head sizes, use `post_fcnet_hiddens`, instead. For example,
    if you set `fcnet_hiddens=[32, 32]` and `post_fcnet_hiddens=[64]`, you would get
    an RLModule with a [32, 32] encoder, a [64, act-dim] policy head, and a [64, 1]
    value head (if applicable).

    Passing a list as a string: --fcnet_hiddens="[256, 256]" is equivalent to --fcnet_hiddens 256 256
    """

    fcnet_activation: str = DefaultModelConfig.fcnet_activation  # tanh
    """
    Activation function descriptor for the stack configured by `fcnet_hiddens`.
    Supported values are: 'tanh', 'relu', 'swish' (or 'silu', which is the same),
    and 'linear' (or None).
    """

    fcnet_kernel_initializer: Optional[str] = None
    """
    Initializer function descriptor for the weight/kernel matrices in the stack
    configured by `fcnet_hiddens`. Supported values are the names of initializers
    as strings. See https://pytorch.org/docs/stable/nn.init.html for `torch`. If
    `None` (default), the default initializer defined by `torch` is used.
    """

    fcnet_bias_initializer: Optional[str] = None
    """
    Initializer function descriptor for the bias vectors in the stack
    configured by `fcnet_hiddens`. Supported values are the names of initializers
    as strings. See https://pytorch.org/docs/stable/nn.init.html for `torch`. If
    `None` (default), the default initializer defined by `torch` is used.
    """

    # ====================================================
    # Head configs (e.g. policy- or value function heads)
    # ====================================================
    head_fcnet_hiddens: list[int] = _head_fcnet_hiddens_default
    """
    List containing the sizes (number of nodes) of a fully connected (MLP) head
    (e.g., policy-, value-, or Q-head). To configure the encoder architecture,
    use `fcnet_hiddens` instead.

    Passing a list string: --head_fcnet_hiddens="[256, 256]" is equivalent to --head_fcnet_hiddens 256 256

    Default:
        Empty list; no layers
    """

    head_fcnet_activation: str = DefaultModelConfig.head_fcnet_activation  # relu
    """
    Activation function descriptor for the stack configured by `head_fcnet_hiddens`.
    Supported values are: 'tanh', 'relu', 'swish' (or 'silu', which is the same),
    and 'linear' (or None).
    """

    head_fcnet_kernel_initializer: Optional[str] = None
    """
    Initializer function descriptor for the weight/kernel matrices in the stack
    configured by `head_fcnet_hiddens`. Supported values are the names of initializers
    as strings. See https://pytorch.org/docs/stable/nn.init.html for `torch`. If `None`
    (default), the default initializer defined by `torch` is used.
    """

    head_fcnet_bias_initializer: Optional[str] = None
    """
    Initializer function descriptor for the bias vectors in the stack configured
    by `head_fcnet_hiddens`. Supported values are the names of initializers as strings.
    See https://pytorch.org/docs/stable/nn.init.html for `torch`. If `None` (default),
    the default initializer defined by `torch` is used.
    """

    vf_share_layers: bool = DefaultModelConfig.vf_share_layers  # True
    """
    Whether encoder layers (defined by `fcnet_hiddens` or `conv_filters`) should be
    shared between policy- and value function.
    """

    def configure(self) -> None:
        super().configure()
        self.add_argument("--fcnet_hiddens", nargs="*", type=literal_eval)
        self.add_argument("--head_fcnet_hiddens", nargs="*", type=literal_eval)

    def process_args(self) -> None:
        # flatten list of lists
        self.fcnet_hiddens = tree.flatten(self.fcnet_hiddens)
        self.head_fcnet_hiddens = tree.flatten(self.head_fcnet_hiddens)
        super().process_args()


class MLPArgumentParser(SimpleMLPParser, DefaultArgumentParser):
    pass
