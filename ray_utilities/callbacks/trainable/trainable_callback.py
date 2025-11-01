from typing import TYPE_CHECKING

from ray.rllib.callbacks.callbacks import RLlibCallback

if TYPE_CHECKING:
    from ray_utilities.training.default_class import TrainableBase


class TrainableCallbackExtension(RLlibCallback):
    """Mixin for RLlibCallbacks that gets access to the :class:`DefaultBase` instance."""

    def on_trainable_setup(
        self,
        *,
        trainable: "TrainableBase",
        **kwargs,
    ) -> None:
        """Called when the trainable is setup, this allows modification of other :class:`RLlibCallback` methods
        that need access to the advanced trainable instance parameters.

        Args:
            trainable: The trainable instance being set up.
            **kwargs: Additional keyword arguments.
        """
