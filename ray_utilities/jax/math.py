import jax
import jax.numpy as jnp
from ray.rllib.utils.numpy import SMALL_NUMBER


def explained_variance(y: jax.Array, pred: jax.Array) -> jax.Array:
    """
    Code taken from from ray.rllib.utils.torch_utils import explained_variance

    Computes the explained variance for a pair of labels and predictions.

    The formula used is:
    max(-1.0, 1.0 - (std(y - pred)^2 / std(y)^2))

    Args:
        y: The labels.
        pred: The predictions.

    Returns:
        The explained variance given a pair of labels and predictions.
    """
    y_var = jnp.var(y, axis=[0])
    diff_var = jnp.var(y - pred, axis=[0])
    compare = jnp.array([-1.0, 1 - (diff_var / (y_var + SMALL_NUMBER))])
    # min_ = jax.device_put(min_, pred.device)
    # NOTE: For torch this is max(input, other); for jnp/numpy this is max(input, axis)
    return jnp.max(compare)


@jax.custom_vjp
def clip_gradient(lo, hi, x):  # noqa: ARG001
    """
    Experimental implementation of a gradient clipping function.

    Implementation taken from:
        https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#gradient-clipping
    """
    return x  # identity function


def clip_gradient_fwd(lo, hi, x):
    return x, (lo, hi)  # save bounds as residuals


def clip_gradient_bwd(res, g):
    lo, hi = res
    return (None, None, jnp.clip(g, lo, hi))  # use None to indicate zero cotangents for lo and hi


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
