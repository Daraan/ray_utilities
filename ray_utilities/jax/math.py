import jax
import jax.numpy as jnp
import optax
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


def clip_gradients(gradients_dict, grad_clip, grad_clip_by):
    """
    Clips gradients and returns the global norm.
    Matches the interface expected by RLlib's Learner._get_clip_function.
    """
    if grad_clip is None:
        return None

    if grad_clip_by == "global_norm":
        global_norm = optax.global_norm(gradients_dict)
        # Scale down if global_norm > grad_clip
        scale = jnp.where(global_norm > grad_clip, grad_clip / (global_norm + SMALL_NUMBER), 1.0)
        clipped = jax.tree_util.tree_map(lambda g: g * scale, gradients_dict)

        # Update dict in place to satisfy RLlib interface
        if isinstance(gradients_dict, dict):
            for k, v in clipped.items():
                gradients_dict[k] = v
        return global_norm

    elif grad_clip_by == "norm":

        def _clip(g):
            norm = jnp.linalg.norm(g)
            scale = jnp.where(norm > grad_clip, grad_clip / (norm + SMALL_NUMBER), 1.0)
            return g * scale

        clipped = jax.tree_util.tree_map(_clip, gradients_dict)
        if isinstance(gradients_dict, dict):
            for k, v in clipped.items():
                gradients_dict[k] = v
        return None

    elif grad_clip_by == "value":
        clipped = jax.tree_util.tree_map(lambda g: jnp.clip(g, -grad_clip, grad_clip), gradients_dict)
        if isinstance(gradients_dict, dict):
            for k, v in clipped.items():
                gradients_dict[k] = v
        return None

    else:
        raise ValueError(f"`grad_clip_by` ({grad_clip_by}) must be one of [value|norm|global_norm]!")


@jax.custom_vjp
def clip_gradient(lo, hi, x):  # noqa: ARG001
    """
    Experimental implementation of a gradient clipping function.

    Used during forward passes to clip gradients during backpropagation.

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
