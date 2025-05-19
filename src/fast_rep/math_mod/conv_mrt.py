from functools import partial
import jax
from jax import lax, jit
import jax.numpy as jnp


@jit
def grouped_convolution(lambdai_stacked, mask):
    # lambdai_stacked: shape (B, L)
    # mask: shape (B, W)
    B, L = lambdai_stacked.shape
    W = mask.shape[1]

    # Reshape input: (B, L) -> (1, L, B)
    input_grouped = jnp.transpose(lambdai_stacked, (1, 0))[None, :, :]

    # Reshape kernel: (B, W) -> (W, 1, B)
    kernel_grouped = jnp.transpose(mask, (1, 0))[:, None, :]

    # Apply grouped convolution
    conv_out = lax.conv_general_dilated(
        input_grouped,
        kernel_grouped,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=B
    )
    # conv_out shape is (1, L - W + 1, B); transpose back to (B, L - W + 1)
    result = jnp.transpose(conv_out[0], (1, 0))
    return result

@partial(jax.jit,static_argnames=["max_n"])
def compute_updates_conv(lambdai, max_n, v):#:, global_offset=0, original_length=None):
    """
    Computes MRT updates using convolution with triangular masks.
    """
    #original_length = original_length if original_length is not None else lambdai.shape[0]
    #max_n = int(jnp.max(n_values))
    max_n_plus = max_n + 1  # Maximum n in mask
    W = 2 * max_n + 1
    new_center = W // 2  # Center of mask

    # Create triangular mask
    positions = jnp.arange(W)[None, :]
    n_values_b = jnp.arange(max_n + 2)[:, None]  # n from 0 to max_n+1 inclusive
    lower = new_center - (n_values_b - 1)
    upper = new_center + (n_values_b - 1)
    condition = (positions >= lower) & (positions <= upper)
    mask = jnp.where(condition, 1, 0).astype(jnp.float32)

    # Pad lambdai with max_n zeros on both sides
    padding = int(max_n)
    padded_lambdai = jnp.pad(lambdai, (padding, padding), mode='constant')

    # Stack padded_lambdai for parallel convolution
    lambdai_stacked = jnp.tile(padded_lambdai[None, :], (max_n_plus + 1, 1))

    # Apply vmap for parallel convolution
    #Sum = vmap(convolve_single, in_axes=(0, 0))(lambdai_stacked, mask)
    Sum = grouped_convolution(lambdai_stacked, mask)
    #print(Sum.shape)

    # Compute cumulative sum divided by v along the spatial dimension (axis=1)
    CumSum = jnp.cumsum(Sum / v, axis=0)  # Ensure axis=1 (spatial dimension)
    ExpCumsum = jnp.exp(-CumSum)

    # Compute differences and terms
    delta = ExpCumsum[:-1,:] - ExpCumsum[1:,:]
    denominator = Sum[1:,:] + 1e-7
    term = delta / denominator

    # Sum over all n to get final MRT (sum over axis=0)
    MRT = jnp.sum(term, axis=0)
    final_mrt = MRT[:]#original_length]

    return final_mrt # Return only the result needed for gradients