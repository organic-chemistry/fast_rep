import jax
from fast_rep.interface import compute_mrt_and_derivatives
import jax.numpy as jnp



def generate_synthetic_data(lambdai, max_n, v, noise_level=0.05,method="forward",shift=5, key=None):
    """Generate synthetic RFD data based on true lambda values with added noise."""
    if key is None:
        key = jax.random.PRNGKey(0)  # Default seed if none provided
        

    mrt,true_rfd = compute_mrt_and_derivatives(lambdai, max_n, v,method=method,shift=shift)
 
    noise_scale = noise_level * jnp.max(jnp.abs(true_rfd))
    noise = noise_scale * jax.random.normal(key, true_rfd.shape)
    noisy_rfd = true_rfd + noise
    return {"mrt":mrt,"noisy_rfd":noisy_rfd,"true_rfd":true_rfd}
