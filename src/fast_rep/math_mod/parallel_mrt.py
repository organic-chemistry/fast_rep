import optax
from functools import partial 


#Not working....

@partial(jax.jit, static_argnums=(1, 2))
def split_into_overlapping_chunks(arr, chunk_size, max_n):
    L = arr.shape[0]
    step = chunk_size - max_n  # Step between chunks
    num_chunks = int((L + step - 1) // step ) # Ceiling division
    chunks = []
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        if end > L:
            start = L - chunk_size
            end = L
        chunk = arr[start:end]
        chunks.append(chunk)
    return jnp.stack(chunks)  # Shape: (num_chunks, chunk_size)


@partial(jax.jit, static_argnums=(3, 4))
def parallel_mrt(lambdai, n_values, v, chunk_size=25000, tolerance=1e-4):
    max_n = jnp.max(n_values)
    chunks = split_into_overlapping_chunks(lambdai, chunk_size, max_n)
    
    # Vectorize compute_updates over chunks
    chunk_mrt = jax.vmap(
        compute_updates,
        in_axes=(0, None, None, None)  # lambdai_batch is batched, others are fixed
    )(chunks, n_values, v, tolerance)
    
    # Stitch chunks back into a single array
    L = lambdai.shape[0]
    step = chunk_size - max_n
    result = jnp.zeros(L)
    for i in range(chunks.shape[0]):
        start = i * step
        end = start + chunk_size
        chunk_start = max(0, start)
        chunk_end = min(end, L)
        # Take central part (excluding overlaps)
        mid_start = max_n // 2
        mid_end = mid_start + (chunk_end - chunk_start)
        result = result.at[chunk_start:chunk_end].set(
            chunk_mrt[i, mid_start:mid_end]
        )
    return result



    
from jax import eval_shape

from functools import partial

# Modified to accept static N

@partial(jax.jit, static_argnames=('chunk_size',))
def parallel_compute_updates(lambdai, n_values, v, chunk_size=1024*5, tolerance=1e-4):
    """Parallel version with automatic N calculation"""
    # Compute N from n_values (assumes n_values is a static array)
    N = n_values.shape[0] - 1  # For n_values = [0, 1, 2, ..., N]
    orig_len = lambdai.shape[0]

    # Pad with zeros
    padded = jnp.pad(lambdai, (N, N), mode='constant')
    
    # Create chunks with overlap
    def get_chunk(i):
        start = i * chunk_size
        return jax.lax.dynamic_slice(padded, (start,), (chunk_size + 2*N,))
    
    chunks = jax.vmap(get_chunk)(jnp.arange(0, orig_len, chunk_size))
    
    # Calculate global offsets
    global_offsets = jnp.arange(0, orig_len, chunk_size) - N
    
    # Batched computation with corrected argument order
    results = jax.vmap(compute_updates, in_axes=(0, None, None, None, 0, None))(
        chunks,  # lambdai_chunk (batched)
        n_values,  # n_values (shared)
        v,        # v (shared)
        tolerance, # tolerance (shared)
        global_offsets,  # global_offset (batched)
        orig_len   # original_length (shared)
    )
    
    # Combine results
    return jnp.concatenate(results[:, N:-N])[:orig_len]




gp_hyperparams = {
    "sigma": .1,     # Amplitude of variations
    "lengthscale": 10.0,  # Smoothness (for Approach 1)
    # "rho": 0.95       # Correlation (for Approach 2)
}



