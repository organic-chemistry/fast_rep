import jax
import jax.numpy as jnp
from jax import lax, vmap
from functools import partial


def original_compute_delta_mrt(cum_n, delta_cum, new_sum_n):
    exp_cum = jnp.exp(-cum_n)
    exp_term = -jnp.expm1(-delta_cum)  # 1 - exp(-delta_cum) via expm1 for stability
    numerator = exp_cum * exp_term

    # Update cumulative terms
    new_cum_n = cum_n + delta_cum  # Avoids subtraction of large numbers
    term = new_sum_n + 1e-8  # Increased epsilon for better stability
    delta_mrt = numerator / term
    return delta_mrt, new_cum_n




import jax
import jax.numpy as jnp

@jax.jit
def compute_delta_mrt(cum_n, delta_cum, new_sum_n):
    safe = lambda x: jnp.where(x > 0, x, 1e-10)  # Avoid log(0) or negative inputs

    log_exp_term = jnp.log(-jnp.expm1(-safe(delta_cum)))
    log_safe_denom = jnp.log(safe(new_sum_n))
    log_result = -cum_n + log_exp_term - log_safe_denom
    delta_mrt = jnp.exp(log_result)
    new_cum_n = cum_n + delta_cum
    return delta_mrt, new_cum_n


def compute_cum_sum_old(lambdai,n_values,original_length=None):
    original_length = original_length if original_length is not None else lambdai.shape[0]
    arr_global = jnp.arange(lambdai.shape[0])
    def roll_and_mask(n):
        mask_left = arr_global < (original_length - n)
        mask_right = arr_global >= n
        rolled_left = jnp.roll(lambdai, -n) * mask_left
        rolled_right = jnp.roll(lambdai, n) * mask_right
        return rolled_left + rolled_right

    # Batch compute all rolled terms for n > 0
    rolled_terms = vmap(roll_and_mask)(n_values[1:])
    rolled_terms = jnp.concatenate([lambdai[None, ...], rolled_terms])

    # Compute cumulative sums along n axis
    sum_n = jnp.cumsum(rolled_terms, axis=0)
    return sum_n



def compute_all_window_sums(v, n_values):
    # Pad the array to handle boundaries
    #padded_v = jnp.pad(v, (max_k, max_k), mode='constant', constant_values=0)
    
    # Compute prefix sum
    prefix = jax.lax.associative_scan(jnp.add, v,axis=0)
    prefix = jnp.concatenate([jnp.array([0]), prefix])  # Add prefix[0] = 0

    
    # Prepare indices for all k and i
    N = len(v)
    indices = jnp.arange(N) #+ max_k  # Offset due to padding
    
    # Vectorize over k and i to compute all window sums
    def get_window_sum(k):
        left = jnp.clip(indices - k, 0, N)  # Handle i - k < 0
        right = jnp.clip(indices + k + 1, 0, N)  # Handle i + k >= N
        return prefix[right] - prefix[left]
    
    # Compute for all k in [0, max_k]
    #k_values = jnp.arange(max_k + 1)
    sum_n = jax.vmap(get_window_sum)(n_values)
    
    return sum_n  # Remove padding artifacts

@partial(jax.jit,static_argnames=["sum_by_map"])
def compute_updates(lambdai, n_values, v, tolerance=1e-4, original_length=None,sum_by_map=True):


    # Precompute all rolled terms
    if sum_by_map:
        sum_n = compute_cum_sum_old(lambdai,n_values,original_length=original_length)
    else:
        #by lax associative
        sum_n = compute_all_window_sums(lambdai,n_values)
        sum_n = jnp.maximum(sum_n, 1e-12)
    #is_equal = jnp.allclose(sum_n_old, sum_n, atol=1e-4, rtol=1e-4)
    #jax.debug.print("Are the results equal? {is_equal}", is_equal=is_equal)

    delta_cum_all = sum_n / v

    # Compute cumulative cum_n
    cum_n_all = jnp.cumsum(delta_cum_all, axis=0)

    # Compute delta_mrt for all steps
    def compute_delta_mrt_step(cum_n_prev, delta_cum_step, sum_n_step):
        return compute_delta_mrt(cum_n_prev, delta_cum_step, sum_n_step)

    delta_mrt_all, _ = vmap(compute_delta_mrt_step)(
        jnp.roll(cum_n_all, 1, axis=0).at[0].set(0),
        delta_cum_all,
        sum_n
    )

    final_mrt = jnp.sum(delta_mrt_all, axis=0)
    """
    delta_mrt_diff = jnp.abs(jnp.diff(delta_mrt_all, axis=0))
    stop_flags = jnp.max(delta_mrt_diff, axis=1) < tolerance
    first_stop = jnp.argmax(stop_flags) + 1
    """

    return final_mrt



@jax.jit
def compute_updates_old(lambdai, n_values, v, tolerance=1e-4, 
                    global_offset=0, original_length=None):

    original_length = original_length if original_length is not None else lambdai.shape[0]
    arr_global = global_offset + jnp.arange(lambdai.shape[0])

    def scan_step(state, n):
        sum_n, cum_n, mrt, prev_mrt, stop_flag, n_stop = state

        def active_branch():
            # Compute new_sum_n
            new_sum_n = jax.lax.cond(
                n == 0,
                lambda: lambdai,
                lambda: sum_n + (
                    jnp.roll(lambdai, -n) * (arr_global < (original_length - n)) +
                    jnp.roll(lambdai, n) * (arr_global >= n)
                )
            )

            # Compute delta_cum and terms in a numerically stable way
            delta_cum = new_sum_n / v

            delta_mrt,new_cum_n = compute_delta_mrt(cum_n, delta_cum, new_sum_n)

            new_mrt = mrt + delta_mrt

            # Update stop condition using relative + absolute tolerance
            #rel_diff = jnp.abs(new_mrt - mrt) / (jnp.abs(mrt) + 1e-8)
            #abs_diff = jnp.abs(new_mrt - mrt)
            #new_stop_flag = jnp.logical_and(
            #    jnp.max(abs_diff) < tolerance,
            #    jnp.max(rel_diff) < 1e-3
            #)
            new_stop_flag = jnp.max(jnp.abs(new_mrt - mrt)) < tolerance


            return (new_sum_n, new_cum_n, new_mrt, mrt, new_stop_flag,n)

        return jax.lax.cond(
            stop_flag,
            lambda: (sum_n, cum_n, mrt, prev_mrt, stop_flag,n_stop),
            active_branch
        ), None

    init_state = (
        lambdai,
        jnp.zeros_like(lambdai),
        jnp.zeros_like(lambdai),
        jnp.full_like(lambdai, jnp.inf),
        jnp.array(False),
        0
    )

    (final_sum_n, final_cum_n, final_mrt, _,_ ,n_stop), _ = jax.lax.scan(
        scan_step, init_state, n_values
    )
    #jax.debug.print("n round {n_stop} for tolerance {tolerance}",n_stop=n_stop,tolerance=tolerance)

    return final_mrt




