import jax
import jax.numpy as jnp
from jax import random,jit
from functools import partial
import numpy as np

def draw_exponential(key, times):
    # Generate exponential samples with the same shape as times
    # rate=1 is the default, but you can adjust this parameter
    return random.exponential(key, shape=times.shape)
# For Weibull distribution with k=2 (shape parameter)
def draw_weibull(key, times, k=2.0):
    # Weibull can be generated from uniform via inverse transform
    u = random.uniform(key, shape=times.shape)
    # For k=2, the inverse CDF is: (-log(1-u))^(1/2)
    return jnp.power(-jnp.log(1.0 - u), 1.0/k)
# exponential_samples and weibull_samples will be arrays of shape (num_realizations, len(times))
# For exponential with times as scale (1/rate)
def draw_exponential_scaled(key, times):
    # Draw unit exponentials and scale them
    unit_exp = random.exponential(key, shape=times.shape)
    return unit_exp * times  # Scale by times
# For Weibull with k=2 and times as scale
def draw_weibull_scaled(key, times, k=2.0):
    u = random.uniform(key, shape=times.shape)
    # Scale the standard Weibull by times
    return times * jnp.power(-jnp.log(1.0 - u), 1.0/k)

@jit
def compute_mrt_rfd_efficiency(T, v):
    n = len(T)
    indices = jnp.arange(n)
    inv_v = 1.0 / v  # Precompute reciprocal for efficiency
    
    # Left contribution calculations
    left_values = T - indices * inv_v
    
    # We'll track both min values and their indices
    def min_with_index(a, b):
        val_a, idx_a = a
        val_b, idx_b = b
        # Choose the minimum value and its corresponding index
        is_a_smaller = val_a <= val_b  # Using <= ensures we take first occurrence when equal
        min_val = jnp.where(is_a_smaller, val_a, val_b)
        min_idx = jnp.where(is_a_smaller, idx_a, idx_b)
        return min_val, min_idx
    
    # Initialize with values and their indices
    left_pairs = (left_values, indices)
    
    left_min_val, left_min_idx = jax.lax.associative_scan(min_with_index, left_pairs, axis=0)
    left_contribution = left_min_val + indices * inv_v
    #print(left_min_idx)
    
    # Right contribution calculations
    right_values = T + indices * inv_v
    # For the right scan, we need to reverse, scan, and then reverse back
    right_pairs_rev = (right_values[::-1], indices[::-1])
    right_min_val_rev, right_min_idx_rev = jax.lax.associative_scan(min_with_index, right_pairs_rev, axis=0)
    # Reverse back to original order
    right_min_val = right_min_val_rev[::-1]
    right_min_idx = right_min_idx_rev[::-1]  # Adjust indices for the reversal
    right_contribution = right_min_val - indices * inv_v
    #print(right_min_idx)

    # Calculate MRT (minimum of left and right contributions)
    is_left_smaller = left_contribution <= right_contribution
    mrt = jnp.minimum(left_contribution, right_contribution)
    
    # Calculate RFD (-1 for left, 1 for right, 0 for equal)
    rfd = jnp.where(is_left_smaller, 1, -1)
    rfd = jnp.where(left_contribution == right_contribution, 0, rfd)
    
    # Calculate observed efficiency
    # This is based on whether the source of the min (left_min_idx or right_min_idx)
    # matches the position when rfd is -1 or 1 respectively
    
    # For each position i:
    # - If rfd[i] = -1, efficiency is good if left_min_idx[i] = i
    # - If rfd[i] = 1, efficiency is good if right_min_idx[i] = i
    
    left_efficiency = (left_min_idx == indices) & ((rfd == 0)  |  (rfd == 1) )
    right_efficiency = (right_min_idx == indices)& ((rfd == 0)  |  (rfd == -1) ) #& ((rfd == -1) | (rfd == 0))
    # Positions where rfd = 0 count as efficient
    #equal_efficiency = (rfd == 0)
    
    # Overall efficiency for each position
    position_efficiency = left_efficiency | right_efficiency # | equal_efficiency
    #position_efficiency = left_min_idx
    # Compute average efficiency across the array
    #observed_efficiency = jnp.mean(position_efficiency.astype(jnp.float32))
    
    return mrt, rfd, position_efficiency

#@partial(jax.jit,static_argnames=["distribution","n_sim"])
def sim(n_sim, kis,tis, v, distribution="Exponential"):
    if np.any(kis) == 0:
        raise "kis should have non zero value (set a low background) "
    
    return sim_safe(n_sim, 1/kis,tis, v, distribution=distribution)


@partial(jax.jit,static_argnames=["distribution","n_sim"])
def sim_safe(n_sim, kis,tis, v, distribution="Exponential"):
    # Input validation
   
    master_key = random.PRNGKey(0)
    sim_keys = random.split(master_key, n_sim)
    

    # Draw the scaled realizations
    if distribution == "Exponential":
        time_scaled = jax.vmap(draw_exponential_scaled, in_axes=(0, None))(sim_keys, kis)
    else:
        time_scaled = jax.vmap(draw_weibull_scaled, in_axes=(0, None, None))(sim_keys, kis, 2.0)

    time_scaled += tis
    
    # Apply our enhanced function that also computes efficiency
    MRTs_RFDs_Efficiencies = jax.vmap(lambda x: compute_mrt_rfd_efficiency(x, v), in_axes=0)(time_scaled)
    MRTs, RFDs, observed_efficiencies = MRTs_RFDs_Efficiencies
    
    return {
        "hist_RFD": RFDs,
        "hist_MRT": MRTs,
        "activation_times":time_scaled,
        "MRT_time": jnp.mean(MRTs, axis=0),
        "RFD": jnp.mean(RFDs, axis=0),
        "observed_efficiencies": observed_efficiencies,
        "mean_observed_efficiency": jnp.mean(observed_efficiencies,axis=0)
    }
