from fast_rep.math_mod.loop_mrt import compute_delta_mrt
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from functools import partial
import numpy as np


@jax.jit
def compute_mrt_exp(pos_to_compute,Lambda, extra_t, xis, v):
    """
    Vectorized MRT calculation using JAX
    
    Parameters:
    pos_to_compute (jnp.ndarray): Positions to compute MRT at (1D) [n_positions]
    Lambda (jnp.ndarray): Array of λ values (1D) [n_origins]
    extra_t (jnp.ndarray): Array of extra times (1D) [n_origins]
    xis (jnp.ndarray): Origin positions (1D) [n_origins]
    v (float): Velocity
    
    Returns:
    jnp.ndarray: MRT values at requested positions [n_positions]
    """
    # Compute arrival times for all positions and origins
    #print(xis[None, :].shape,pos_to_compute[:, None].shape)


    delta_pos = xis[None, :] - pos_to_compute[:, None]
    arrival_time = jnp.abs(delta_pos) / v + extra_t[None, :]
    
    # Sort arrival times and get indices
    sorted_indices = jnp.argsort(arrival_time, axis=1)
    
    # Gather sorted values
    sorted_t = jnp.take_along_axis(arrival_time, sorted_indices, axis=1)
    sorted_lambda = jnp.take_along_axis(Lambda[None, :], sorted_indices, axis=1)
    
    # Create shifted version of sorted times for t_{i+1}
    
    
    # Compute cumulative sums
    Slambda = jnp.cumsum(sorted_lambda, axis=1)
    Cum_ti = jnp.cumsum(sorted_lambda * sorted_t, axis=1)
    
    # Calculate exponents
    sum_i = Slambda[:,:-1] * sorted_t[:,:-1] - Cum_ti[:,:-1]
    #sum_i = Slambda[:,:-1] * sorted_t[:,:-1] - Cum_ti[:,:-1]

    
    delta = Slambda[:,:-1] * (sorted_t[:,1:] -  sorted_t[:,:-1] )
    # Compute delta terms using log-space operations

    #log_term = -sum_i + jnp.log(-jnp.expm1(-delta)) - jnp.log(Slambda[:,:-1])
    safe_term = compute_delta_mrt(sum_i,delta,Slambda[:,:-1])[0]

    # Handle special cases where delta is large

    #safe_term = jnp.exp(log_term)
       
    # Sum terms and add first arrival time
    mrt = sorted_t[:, 0] + jnp.nansum(safe_term, axis=1)
    
    return mrt



prev_sorted_indices = None
prev_arrival_time = None




class MRTState:
    def __init__(self,xis,pos_to_compute,v ,initial_extra_t,N):
        self.prev_extra_t = self.batch_it(initial_extra_t,N)
        self.delta_pos_over_v = self.batch_it(jnp.abs(xis[None, :] - pos_to_compute[:, None]) / v ,N)
        self.prev_sorted =   self.batch_it(np.argsort(self.delta_pos_over_v[0] + initial_extra_t[None, :],axis=1),N)

    def batch_it(self,arr,N):
        r = []
        for i in range(N):
            r.append(arr)
        return jnp.array(r)



@jax.jit
def compute_mrt_weibull_cached(Lambda, extra_t,delta_pos_over_v, prev_extra_t, prev_sorted):

    time_diff = jnp.max(jnp.abs(extra_t - prev_extra_t))
    
    # Sort arrival times and get indices
    arrival_time = delta_pos_over_v+ extra_t[None, :]
    #print(delta_pos_over_v, extra_t[None, :])
    sorted_indices = jax.lax.cond(
        time_diff < 1e-1,
        lambda _: prev_sorted,  # Reuse previous
        lambda x: jnp.argsort(x, axis=1),  # Compute new
        operand=arrival_time
    )
    #jax.debug.print("{time_diff}",time_diff=time_diff)
    #sorted_indices = jnp.argsort(arrival_time, axis=1)
    #sorted_indices = jnp.argsort(arrival_time, axis=1)   
    #print(prev_sorted.shape,jnp.argsort(arrival_time, axis=1).shape)
    #print(sorted_indices.shape,arrival_time.shape,Lambda.shape)
    
    return compute_mrt_fast_weibull(sorted_indices,arrival_time,Lambda),extra_t,sorted_indices





@jax.jit
def compute_mrt_weibull(pos_to_compute,Lambda, extra_t, xis, v):
    
    delta_pos = xis[None, :] - pos_to_compute[:, None]
    arrival_time = jnp.abs(delta_pos) / v + extra_t[None, :]
    
    # Sort arrival times and get indices
    sorted_indices = np.argsort(arrival_time, axis=1)
    #sorted_indices = jnp.arange(len(extra_t))[None,:] # jnp.argsort(arrival_time, axis=1)

    return compute_mrt_fast_weibull(sorted_indices,arrival_time,Lambda)

@jax.jit
def compute_mrt_fast_weibull(sorted_indices,arrival_time,Lambda):
    """
    Vectorized MRT calculation using JAX
    
    Parameters:
    pos_to_compute (jnp.ndarray): Positions to compute MRT at (1D) [n_positions]
    Lambda (jnp.ndarray): Array of λ values (1D) [n_origins]
    extra_t (jnp.ndarray): Array of extra times (1D) [n_origins]
    xis (jnp.ndarray): Origin positions (1D) [n_origins]
    v (float): Velocity
    
    Returns:
    jnp.ndarray: MRT values at requested positions [n_positions]
    """
    # Compute arrival times for all positions and origins
    #print(xis[None, :].shape,pos_to_compute[:, None].shape)


    #print(sorted_indices.shape)
    #sorted_indices = jnp.arange(len(extra_t))[None,:] # jnp.argsort(arrival_time, axis=1)

    
    # Gather sorted values
    sorted_t = jnp.take_along_axis(arrival_time, sorted_indices, axis=1)
    sorted_lambda_2 = jnp.take_along_axis(Lambda[None, :]**2, sorted_indices, axis=1)
    
    #sorted_lambda_2 += 1e-7
    # Compute cumulative sums
    One = jnp.cumsum(sorted_lambda_2, axis=1)
    T = jnp.cumsum(sorted_lambda_2 * sorted_t, axis=1)
    T2 = jnp.cumsum(sorted_lambda_2 * sorted_t ** 2, axis=1)

    Tm = T/(One)
    rone = One**0.5
    #print(One)
    #Here either add to sorted_t a new line with heigh value so that erf==1
    complete = 1/rone[:,:-1] * jnp.exp(T[:,:-1]**2/One[:,:-1]-T2[:,:-1]) * jnp.pi **0.5/2 * (erf(rone[:,:-1] * (sorted_t[:,1:]   - Tm[:,:-1])) - \
                                                                                         erf(rone[:,:-1] * (sorted_t[:,:-1]  - Tm[:,:-1])))
    
    #or compute the last temr by hand
    last_term =  1/rone[:,-1] * jnp.exp(T[:,-1]**2/One[:,-1]-T2[:,-1]) * jnp.pi **0.5/2 * (1 - erf(rone[:,-1] * (sorted_t[:,-1]   - Tm[:,-1])))
    
    
    mrt = sorted_t[:, 0] + jnp.nansum(complete, axis=1) + last_term
    
    return mrt



@jax.jit
def compute_mrt_single(pos_to_compute,Lambda, extra_t, xis, v):
    """
    Vectorized MRT calculation using JAX
    
    Parameters:
    Lambda (jnp.ndarray): Array of λ values (1D) [n_origins]
    extra_t (jnp.ndarray): Array of extra times (1D) [n_origins]
    xis (jnp.ndarray): Origin positions (1D) [n_origins]
    pos_to_compute (jnp.ndarray): Positions to compute MRT at (1D) [n_positions]
    v (float): Velocity
    
    Returns:
    jnp.ndarray: MRT values at requested positions [n_positions]
    """
    # Compute arrival times for all positions and origins
    delta_pos = xis - pos_to_compute
    arrival_time = jnp.abs(delta_pos) / v + extra_t
    
    # Sort arrival times and get indices
    sorted_indices = jnp.argsort(arrival_time)
    
    # Gather sorted values
    sorted_t = arrival_time[sorted_indices]
    sorted_lambda = Lambda[sorted_indices]
    
    # Create shifted version of sorted times for t_{i+1}
    
    
    # Compute cumulative sums
    Slambda = jnp.cumsum(sorted_lambda)
    Cum_ti = jnp.cumsum(sorted_lambda * sorted_t)
    
    # Calculate exponents
    sum_i = Slambda[:-1] * sorted_t[:-1] - Cum_ti[:-1]
    
    delta = Slambda[:-1] * (sorted_t[1:] -  sorted_t[:-1] )
    # Compute delta terms using log-space operations

    #log_term = -sum_i + jnp.log(-jnp.expm1(-delta)+1e-12) - jnp.log(Slambda[:-1]+1e-12)
    safe_term = compute_delta_mrt(sum_i,delta,Slambda[:-1])[0]
    # Handle special cases where delta is large
       
    # Sum terms and add first arrival time
    mrt = sorted_t[ 0] + jnp.nansum(safe_term)
    
    return mrt

import numpy as np

def generate_regions(pos, ps, l):
    """
    Precompute fixed-size regions for all origins.
    
    Args:
        pos: Sorted array of positions (1D numpy array)
        ps: Origin positions (1D numpy array)
        l: Half-width of region (region size = 2l + 1)
        
    Returns:
        region_indices: (n_origins, 2l+1) array of indices
    """
    pos = np.asarray(pos)
    ps = np.asarray(ps)
    region_size = 2 * l + 1
    n_pos = len(pos)
    n_origins = len(ps)
    
    region_indices = np.zeros((n_origins, region_size), dtype=int)
    
    if region_size>= n_pos:
        return np.array([np.arange(n_pos) for o in ps])
    
    for i, origin in enumerate(ps):
        # Find closest position index
        center = np.argmin(np.abs(pos - origin))
        
        # Calculate region bounds
        start = max(0, center - l)
        if start + region_size > n_pos:
            start = max(0, n_pos - region_size)
        
        region_indices[i] = np.arange(start, start + region_size)
    
    return region_indices

@partial(jax.jit,static_argnames=["model"])
def compute_perturbed_mrt(pos,kis, extra_t, qis, xis, v, region_indices, large_t=10000,model="Exponential"):
    """
    Compute perturbed RFD using precomputed fixed-size regions in a JAX-compatible and vectorized manner.
    
    Args:
        region_indices: Precomputed (n_origins, region_size) indices array.
    """
    print(region_indices)

    n_origins = region_indices.shape[0]
    region_size = region_indices.shape[1]
    q_prod = jnp.prod(qis)

    if model == "Exponential":
        mrt_func = compute_mrt_exp
    else:
        mrt_func =  compute_mrt_weibull
    
    original_mrt =  mrt_func(pos,kis, extra_t, xis, v)

    # Compute multiplicative updates for each region
    mask = jnp.zeros((n_origins,pos.shape[0]), dtype=bool)
    mask = mask.at[jnp.arange(n_origins)[:, None], region_indices].set(True)
    product_per_position = jnp.prod(jnp.where(mask, qis[:,None], 1), axis=0)
    result = original_mrt * product_per_position
    
    # Prepare perturbed extra_t values for each origin
    perturbed_extra_t_all = jax.nn.one_hot(jnp.arange(n_origins), n_origins, dtype=extra_t.dtype) * large_t
    
    # Extract position subsets for each origin's region
    pos_subsets = pos[region_indices]  # Shape (n_origins, region_size)
    
    # Vectorized computation of perturbed RFD for all origins
    pert_all = jax.vmap(mrt_func, in_axes=(0,None, 0, None, None))(
        pos_subsets,kis, extra_t + perturbed_extra_t_all, xis, v
    )
    
    # Calculate additive terms and apply updates
    #print("A")

    q_terms = q_prod * (1 - qis) / (qis)# + 1e-8)
    #jax.debug.print("{q_prod} {q_terms}",q_terms=q_terms,q_prod=q_prod)
    updates = q_terms[:, None] * pert_all  # Shape (n_origins, region_size)
    result = result.at[region_indices].add(updates) 
    result /= q_prod + jnp.sum(q_terms)   # normalise to 1 for all the contributions
    
    return result