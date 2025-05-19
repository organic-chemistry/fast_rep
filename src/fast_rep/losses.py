import jax
import jax.numpy as jnp
#import jax.scipy.linalg
from fast_rep.interface import compute_mrt
from fast_rep.math_mod.compute_rfd import compute_derivatives
from functools import partial
from dataclasses import dataclass, field
from typing import Union, Tuple


# Define parameter classes
@dataclass(frozen=True)
class ARParams:
    sigma: float = 10.0
    rho: float = 0.7

@dataclass(frozen=True)
class GPParams:
    mean: float = 0.0
    sigma: float = 1.0
    lengthscale: float = 10.0

@dataclass(frozen=True)
class PriorConfig:
    type: str
    params: Union[ARParams, GPParams]


@partial(jax.jit,static_argnames=["sigma","rho"])
def ar1_prior_loss(lambdai, sigma=10.0, rho=0.7):
    """
    AR(1) prior negative log-likelihood (approximates GP with exponential kernel).
    Args:
        rho: Correlation coefficient (rho = exp(-1/lengthscale))
    """
    n = len(lambdai)
    term1 = 0.5 * (lambdai[0] ** 2) / (sigma ** 2)
    residuals = lambdai[1:] - rho * lambdai[:-1]
    term2 = 0.5 * jnp.sum(residuals ** 2) / (sigma ** 2 * (1 - rho ** 2))
    logdet = 0.5 * (n - 1) * jnp.log(1 - rho ** 2)
    #jax.debug.print("{sigma}",sigma=sigma)

    return (term1 + term2 + logdet)/n


@partial(jax.jit,static_argnames=["mean","sigma","lengthscale"])
def gp_prior_loss(lambdai, mean=0.0, sigma=1.0, lengthscale=10.0):
    """
    GP prior negative log-likelihood using a squared exponential kernel.
    """
    n = len(lambdai)
    X = jnp.arange(n)[:, None]  # Shape (n, 1)

    # Squared exponential kernel
    def kernel(x1, x2):
        return sigma**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / lengthscale**2)

    # Build covariance matrix using nested vmap
    K = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(X))(X)
    K += 1e-6 * jnp.eye(n)  # Add jitter for numerical stability

    # Cholesky decomposition and solve
    L = jax.scipy.linalg.cholesky(K, lower=True)
    diff = lambdai - mean
    alpha = jax.scipy.linalg.cho_solve((L, True), diff)
    #jnp.debug.print("{sigma}",sigma=sigma)
    
    return (0.5 * diff.T @ alpha + jnp.sum(jnp.log(jnp.diag(L))))/n

@partial(jax.jit, static_argnames=["method", "measurement_type", "shift",
                                    "weights", "prior_config", "max_n", 
                                    "fit_resolution", "tolerance","sum_by_map"])
def loss_function(lambdai, max_n, v, data, error_data, fit_resolution=1, reg_loss=1000, 
                  method='forward', measurement_type="rfd", shift=5,experiment_resolution=1., #in kb
                  weights=(1.0, 1.0),prior_config: PriorConfig = PriorConfig("AR", ARParams()),tolerance=1e-4,sum_by_map=True):
    """
    Loss function that handles static arguments correctly.    """
    # Common calculation for all measurement types - do it only once
    mrt = compute_mrt(lambdai, max_n, v,tolerance=tolerance,sum_by_map=sum_by_map)
    
    # Handle different measurement types
    if measurement_type == "rfd":
        if method in ['forward', 'central']:
            # Use the jitted version with static arguments
            computed = compute_derivatives(mrt, v, method)
        else:
            raise ValueError(f"Invalid method {method} for RFD measurement")
        
    elif measurement_type == "mrt":
        computed = mrt
        
    elif measurement_type == "delta_mrt":
        # Use the jitted version with static arguments
        computed = compute_derivatives(mrt, v, 'delta', shift,resolution=experiment_resolution)
        
    elif measurement_type == "both":
        # Compute both quantities from shared MRT
        rfd = compute_derivatives(mrt, v, method,resolution=experiment_resolution)
        
        # Slice and compute losses
        rfd_comp = rfd[::fit_resolution]
        mrt_comp = mrt[::fit_resolution]
        
        # Assume data is tuple (rfd_data, mrt_data)
        mse_rfd = jnp.mean(jnp.square(rfd_comp - data[0][:len(rfd_comp)]))
        mse_mrt = jnp.mean(jnp.square(mrt_comp - data[1][:len(mrt_comp)]))
        mse = weights[0]*mse_rfd + weights[1]*mse_mrt
        
        # Add regularization and return early
        return mse + ar1_prior_loss(lambdai) * reg_loss
        
    else:
        raise ValueError(f"Unknown measurement_type: {measurement_type}")

    # Handle regular cases (not 'both')
    computed = computed[::fit_resolution]
    mse = jnp.mean(jnp.square( (computed - data[:len(computed)]) * error_data))
    
    if prior_config.type == "AR":
        params = prior_config.params
        prior_loss = ar1_prior_loss(lambdai, sigma=params.sigma, rho=params.rho)
    else:
        params = prior_config.params
        prior_loss = gp_prior_loss(
            lambdai,
            mean=params.mean,
            sigma=params.sigma,
            lengthscale=params.lengthscale
        )
    # Use faster squared operation instead of **2
    #return mse + ar1_prior_loss(lambdai,**prior) * reg_loss
    return mse + prior_loss * reg_loss