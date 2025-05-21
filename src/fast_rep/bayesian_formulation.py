from fast_rep.math_mod.compute_rfd import compute_derivatives
from fast_rep.math_mod.compute_mrt_at_pos import compute_perturbed_mrt,generate_regions
from fast_rep.math_mod.compute_mrt_at_pos import compute_mrt_exp,compute_mrt_weibull,compute_mrt_weibull_cached
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm,beta,expon


@partial(jax.jit,static_argnames=["model","measurement_type","use_extra_t","use_qi"])
def compute_theo(params,pos_to_compute,xis,v,measurement_type,use_extra_t,use_qi,regions_indices,resolution,model="Exponential"):
    Lambda = params["kis"]
    Lambda = jnp.clip(Lambda,1e-7,100)

    if use_extra_t:
        extra_t = params["extra_t"]
        extra_t -= jnp.min(extra_t)
    else:
        extra_t = jnp.zeros_like(Lambda)

    if use_qi:
        qis = params["qis"]
        mrt = compute_perturbed_mrt(pos_to_compute, Lambda, extra_t, qis, xis, v, regions_indices, large_t=10000,model=model)
    else:
        if model == "Exponential":
            mrt = compute_mrt_exp(pos_to_compute, Lambda, extra_t, xis, v)
        else:
            mrt = compute_mrt_weibull(pos_to_compute, Lambda, extra_t, xis, v)

    theo = mrt
    if measurement_type == "RFD":
        theo = compute_derivatives(mrt, v/resolution, method="central", shift=None)
    if measurement_type == "deltaMRT":
        theo = compute_derivatives(mrt, 1, method="central", shift=None)
    
    return theo

@partial(jax.jit,static_argnames=["model","measurement_type","use_extra_t","use_qi"])
def compute_theo_cached(params,delta_pos_over_v,prev_extra_t, prev_sorted,pos_to_compute,xis,v,measurement_type,use_extra_t,use_qi,regions_indices,resolution,model="Exponential"):
    Lambda = params["kis"]
    if use_extra_t:
        extra_t = params["extra_t"]
        extra_t -= jnp.min(extra_t)
    else:
        extra_t = jnp.zeros_like(Lambda)

    if use_qi:
        qis = params["qis"]
        mrt = compute_perturbed_mrt(pos_to_compute, Lambda, extra_t, qis, xis, v, regions_indices, large_t=10000,model=model)
    else:
        if model == "Exponential":
            mrt = compute_mrt_exp(pos_to_compute, Lambda, extra_t, xis, v)
        else:
            mrt,extra_t,sorted_a = compute_mrt_weibull_cached(Lambda, extra_t, delta_pos_over_v,prev_extra_t, prev_sorted)

    theo = mrt
    if measurement_type == "RFD":
        theo = compute_derivatives(mrt, v/resolution, method="central", shift=None)
    if measurement_type == "deltaMRT":
        theo = compute_derivatives(mrt, 1, method="central", shift=None)
    
    return theo,extra_t,sorted_a


def weibull_logpdf(x, shape, scale):
    """Log-PDF of the Weibull distribution, compatible with JAX.
    
    Args:
        x: Input value (>=0).
        shape: Shape parameter (k > 0).
        scale: Scale parameter (λ > 0).
        
    Returns:
        log_pdf: Log probability density at `x`.
    """
    # Ensure numerical stability for x <= 0
    safe_x = jnp.where(x > 0, x, 1e-8)  # Avoid log(0) or negative inputs
    
    # Compute log terms
    log_term = (
        jnp.log(shape) 
        - shape * jnp.log(scale) 
        + (shape - 1) * jnp.log(safe_x) 
        - (safe_x / scale) ** shape
    )
    
    # Mask invalid values (x <= 0)
    return jnp.where(x > 0, log_term, -jnp.inf)


@partial(jax.jit,static_argnames=["use_qi","use_extra_t"])
def log_prior_fun(theta,prior_lambda,prior_qi,prior_extra_t,use_qi=False,use_extra_t=True):
    log_prior = 0.0
    
    kis = theta['kis']
    kis = jnp.clip(kis,1e-7,100)

    # Prior for kis: Weibull(2, init_v[i])
    log_prior += jnp.sum(weibull_logpdf(kis, shape=2, scale=prior_lambda))    
    # Prior for qis: Beta if present
    if use_qi:
        f = 10
        alpha_p = f * prior_qi
        beta_p = f * (1 - prior_qi)
        log_prior += jnp.sum(beta.logpdf(theta['qis'], alpha_p, beta_p))
    
    # Prior for extra_t: Exponential if fit_time
    if use_extra_t:
        #print("Sub")
        #theta['extra_t'] -= jnp.min(theta['extra_t'] )
        #extra 
        scale_extra_t = prior_extra_t
        log_prior += jnp.sum(expon.logpdf(theta['extra_t'], scale=scale_extra_t))
    
    return log_prior

@partial(jax.jit,static_argnames=["theo"])
def log_lik_fun(params, data, theo ,sigma):
    
    theo_point = theo(params)

    return jnp.sum(norm.logpdf(data, theo_point, sigma))



jax.jit
def log_lik_fun_cached(params,delta_pos_over_v,prev_extra_t, prev_sorted,data, theo ,sigma):
    
    theo_point,extra_t,sorted = theo(params,delta_pos_over_v,prev_extra_t, prev_sorted)

    return jnp.sum(norm.logpdf(data, theo_point, sigma)),extra_t,sorted