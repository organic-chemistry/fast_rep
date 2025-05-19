from jax_advi.advi import optimize_advi
from jax_advi.constraints import constrain_range_stable,constrain_sigmoid_stable,constrain_exp
from jax_advi.guide import LowRankGuide , MeanFieldGuide,FullRankGuide, VariationalGuide
from typing import Tuple, Dict, Callable, Any,List
import numpy as np
import optax
import optax.tree_utils as otu


from dataclasses import dataclass
from fast_rep.bayesian_formulation import log_lik_fun,log_prior_fun,compute_theo_cached,log_lik_fun_cached
from fast_rep.math_mod.compute_mrt_at_pos import generate_regions,MRTState

from functools import partial
import jax
import jax.numpy as jnp


from jax_advi.advi import reconstruct,apply_constraints,flatten_and_summarise,optimize_with_jac,optimize_with_hvp
from fast_rep.bayesian_optim import generate_shapes_and_constraints,log_prior_fun,ADVI_params



def _calculate_log_posterior(
    flat_theta, log_lik_fun, log_prior_fun, constrain_fun_dict, summary,delta_pos_over_v,prev_extra_t, prev_sorted,verbose=False
):

    cur_theta = reconstruct(flat_theta, summary, jnp.reshape)
    #if verbose:
    #    print(cur_theta)

    # Compute the log determinant of the constraints
    cur_theta, cur_log_det = apply_constraints(cur_theta, constrain_fun_dict)
    if verbose:
        print(cur_theta)
    cur_likelihood,extra_t, sorted_t= log_lik_fun(cur_theta,delta_pos_over_v,prev_extra_t, prev_sorted)
    cur_prior = log_prior_fun(cur_theta)
    if verbose:
        print("like",cur_likelihood)
        print("prior",cur_prior)
        print("logdet",cur_log_det)

    return cur_likelihood + cur_prior + cur_log_det,extra_t,sorted_t


def _calculate_objective(var_params_flat,delta_pos_over_v,prev_extra_t, prev_sorted, summary, constrain_fun_dict, 
                        log_lik_fun, log_prior_fun, zs,guide: VariationalGuide,verbose):
    
    cur_entropy = guide.entropy(var_params_flat)
    
    def calculate_log_posterior(z,delta_pos_over_v,prev_extra_t, prev_sorted):
        flat_theta = guide.transform_draws(z, var_params_flat)
        return _calculate_log_posterior(
            flat_theta, log_lik_fun, log_prior_fun, constrain_fun_dict, summary,delta_pos_over_v,prev_extra_t, prev_sorted
        )
    print(zs.shape,delta_pos_over_v.shape,prev_extra_t.shape, prev_sorted.shape)
    posterior,extra_t,sorted_t = jax.vmap(calculate_log_posterior,in_axes=(0,0,0,0))(zs,delta_pos_over_v,prev_extra_t, prev_sorted)
    return -jnp.mean(posterior) - cur_entropy , extra_t,sorted_t


def _build_objective_fun(theta_shape_dict, constrain_fun_dict, log_lik_fun, 
                        log_prior_fun, seed, M, guide: VariationalGuide,verbose=False):
    
    theta = {k: jnp.empty(v) for k, v in theta_shape_dict.items()}
    flat_theta, summary,indices = flatten_and_summarise(**theta)
    print("ft",flat_theta)
    var_params = guide.init_params(flat_theta)
    print(var_params)
    
    # Generate noise draws
    zs = jax.random.normal(jax.random.PRNGKey(seed), (M, guide.z_dim()))
    # Create objective
    to_minimize = partial(
        _calculate_objective,
        summary=summary,
        constrain_fun_dict=constrain_fun_dict,
        log_lik_fun=log_lik_fun,
        log_prior_fun=log_prior_fun,
        zs=zs,
        guide=guide,
        verbose=verbose
    )
    
    return flat_theta, summary, to_minimize,var_params



def value_and_grad_from_state( value_fn) :
  r"""Alternative to ``jax.value_and_grad`` that fetches value, grad from state.

  """

  def _value_and_grad(
      params,
      aux ,
      state,
  ):
    value = otu.tree_get(state, 'value')
    grad = otu.tree_get(state, 'grad')
    if (value is None) or (grad is None):
      raise ValueError(
          'Value or gradient not found in the state. '
          'Make sure that these values are stored in the state by the '
          'optimizer.'
      )
    (value,aux), grad = jax.lax.cond(
        (~jnp.isinf(value)) & (~jnp.isnan(value)),
        lambda *_: ((value,aux), grad),
        lambda p, a : jax.value_and_grad(value_fn,has_aux=True)(p, a ),
        params,
        aux,
    )
    return (value,aux), grad

  return _value_and_grad


def optimize_with_optax(to_minimize,initial_params, initial_cache, max_iter=100, patience=5, tol=1e-4):
    # Initialize optimizer
    opt = optax.lbfgs()
    #delta_pos_over_v,prev_extra_t,prev_sorted = initial_state

    @jax.jit
    def compute_loss(params, cached):
        """Compute loss and updated state together"""
        # Your custom computation using params and previous state
        print("P",params)
        print(cached)
        delta_pos_over_v,prev_extra_t,prev_sorted = cached
        result, new_extra_t, new_sorted = to_minimize(
            params, 
            delta_pos_over_v,
            prev_extra_t,
            prev_sorted
        )
        
        # Return loss + new state
        return result,(delta_pos_over_v,new_extra_t, new_sorted )

    # Value and grad function
    value_and_grad_fun =  value_and_grad_from_state(compute_loss)

    @jax.jit
    def step(carry):
        (params,cached), state = carry
        #print("P2",params)
        #jax.debug.print("D {params}",params=params)
        #jax.debug.print("D ")

        (value,cached), grad = value_and_grad_fun(params,cached,state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, 
            value_fn=lambda x:compute_loss(x,cached)[0]
        )
        params = optax.apply_updates(params, updates)
        #print("Parasm",params)
        jax.debug.print("loss {loss} ",loss=value)
        return (params,cached), state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))
    

    init_carry = ((initial_params,initial_cache), opt.init(initial_params))
    #final_params, final_state = jax.lax.while_loop(
    #    continuing_criterion, step, init_carry
    #)
    while_loop = jax.jit(lambda carry: jax.lax.while_loop(continuing_criterion, step, carry))
 
    (final_params,cached), final_state = while_loop(init_carry)

    return final_params, compute_loss(final_params,cached)[0]



def optimize_advi(
    theta_shape_dict: Dict[str, Tuple],
    log_prior_fun: Callable[[Dict[str, jax.Array]], float],
    log_lik_fun: Callable[[Dict[str, jax.Array]], float],
    guide: VariationalGuide = MeanFieldGuide(),
    state = None,
    M: int = 100,
    constrain_fun_dict: Dict = {},
    verbose: bool = False,
    seed: int = 2,
    n_draws: int = 1000,
    opt_method: str = "trust-ncg",
    minimize_kwargs = {}
) -> Dict[str, Any]:
    """
    Redifinition for caching the sorted indices
    """
    
    # Build objective function
    flat_theta, summary, to_minimize, var_params = _build_objective_fun(
        theta_shape_dict, constrain_fun_dict, log_lik_fun, log_prior_fun, 
        seed, M, guide
    )
    print(guide)
    print(flat_theta.shape)
    
   # print("STate",O.state.prev_extra_t)

    # Initialize variational parameters
    # Optimization
  
    if opt_method == "L-BFGS-B":
        final_params, final_loss = optimize_with_optax(to_minimize,
        jnp.zeros_like(var_params.reshape(-1)),
        (state.delta_pos_over_v,state.prev_extra_t+0.0,state.prev_sorted),
        max_iter=1000,
        patience=5,
        tol=1e-4
    )

    else:
        result = optimize_with_hvp(O.objective, var_params.reshape(-1), opt_method, verbose,minimize_kwargs=minimize_kwargs)[0]
    print("STate",state.prev_extra_t)

    #To be able to drow from the guide:
    guide.final_var_params_flat =final_params
    guide.summary = summary
    guide.constrain_fun_dict = constrain_fun_dict
    print(final_loss)

    #O(result.x)

    return  {
        "free_params": final_params,
        "elbo" : -final_loss,
       # "opt_result": result,
        "guide":guide,
        "draws": guide.posterior_draw_and_transform(n_draws)
    }




def fit_at_pos_cached(
    pos_to_compute,
    xis,
    data,
    prior_on_lambda,
    prior_on_extra_t=None,
    prior_on_qis=None,
    S_phase_duration = None,    # needed for region computation
    v: float = 1000,    # needed for region computation
    measurement_type: str = "MRT",
    sigma = 0.01/5**0.5,
    ADVI_P = ADVI_params(),
    model="Exponential"):
    
    resolution= pos_to_compute[1]-pos_to_compute[0]
    assert(jnp.all( (pos_to_compute[1:]-pos_to_compute[:-1]) == resolution))
    # Initialize parameters
    use_qi = prior_on_qis is not None
    use_extra_t = prior_on_extra_t is not None
    
    print("use_qi",use_qi)
    print("use_extrat",use_extra_t)
        
    regions_indices=None
    if prior_on_qis is not None:
        if S_phase_duration == None:
            raise
        l = int((S_phase_duration * v) * jnp.mean(pos_to_compute[1:]-pos_to_compute[:-1]))
        regions_indices = generate_regions(pos_to_compute,xis,l)
    #print(measurement_type)


    def compute_theo_measurement(params,delta_pos_over_v,prev_extra_t, prev_sorted):

        return compute_theo_cached(params,delta_pos_over_v,prev_extra_t, prev_sorted,pos_to_compute,xis,v,measurement_type,use_extra_t,use_qi,regions_indices,resolution,model)
    
    # Initialize optimizer

    def log_lik_wrapper(params, delta_pos_over_v, prev_extra_t, prev_sorted):
        return log_lik_fun_cached(
            params, delta_pos_over_v, prev_extra_t, prev_sorted, data, compute_theo_measurement,sigma
        )
   # params,delta_pos_over_v,prev_extra_t, prev_sorted,data, theo ,sigma):

    curried_lik = jax.jit(log_lik_wrapper)

    curried_prior  = jax.jit(partial(log_prior_fun,prior_lambda=prior_on_lambda,prior_qi=prior_on_qis,prior_extra_t=prior_on_extra_t,
                                     use_qi=use_qi,use_extra_t=use_extra_t))
    #Is it better to use regular adam optimiser?
    
    theta_shapes,theta_constraints = generate_shapes_and_constraints(len(xis),use_qi,use_extra_t)

    state = MRTState(xis,pos_to_compute,v,jnp.zeros_like(xis),ADVI_P.M)
    result = optimize_advi(
        theta_shapes,
        log_prior_fun=curried_prior,
        log_lik_fun=curried_lik,
        constrain_fun_dict=theta_constraints,
        state=state,
        verbose=ADVI_P.verbose,
        guide=ADVI_P.guide, #OptiGuide() #FullRankGuide() ,#LowRankGuide(rank=5) , #FullRankGuide(),#MeanFieldGuide(),#LowRankGuide(rank=5), # FullRankGuide(), #,
        M=ADVI_P.M ,  # Number of MC samples
        opt_method=ADVI_P.opt_method
        #M=10
    )


    best_params= {"kis":jnp.mean(result["draws"]["kis"],axis=0),
                "kis_std":jnp.std(result["draws"]["kis"],axis=0)}
    if use_extra_t:
        best_params["extra_t"] = jnp.mean(result["draws"]["extra_t"],axis=0)
        best_params["extra_t"] -= jnp.min(best_params["extra_t"])

        best_params["extra_t_std"] = jnp.std(result["draws"]["extra_t"],axis=0)
    if use_qi:  
        best_params["qis"] = jnp.mean(result["draws"]["qis"],axis=0)
        best_params["qis_std"] = jnp.std(result["draws"]["qis"],axis=0)

    #print("lik ",curried_lik(best_params),
    #      "prior ",curried_prior(best_params),
    #      "entropy ",ADVI_P.guide.entropy(result["free_params"]))

    extra_t = jnp.zeros_like(xis)
    delta_pos_over_v = jnp.abs(xis[None, :] - pos_to_compute[:, None]) / v 
    prev_sorted = jnp.argsort(delta_pos_over_v + extra_t[None, :],axis=1)
    print(extra_t.shape,delta_pos_over_v.shape,prev_sorted.shape)


    theo,_,_ = compute_theo_measurement(best_params,delta_pos_over_v,extra_t,prev_sorted)

    return {"params":best_params,"elbo":result["elbo"],"other":result,"theo":theo}