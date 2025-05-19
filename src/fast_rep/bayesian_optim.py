from jax_advi.advi import optimize_advi
from jax_advi.constraints import constrain_range_stable,constrain_sigmoid_stable,constrain_exp
from jax_advi.guide import MAPGuide,LowRankGuide , MeanFieldGuide,FullRankGuide, VariationalGuide,LaplaceApproxGuide
from typing import Tuple, Dict, Callable, Any,List


from dataclasses import dataclass
from fast_rep.bayesian_formulation import log_lik_fun,log_prior_fun,compute_theo
from fast_rep.math_mod.compute_mrt_at_pos import generate_regions

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


def generate_shapes_and_constraints(n_ori,use_qi,use_extra_t):
    e=1e-10
    theta_shapes = {'kis': (n_ori,)}
    theta_constraints = {'kis': constrain_exp}
    #theta_constraints = {}
    if use_qi:

        theta_shapes['qis'] = (n_ori,)
        theta_constraints['qis'] = constrain_exp # Custom constraint for [0,1]
        #theta_constraints['qis'] = lambda x:constrain_range_stable(x,e,1-e)  # Custom constraint for [0,1]
    if use_extra_t:
        theta_shapes['extra_t'] = (n_ori,)
        theta_constraints['extra_t'] = constrain_exp
    return theta_shapes,theta_constraints


def generate_shapes_and_constraintso(n_ori,use_qi,use_extra_t):
    e=1e-10
    theta_shapes = {'kis': (n_ori,)}
    theta_constraints = {'kis': lambda x:constrain_range_stable(x,e,10)}
    #theta_constraints = {}
    if use_qi:

        theta_shapes['qis'] = (n_ori,)
        theta_constraints['qis'] = lambda x:constrain_range_stable(x,0.8,1-e)  # Custom constraint for [0,1]
        #theta_constraints['qis'] = lambda x:constrain_range_stable(x,e,1-e)  # Custom constraint for [0,1]
    if use_extra_t:
        theta_shapes['extra_t'] = (n_ori,)
        theta_constraints['extra_t'] = lambda x:constrain_sigmoid_stable(x,100)
    return theta_shapes,theta_constraints



@dataclass(frozen=True)
class ADVI_params:
    M: int = 20
    opt_method: str  = "L-BFGS-B"
    verbose: bool = True
    guide: VariationalGuide = MeanFieldGuide()  # OptiGuide , FullRankGuide

def fit_at_pos(
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
    minimize_kwargs={},
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
    compute_theo_measurement = lambda x:compute_theo(x,pos_to_compute,xis,v,measurement_type,use_extra_t,use_qi,regions_indices,resolution,model)
    
    # Initialize optimizer

    curried_lik = jax.jit(partial(log_lik_fun, data=data, theo = compute_theo_measurement ,sigma = sigma))

    curried_prior  = jax.jit(partial(log_prior_fun,prior_lambda=prior_on_lambda,prior_qi=prior_on_qis,prior_extra_t=prior_on_extra_t,
                                     use_qi=use_qi,use_extra_t=use_extra_t))
    #Is it better to use regular adam optimiser?
    
    theta_shapes,theta_constraints = generate_shapes_and_constraints(len(xis),use_qi,use_extra_t)

    result = optimize_advi(
        theta_shapes,
        log_prior_fun=curried_prior,
        log_lik_fun=curried_lik,
        constrain_fun_dict=theta_constraints,
        verbose=ADVI_P.verbose,
        guide=ADVI_P.guide, #OptiGuide() #FullRankGuide() ,#LowRankGuide(rank=5) , #FullRankGuide(),#MeanFieldGuide(),#LowRankGuide(rank=5), # FullRankGuide(), #,
        M=ADVI_P.M ,  # Number of MC samples
        opt_method=ADVI_P.opt_method,
        minimize_kwargs=minimize_kwargs
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

    theo = compute_theo_measurement(best_params)

    return {"params":best_params,"elbo":result["elbo"],"other":result,
            "theo":theo}



def fit_at_pos_using_map_as_starting_point(pos_to_compute,xis,prior_on_lambda,prior_on_extra_t,prior_on_qis,
          fork_speed,data,S_phase_duration,sigma,measurement_type,model,starting_points=None,
          only_optim=False,verbose=True,final_M=50,mode="MAP"):
    
    if (not starting_points) or (mode in ["MAP","Laplace"]):
        
        G = MAPGuide()
        if mode == "Laplace":
            G = LaplaceApproxGuide()
        G.init_mean_scale=-2
        r= fit_at_pos(
            pos_to_compute=pos_to_compute, # en bp
            xis=xis,
            prior_on_lambda=prior_on_lambda,
            prior_on_extra_t=prior_on_extra_t,
            prior_on_qis=prior_on_qis,
            v=fork_speed, # en bp/minute
            data=data,
            S_phase_duration=S_phase_duration,  # en minute
            sigma=sigma,
            measurement_type=measurement_type,
            model=model,
            ADVI_P= ADVI_params(1,"L-BFGS-B",verbose,G),
            minimize_kwargs={"tol":1e-6})

        if mode in ["MAP","Laplace"]:
            return r
        #M = FullRankGuide()
        M = MeanFieldGuide()
        a = np.array(jnp.diag(r["other"]["hessian"]))
        #print(a.shape)
        stdv =  1/(a+1e-7)
        stdv[stdv<1e-1] = 1e-1  #Needed because if not lead to early convergence to higher elbo
        stdv = np.log(stdv)
        #print(a)
        M.init_mean_scale = r["other"]["free_params"]
        M.init_log_sd_scale = stdv #-1 #
        
        #ld = r["other"]["free_params"][npv:]
    else:
        M = MeanFieldGuide()
        M.init_mean_scale = starting_points[0]
        M.init_log_sd_scale = starting_points[1]
    r= fit_at_pos(
        pos_to_compute=pos_to_compute, # en bp
        xis=xis,
        prior_on_lambda=prior_on_lambda,
        prior_on_extra_t=prior_on_extra_t,
        prior_on_qis=prior_on_qis,
        v=fork_speed, # en bp/minute
        data=data,
        S_phase_duration=S_phase_duration,  # en minute
        sigma=sigma,
        measurement_type=measurement_type,
        model=model,
        ADVI_P= ADVI_params(final_M,"L-BFGS-B",verbose,M),#ADVI_params(20,"L-BFGS-B",True,M),
        minimize_kwargs={"tol":1e-6})

    return r



def estimate_n_ori(Ori_pos,model,pos_to_compute,data,sigma,
                   inv_prior_on_lambda,inv_prior_on_extra_t,S_phase_duration,fork_speed,
                   measurement_type,verbose=False,independent=False,mode="ADVI"):
    
    print(Ori_pos,model,pos_to_compute,data,sigma,inv_prior_on_lambda,inv_prior_on_extra_t,S_phase_duration,fork_speed,measurement_type)

    #raise
    no = []
    elbos = []
    Pos_oris = []
    Results = []

    fit_time = True
    if inv_prior_on_extra_t == None:
        fit_time = False

    for xis in Ori_pos:
        n_ori = len(xis)


        initial_lambda = jnp.array([1/(inv_prior_on_lambda)] * n_ori)
        if fit_time:
            init_extra_t = jnp.array([1/inv_prior_on_extra_t]  * n_ori)
        else:
            init_extra_t = None

        init_qis = None # jnp.array([0.9]  * n_ori)
        if len(Pos_oris) == 0 or (independent):
            starting_points = []
        else:
            old=[]
            for ori in xis:
                if ori in Pos_oris:
                    old.append(Pos_oris.index(ori))
            old = np.array(old)

            G = r["other"]["guide"]
            if mode == "ADVI":
                mu_p,std_p = G.get_stuctured_mean_std_before_transformation()

                init_mean_scale = list(mu_p["kis"][old]) 
                init_log_sd_scale = list(jnp.log(std_p["kis"][old]) )
                if fit_time:
                    init_mean_scale += list(mu_p["extra_t"][old])
                    init_log_sd_scale += list(jnp.log(std_p["extra_t"][old]))

                init_mean_scale = np.array(init_mean_scale)
                init_log_sd_scale = np.array(init_log_sd_scale)
                starting_points=[init_mean_scale,init_log_sd_scale]
            else:
                starting_points=[]#[G.final_var_params_flat]

            #init_log_sd_scale = -1
        Pos_oris = list(xis)
            
        #print(len(xis),len(init_extra_t),len(initial_lambda))
        r = fit_at_pos_using_map_as_starting_point(pos_to_compute=pos_to_compute,xis=xis,prior_on_lambda=initial_lambda,prior_on_extra_t=init_extra_t,
                                                  prior_on_qis=init_qis,fork_speed=fork_speed,data=data,
                                                  S_phase_duration=S_phase_duration,sigma=sigma,measurement_type=measurement_type,verbose=verbose,
                                                  model=model,mode=mode,starting_points=starting_points)


        no.append(len(xis))
        elbos.append(r["elbo"])
        Results.append(r)
    return {"number_of_origins":no,"origin_positionning":Ori_pos,"elbo":elbos,"detail_results":Results}