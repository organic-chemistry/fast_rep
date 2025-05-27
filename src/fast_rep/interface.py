from functools import partial
import jax
import jax.numpy as jnp


from fast_rep.math_mod.loop_mrt import compute_updates
from fast_rep.math_mod.conv_mrt import compute_updates_conv
from fast_rep.math_mod.compute_rfd import compute_derivatives
from fast_rep.simulations import sim
from fast_rep.math_mod.compute_mrt_at_pos import compute_mrt_exp,compute_mrt_weibull,compute_E_weibull,compute_E_exp


parallel=False
convolve=False
if parallel:
    raise" does not work yet"
    updates = parallel_compute_updates
else:
    updates = compute_updates




@partial(jax.jit,static_argnames=["max_n","sum_by_map"])
def compute_mrt(lambdai, max_n, v, tolerance=1e-4,sum_by_map=True):
    n_values = jnp.arange(max_n)
    return updates(lambdai, n_values, v, tolerance=tolerance,sum_by_map=sum_by_map)

"""
@partial(jax.jit,static_argnames=["max_n","sum_by_map"])
def compute_mrt(lambdai, max_n, v, tolerance=1e-4,sum_by_map=True):


    data = sim(10000, 1/lambdai, v, "Exponential")
    return data["MRT_time"]
"""
if convolve:
    def compute_mrt(lambdai, max_n, v, tolerance=1e-4):
        return compute_updates_conv(lambdai, max_n, v)
    

@partial(jax.jit,static_argnames=["max_n",'method', 'shift',"max_n"])
def compute_mrt_and_derivatives(lambdai, max_n, v, method='forward', shift=None,tolerance=1e-4,resolution=1):
    """
    Compute both MRT and its derivatives in one pass.
    
    Returns:
    --------
    mrt: array
        Mean Response Time values
    derivative: array
        The derivative of MRT (RFD or delta_MRT based on method)
    """
    mrt = compute_mrt(lambdai, max_n, v,tolerance=tolerance)
    derivative = compute_derivatives(mrt, v, method, shift,resolution=resolution)
    return mrt, derivative



@partial(jax.jit,static_argnames=['method', 'shift',"model"])
def compute_mrt_and_derivatives_pos(pos_to_compute,Lambda, extra_t, xis, v, model="Exponential",method='forward', shift=None,resolution=1.):
    """
    Compute both MRT and its derivatives in one pass.
    
    Returns:
    --------
    mrt: array
        Mean Response Time values
    derivative: array
        The derivative of MRT (RFD or delta_MRT based on method)
    """
    if model == "Exponential":
        mrt = compute_mrt_exp(pos_to_compute,Lambda, extra_t, xis, v)
    else:
        mrt = compute_mrt_weibull(pos_to_compute,Lambda, extra_t, xis, v)

    derivative = compute_derivatives(mrt, v, method, shift,resolution=resolution)
    return mrt, derivative

def compute_E(Lambda, extra_t, xis, v, model="Exponential"):
    if model == "Exponential":
        return compute_E_exp(Lambda, extra_t, xis, v)

    else:
        return compute_E_weibull(Lambda, extra_t, xis, v)

