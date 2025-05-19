import jax
import jax.numpy as jnp
from functools import partial

# First, let's properly define the compute_derivatives function with static_argnames

@partial(jax.jit,static_argnames=['method', 'shift'])
def compute_derivatives(mrt, v, method='forward', shift=None,resolution=1):
    """
    Compute derivatives from MRT values using different methods.
    
    Parameters:
    -----------
    mrt: array
        Input MRT values
    v: float
        Scaling parameter
    method: str
        Method to compute the derivative ('forward', 'central', or 'delta')
    shift: int
        Shift parameter for 'delta' method
        
    Returns:
    --------
    derivative: array
        The computed derivative values
    """
    if method == 'forward':
        # Forward difference
        #derivative = jnp.diff(mrt, append=mrt[-1])
        derivative = jnp.diff(mrt)

        # Append the last element to the derivative
        derivative = jnp.concatenate([derivative,jnp.array([0.])]) / resolution
    elif method == 'central':
        # Central difference for interior, forward/backward for boundaries
        n = len(mrt)
        interior = (mrt[2:] - mrt[:-2]) / 2.0
        left_boundary = mrt[1] - mrt[0]
        right_boundary = mrt[-1] - mrt[-2]
        derivative = jnp.concatenate([left_boundary.reshape(1), interior, right_boundary.reshape(1)]) / resolution
    elif method == 'delta':
        # Delta method
        derivative = (mrt[shift:]-mrt[:-shift]) / resolution
    else:
        raise ValueError(f"Unknown derivative method: {method}")
    
    return derivative * v
