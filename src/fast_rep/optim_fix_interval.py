from fast_rep.losses import loss_function,PriorConfig,ARParams
import optax
import jax
import jax.numpy as jnp
from typing import Tuple



def fit_lambdai(
    initial_lambdai,
    max_n: int,
    v: float,
    data,
    weight_error = None,
    method: str = 'forward',
    fit_resolution: int = 1,
    experiment_resolution: float = 1., #in kb
    reg_loss: float = 1000,
    learning_rate: float = 0.001,
    num_iterations: int = 1000,
    patience: int = 10,
    measurement_type: str = "rfd",
    shift: float = 5,
    weights: Tuple[float, float] = (1.0, 1.0),
    prior_config: PriorConfig = PriorConfig("AR", ARParams()),
    tolerance: float = 1e-3,
    floor_v: float = 1e-6,
    sum_by_map=True

):
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_lambdai)
    
    lambdai = initial_lambdai
    best_loss = float('inf')
    losses = []
    no_improve=0

    if weight_error == None:
        weight_error = jnp.ones_like(initial_lambdai)
    
   
    def loss_fn(params):
        return loss_function(
            params, max_n, v, data, weight_error, fit_resolution, reg_loss, 
            method, measurement_type, shift, weights,prior_config,tolerance,
            sum_by_map
        )
    
    # Get value and gradient function (no jit here, we do that later)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    
    # If you want to jit this, do it explicitly with no static args needed
    # since they're already captured in the closure
    jitted_value_and_grad_fn = jax.jit(value_and_grad_fn)
    
    for i in range(num_iterations):
        # Compute loss and gradient
        loss, grad = jitted_value_and_grad_fn(lambdai)

        losses.append(loss.item())

        if loss < best_loss:
            best_loss = loss
            best_lambdai = lambdai
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at iteration {i}")
               
                break
        
        # Apply updates
        updates, opt_state = optimizer.update(grad, opt_state)
        lambdai = optax.apply_updates(lambdai, updates)
        lambdai = jnp.maximum(lambdai, floor_v) #This has a strong effect on the results it seems to be related to tolerance also... and initial background value
        
        # Early stopping
      
        
        # Progress monitoring
        if i % 100 == 0:
            loss_noreg = loss_function(
                lambdai, max_n, v, data, weight_error, fit_resolution, 0, 
                method, measurement_type, shift, weights,prior_config,tolerance,
                sum_by_map
            )
            #lambdai = smooth(lambdai,2)
            print(f"Iter {i}: Loss={loss:.4f} (NoReg={loss_noreg:.4f})")
    
    loss_noreg = loss_function(
                lambdai, max_n, v, data, weight_error, fit_resolution, 0, 
                method, measurement_type, shift, weights,prior_config,tolerance,
                sum_by_map
            )
            #lambdai = smooth(lambdai,2)
    print(f"Final loss at iter {i}: Loss={loss:.4f} (NoReg={loss_noreg:.4f})")
    return best_lambdai, losses




#Function to update before using
"""
def fit_lambdai_adam_decay(initial_lambdai, n_values, v, data, 
                          init_learning_rate=0.1, decay_steps=100,
                          decay_rate=0.95, num_iterations=2000,
                          patience=20, reg_loss=1000, method='forward',
                          measurement_type="rfd"):
    #Adam optimizer with exponential learning rate decay
    # Create learning rate schedule
    schedule = optax.exponential_decay(
        init_value=init_learning_rate,
        decay_rate=decay_rate,
        transition_steps=decay_steps
    )
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    opt_state = optimizer.init(initial_lambdai)

    # Initialize variables
    lambdai = initial_lambdai
    best_loss = float('inf')
    losses = []
    best_params = initial_lambdai
    no_improve = 0
   
    def loss_fn(params):
        return loss_function(
            params, max_n, v, data, fit_resolution, reg_loss, 
            method, measurement_type, shift, weights,prior
        )
    
    # Get value and gradient function (no jit here, we do that later)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    
    # If you want to jit this, do it explicitly with no static args needed
    # since they're already captured in the closure
    jitted_value_and_grad_fn = jax.jit(value_and_grad_fn)
    
    for i in range(num_iterations):
        # Compute loss and gradient
        loss, grads = jitted_value_and_grad_fn(lambdai)
        losses.append(loss.item())

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        lambdai = optax.apply_updates(lambdai, updates)
        lambdai = jnp.maximum(lambdai, 1e-7)  # Positivity constraint

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            best_params = lambdai.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at iteration {i}")
                loss_noreg = loss_function(lambdai, n_values, v, data,
                                       reg_loss=0, method=method,
                                       measurement_type=measurement_type)
                print(f"Iter {i}: Loss={loss:.4f} (NoReg={loss_noreg:.4f}), LR={current_lr:.2e}")
                break

        # Progress monitoring
        if i % 100 == 0:
            current_lr = schedule(i)
            loss_noreg = loss_function(lambdai, n_values, v, data,
                                       reg_loss=0, method=method,
                                       measurement_type=measurement_type)
            print(f"Iter {i}: Loss={loss:.4f} (NoReg={loss_noreg:.4f}), LR={current_lr:.2e}")

    return best_params, losses
"""