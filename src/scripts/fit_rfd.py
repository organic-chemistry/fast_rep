import typer
from dataclasses import dataclass
from dataclasses import asdict
import numpy as np
from typing import List, Tuple

from fast_rep.losses import ARParams, GPParams, PriorConfig
from fast_rep.interface import compute_mrt_and_derivatives
from fast_rep.optim_fix_interval import fit_lambdai
from fast_rep.read_data import  load_RFD_from_bedGraph,write_custom_bedgraph,write_custom_bedgraph_pandas
from fast_rep.validate_region import validate_region
from fast_rep.rfd_tools import smooth
from typing import Optional

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)  # Ensure computations default to 32-bit
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import inspect

app = typer.Typer()


def filter_params(func, config_dict):
    sig = inspect.signature(func)
    return {k: v for k, v in config_dict.items() if k in sig.parameters}

    
def smart_start(initial_lambdai,data,background_rep,smoothv=20,th=20):

    initial_lambdai[1:-1] = smooth(data[2:],smoothv)-smooth(data[:-2],smoothv)
    thr = np.percentile(initial_lambdai,[100-th])[0]
    initial_lambdai[initial_lambdai<thr]=1e-7
    initial_lambdai = smooth(initial_lambdai,2)
    initial_lambdai += background_rep
    return jnp.array(initial_lambdai)


@dataclass
class FitConfig:
    """Central container for all pipeline parameters"""
    # Parameters that need special handling
    fork_speed: float    # in kb/min
    sphase: float
    prior_config: PriorConfig  # Special case
    sum_by_map: bool
    
    # All other parameters with default values
    experiment_resolution: int = None  #data resolution will be obtained from the bedgraph (In the bedgraph it is in bp), but here it is converted in kb
    method: str = "forward"
    background_rep: float = 0.00001
    fit_resolution: int = 1
    reg_loss: float = 1000.0
    learning_rate: float = 0.001
    num_iterations: int = 1000
    patience: int = 10
    measurement_type: str = "rfd"
    shift: int = 5
    weights: tuple = (1.0, 1.0)
    tolerance: float = 1e-3
    floor_v: float = 1e-6
    flat: bool = False
    parallel: bool = False

    # Computed fields
    max_n: int = None
    v: int = 50  # Default value for numerical stability

    def __post_init__(self):
        # Automatically compute derived values
        self.max_n = int(self.sphase / (self.experiment_resolution / self.fork_speed))



def process_single_region(args):
    """No parameters in signature except config!"""
    key, RFD, config = args
    
    # Access parameters directly from config
    initial_lambdai = np.zeros_like(RFD[key]["signals"]) + config.background_rep
    data = jnp.array(RFD[key]["signals"])
    
    if not config.flat:
        initial_lambdai = smart_start(initial_lambdai, data, config.background_rep)

    # Dynamic parameter passing to fit_lambdai
    extra = filter_params(fit_lambdai, asdict(config))
    extra["prior_config"] = config.prior_config
    #print()
    result = fit_lambdai(
        initial_lambdai=initial_lambdai,
        data=data,
        weight_error=RFD[key]["weights"],
        **extra  # Automatically pass all relevant parameters
    )
    
    # Compute results using config values
    mrt, rfd = compute_mrt_and_derivatives(
        result[0], 
        max_n=config.max_n,
        v=config.v,
        method='forward',
        tolerance=config.tolerance
    )
    

    # Compute MRT and RFD derivatives
    fitted_lambda = result[0]
    mrt, rfd = compute_mrt_and_derivatives(fitted_lambda, config.max_n, config.v, method='forward', shift=None, tolerance=config.tolerance)

    return key, {"lambdai": fitted_lambda, "original_rfd": RFD[key]["signals"], "theo_rfd": rfd, "theo_mrt": mrt}

def run_fitting(RFD,config,parallel=True):
    keys = list(RFD.keys())

    if parallel:
        with ProcessPoolExecutor() as executor:
            args =   [(key, RFD, config) for key in RFD]
            results = executor.map(process_single_region, args)

        # Update RFD with results
        for key, processed in results:
            RFD[key]["signals"] = processed

    else:
        for key in keys:
            key, processed = process_single_region((key, RFD, config))
            RFD[key]["signals"] = processed

    return RFD


@app.command()
def main(
    bedgraph_file: str = typer.Argument(..., help="Path to bedGraph file"),
    output_file: str = typer.Argument(..., help="Path to bedGraph file"),

    fork_speed: float = typer.Argument(..., help="in kb/min"),
    sphase: float = typer.Argument(..., help="S phase duration (in min)"),

    error_bedgraph_file: str = typer.Argument("", help="Path to count of corresponding to RFD file"),


    regions: Optional[list[str]] = typer.Option(
        None,
        "--regions",
        "-r",
        help=(
            "Regions to process (e.g., 'chr1:1000-2000'). "
            "Multiple regions can be comma-separated or passed as separate arguments."
        ),
   #     callback=lambda x: [validate_region(r) for r in x],
        metavar="REGION[,REGION...]",
    ),

    parallel: bool = typer.Option(
        False,
        help="Run all the computation in parallel"
    ),
    flat: bool = typer.Option(
        False,
        help="Start with a flat profile"
    ),

    method: str = typer.Option(
        "forward",
        help="Derivative method (e.g., forward, central)"
    ),
    background_rep: float = typer.Option(
        0.00001,
        help="Initial background replication"
    ),
    fit_resolution: int = typer.Option(
        1,
        help="Downsampling factor for computed values"
    ),
    reg_loss: float = typer.Option(
        1000.0,
        help="Weight of the regularization term in the loss function"
    ),
    learning_rate: float = typer.Option(
        0.001,
        help="Learning rate for the optimizer"
    ),
    num_iterations: int = typer.Option(
        1000,
        help="Maximum number of training iterations"
    ),
    patience: int = typer.Option(
        10,
        help="Patience for early stopping"
    ),
    measurement_type: str = typer.Option(
        "rfd",
        help="Measurement type (e.g., rfd, mrt, delta_mrt)"
    ),
    shift: int = typer.Option(
        5,
        help="Shift parameter for delta_mrt measurement type"
    ),
    weights_str: str = typer.Option(
        "1.0,1.0",
        help="Comma-separated weights for combined loss (e.g., '1.0,1.0')"
    ),
    prior_type: str = typer.Option(
        "AR",
        help="Prior type (AR or GP)"
    ),
    ar_sigma: float = typer.Option(
        10.0,
        help="Sigma parameter for AR prior (only if prior_type=AR)"
    ),
    ar_rho: float = typer.Option(
        0.7,
        help="Rho parameter for AR prior (only if prior_type=AR)"
    ),
    gp_mean: float = typer.Option(
        0.0,
        help="Mean parameter for GP prior (only if prior_type=GP)"
    ),
    gp_sigma: float = typer.Option(
        1.0,
        help="Sigma parameter for GP prior (only if prior_type=GP)"
    ),
    gp_lengthscale: float = typer.Option(
        10.0,
        help="Lengthscale parameter for GP prior (only if prior_type=GP)"
    ),
    tolerance: float = typer.Option(
        1e-3,
        help="Tolerance for MRT computation convergence"
    ),
    floor_v: float = typer.Option(
        1e-6,
        help="Minimum value for lambdai during optimisation"
    ),
    sum_by_lax: bool = typer.Option(
        False,
        help="Compute cumulative sum by lax_associative (it is faster on gpu)"
    ),

):
    """
    Command-line interface for fitting lambda_i parameters using JAX-based optimization.
    """
    #try:
        # Parse weights
    weights = tuple(map(float, weights_str.split(',')))
    if len(weights) != 2:
        raise ValueError("Weights must be two comma-separated values")

    # Create prior configuration
    if prior_type.upper() == "AR":
        prior_config = PriorConfig(
            type="AR",
            params=ARParams(sigma=ar_sigma, rho=ar_rho)
        )
    elif prior_type.upper() == "GP":
        prior_config = PriorConfig(
            type="GP",
            params=GPParams(
                mean=gp_mean,
                sigma=gp_sigma,
                lengthscale=gp_lengthscale
            )
        )
    else:
        typer.echo(
            f"Error: Invalid prior_type '{prior_type}'. "
            "Must be either 'AR' or 'GP'",
            err=True
        )
        raise typer.Exit(code=1)


    if regions:
        # Flatten comma-separated regions into a list
        all_regions = []
        for region in regions:
            all_regions.extend(region.split(","))
        regions_str = ",".join(all_regions)
    else:
        regions_str = None
        # Load data

    sum_by_map=True
    if sum_by_lax:
        sum_by_map=False



    RFD, experiment_resolution,meta = load_RFD_from_bedGraph(bedgraph_file, regions_str=regions_str)   
    error_RFD = {}

    params = locals()
    config = FitConfig(
        **{k: v for k, v in params.items() if k in inspect.get_annotations(FitConfig)}
    )
    

    if error_bedgraph_file != "": 
        n_RFD, experiment_resolution = load_RFD_from_bedGraph(error_bedgraph_file, regions_str=regions_str)    
        # Use a Beta Prior  =  Beta(0.5,0.5)
        tot_error = []
        for key in RFD:
            p = (1+RFD[key]["signals"])/2
            n = n_RFD[key]["signals"]
            X = p * n
            prior_shift = 20
            p_hat = np.square((X+prior_shift)/(n+2*prior_shift))
            var = p_hat * (1-p_hat) / (n_RFD[key]["signals"]+2*prior_shift)
            error_RFD[key] = 1/var**0.5
            tot_error.append(error_RFD[key])
        mean_error = np.mean(np.concatenate(tot_error))
        #Normalize by the mean error so that tho regularisation is invariant with respect to this number
        for key in RFD:
            RFD[key]["weights"] = jnp.array(error_RFD[key] /mean_error)
    else:
        for key in RFD:
            RFD[key]["weights"]  = jnp.ones_like(RFD[key]["signals"])
    
    # as t1 != 0, all the values can be rescaled to the actual v

    
    RFD = run_fitting(RFD, config, parallel=parallel)
    
    for key in RFD:
        RFD[key]["signals"]["lambdai"]  *= config.fork_speed / config.v
        RFD[key]["signals"]["theo_mrt"]  /= (config.fork_speed / config.v) * 60
        RFD[key]["signals"]["weight"] =  RFD[key]["weights"]

    
    write_custom_bedgraph_pandas(output_file,RFD)

if __name__ == "__main__":
    app()