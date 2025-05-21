import typer
from dataclasses import dataclass
from dataclasses import asdict
import numpy as np
from typing import List, Tuple

from fast_rep.interface import compute_mrt_and_derivatives
from fast_rep.optim_fix_interval import fit_lambdai
from fast_rep.read_data import  load_RFD_from_bedGraph,write_custom_bedgraph,write_custom_bedgraph_pandas,load_muli_from_bedGraph
from fast_rep.validate_region import validate_region
from fast_rep.interface import compute_mrt_and_derivatives_pos
import os
from fast_rep.rfd_tools import smooth,extract_most_potent_ori_around_expected_value
from typing import Optional

import jax
import jax.numpy as jnp
#jax.config.update("jax_enable_x64", False)  # Ensure computations default to 32-bit
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent
import inspect
from fast_rep.rfd_tools import find_ori_position,convert_RFD_delta_MRT
from fast_rep.bayesian_optim import fit_at_pos_using_map_as_starting_point,estimate_n_ori
import pickle

import click
from typing import Annotated

app = typer.Typer()


@app.command()
def fit_origins(
    bedgraph_file: str = typer.Argument(..., help="Path to bedGraph file"),
    output_file: str = typer.Argument(..., help="Path to output BED file"),
    fork_speed: float = typer.Argument(..., help="in kb/min"),
    sphase: float = typer.Argument(..., help="S phase duration (in min)"),
    
    # Data loading parameters
    regions: Optional[list[str]] = typer.Option(
        None,
        "--regions",
        "-r",
        help="Genomic regions to process (e.g., 'chr1:1000-2000')"
    ),
    
    # Origin detection parameters
    smoothv: int = typer.Option(11, help="Smoothing window size for origin detection"),
    min_dist_ori: float = typer.Option(1.5, help="Minimum distance between origins (kb)"),
    expected_dist_ori: float = typer.Option(20.0, help="expected distance between origins (kb)"),

    min_rfd_increase: float = typer.Option(
        0.1, 
        help="Minimum RFD increase per kb to detect origin"
    ),
    
    # Model fitting parameters
    prior_lambda: float = typer.Option(2.0, help="Prior variance for lambda parameter in S phase relation-ship"),
    prior_extra_t: float = typer.Option(100.0, help="Prior variance for extra time in S phase relation-ship"),
    noise: float = typer.Option(None, help="Add noise on synthetic data"),

    signal_to_fit: str = typer.Option(
        "RFD", 
        help="signal to fit in the bed"
    ),
    fit_mode: Annotated[str, typer.Option(click_type=click.Choice(["MAP","Laplace","ADVI"]),
                                            help="Type of fitting")
                                            ] = "Laplace" ,
                
    prior_qis: float = typer.Option(None, help="Prior variance for qis parameters"),
    #model_type: Annotated[str, typer.Option(click_type=click.Choice(['1PL', '2PL', '3PL']))] = '1PL' , 
    model_type: Annotated[str, typer.Option(click_type=click.Choice(["Exponential","Weibull"]),
                                            help="Replication timing model (Exponential/Weibull)")
                                            ] = "Exponential" ,
    delta: int = typer.Option(  15,
        help="Subsampling of rfd data"
    ),

    final_samples: int = typer.Option(
        50,
        help="Number of samples for final parameter estimation"
    ),
    
    # Runtime parameters
    starting_points: Optional[str] = typer.Option(
        None,
        help="File containing initial starting points for optimization"
    ),
    parallel: bool = typer.Option(False, help="Enable parallel processing"),
    verbose: bool = typer.Option(True, help="Show detailed output"),
    fit_time: bool = typer.Option(False, help="fit a delay for origin activation"),

):
    """
    Detect and characterize replication origins from RFD data.
    """
    # Load and validate data
    regions_str = ",".join(regions) if regions else None
    try:
        RFD, resolution, meta = load_muli_from_bedGraph(bedgraph_file,
                                                        regions_str,
                                                        column_specs=["RFD","std_RFD","smth_RFD"])
    except:
        RFD, resolution, meta = load_muli_from_bedGraph(bedgraph_file,
                                                        regions_str,
                                                        column_specs=[signal_to_fit])
    
    resolution *= 1000 # in bp 
    
    dir = os.path.split(output_file)[0]
    os.makedirs(dir,exist_ok=True)
        # Add error weighting logic here based on main command
    
    measurement_type = "RFD"

    # Prepare output data structure
    origins = []

    def process_region(key):
        """Process a single genomic region"""

        rfd = RFD[key]["signals"][signal_to_fit]

        expected_n_ori = int(len(rfd) * resolution /  (expected_dist_ori*1000))


        if noise != None:
            GT = rfd + (1-2*np.random.rand(len(rfd))) * noise
        else:
            GT = rfd

        positions = RFD[key]["start"]

        #Needed if from experimental data
        if "std_RFD" in  list(RFD[key]["signals"].keys()):
            std_RFD = RFD[key]["signals"]["std_RFD"]
        

        
        if "smth_RFD" in  list(RFD[key]["signals"].keys()):

            smth_rfd = RFD[key]["signals"]["smth_RFD"]
        else:
            if noise:
                smth_rfd = smooth(GT,smoothv)
            else:
                smth_rfd = smooth(RFD[key]["signals"],smoothv)


        xis,delta_v,vals=find_ori_position(data={"rfd":smth_rfd,"positions":positions},
                                           min_dist_ori=min_dist_ori * 1000,
                                            smoothv=smoothv,
                                            min_rfd_increase_by_kb=0) 


        pos_to_compute,data = convert_RFD_delta_MRT(positions,GT,speed=fork_speed,resolution=resolution,delta=delta,measurement_type=measurement_type)
        Ori_pos = extract_most_potent_ori_around_expected_value(xis,vals["prominences"],expected_n_ori,
                                                                max_factor_expected=2.5,
                                                                min_factor_expected=0.7)
        if noise != None:
            sigma = np.zeros_like(data)+noise/delta**0.5
        else:
            sigma = smooth(std_RFD,delta)[::delta]


        #convert data
        
        # Fit origins


        if not fit_time:
            inv_prior_on_extra_t0 = None
        else:
            inv_prior_on_extra_t0 = sphase/prior_extra_t

        print(inv_prior_on_extra_t0)
        dump_file = output_file.replace(".bed","-") + key + ".pickle"
        
  
        results = estimate_n_ori(
             Ori_pos=Ori_pos,
             pos_to_compute=pos_to_compute,
            inv_prior_on_lambda =sphase/prior_lambda,
            inv_prior_on_extra_t=inv_prior_on_extra_t0,
            fork_speed=fork_speed,
            data=data,
            S_phase_duration=sphase,
            sigma=sigma,
            measurement_type=measurement_type,
            model=model_type,
            verbose=verbose,
            mode=fit_mode,
            independent=False
        )
        with open(dump_file,"wb") as dump_f:
            pickle.dump(results,dump_f)
        signals = {}

        #all the
        m  = np.argmax(results["elbo"])

        params = results["detail_results"][m]["params"]
        xis = results["origin_positionning"][m]

    # Convert results to BED format
        is_ori = np.zeros_like(rfd)
        index_ori = np.array(xis//resolution,dtype=int)
        is_ori[index_ori] = 1

        fitted_lambda = np.zeros_like(rfd)
        fitted_lambda[index_ori] = params["kis"]
        fitted_delay = np.zeros_like(rfd)

        extra_t = np.zeros_like(params["kis"])
        if fit_time:
            fitted_delay[index_ori] = params["extra_t"]
            extra_t = params["extra_t"]

        print("Best elbo" ,results["elbo"][m])
        print(params["kis"])


        mrtr,rfd = compute_mrt_and_derivatives_pos(positions,
                                                params["kis"], 
                                                extra_t, 
                                                xis, 
                                                v=fork_speed, 
                                                model=model_type,
                                                method='central', 
                                                shift=1,
                                                resolution=resolution)

        extra=f"_{len(xis)}"
        extra=""
        signals = {"GT":GT,
                    "is_ori" + extra:is_ori,
                    "lambdai" + extra: fitted_lambda,
                    "theo_rfd" + extra: rfd,
                    "theo_mrt" + extra: mrtr}
        
        
        return key,signals

    # Process regions
    keys = list(RFD.keys())

    if parallel:

        with ProcessPoolExecutor() as executor:
            args =   keys
            results = executor.map(process_region, args)

        # Update RFD with results
        for key, processed in results:
            RFD[key]["signals"] = processed

    else:
        for key in keys:
            key, processed = process_region((key))
            RFD[key]["signals"] = processed



    # Write output
    write_custom_bedgraph_pandas(output_file, RFD,resolution=resolution/1000,meta={"model":model_type,"fork_speed":fork_speed})

if __name__=="__main__":
    app()