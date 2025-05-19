import typer
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
import jax.numpy as jnp
import jax
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor

from fast_rep.read_data import write_custom_bedgraph_pandas,load_bed_data_regions
from fast_rep.interface import compute_mrt_and_derivatives
from fast_rep.validate_region import validate_region
from fast_rep.simulations import sim
jax.config.update("jax_enable_x64", False)  # Ensure computations default to 32-bit

app = typer.Typer()


def process_single_region(args):
    """Function to process a single region (must be global for multiprocessing)"""
    key, data_item, n_sim, v, distribution, extra_t = args
    
    # Extract lambda values for this region
    lambdai = jnp.array(data_item["signals"]) 
    
    # Run simulation
    data = sim(n_sim, 1/lambdai, v, distribution, extra_t)
    #print(data)
    # Return results
    return key, {
        "chrom": data_item["chrom"],
        "starts": data_item["starts"],
        "ends": data_item["ends"],
        "signals": {
            "lambdai": lambdai,
            "simu_mrt": data["MRT_time"] / 60,
            "simu_rfd": data["RFD"],
        }

        #  "hist_RFD": RFDs,
        # "hist_MRT": MRTs,
        # "activation_times":time_scaled,
        # "mean_MRT_time": jnp.mean(MRTs, axis=0),
        # "mean_RFD": jnp.mean(RFDs, axis=0),
        # "observed_efficiencies": observed_efficiencies,
        # "mean_observed_efficiency": jnp.mean(observed_efficiencies)
    }

def run_simulation(data, n_sim, v, distribution, extra_t=jnp.zeros(0), parallel=True):
    """Run simulation on all regions."""
    keys = list(data.keys())
    results = {}
    
    if parallel:
        with ProcessPoolExecutor() as executor:
            sim_results = list(executor.map(process_single_region, 
                                [(key, data[key], n_sim, v, distribution, extra_t) for key in keys]))
                                
        # Process results
        for key, processed in sim_results:
            results[key] = processed
    else:
        for key in keys:
            key, processed = process_single_region((key, data[key], n_sim, v, distribution, extra_t))
            results[key] = processed
    
    return results

@app.command()
def main(
    bed_file: str = typer.Argument(..., help="Path to BED file with lambdai values"),
    output_file: str = typer.Argument(..., help="Path to output bedGraph file"),
    
    fork_speed: float = typer.Argument(..., help="in kb/min"),
    
    n_sim: int = typer.Option(
        1000, 
        help="Number of simulations to run"
    ),
    
    distribution: str = typer.Option(
        "Exponential",
        help="Distribution to use for simulation"
    ),
    
    regions: Optional[list[str]] = typer.Option(
        None,
        "--regions",
        "-r",
        help=(
            "Regions to process (e.g., 'chr1:1000-2000'). "
            "Multiple regions can be comma-separated or passed as separate arguments."
        ),
        metavar="REGION[,REGION...]",
    ),
    
    parallel: bool = typer.Option(
        False,
        help="Run all the computation in parallel"
    ),
    
    column_spec: str = typer.Option(
        "lambdai",
        help="Column specification for the input BED file"
    ),
    
    ):

  

    """
    Command-line interface for simulating replication using the 'sim' function.
    """
    # Process regions if provided
    regions_str = None
    if regions:
        # Flatten comma-separated regions into a list
        all_regions = []
        for region in regions:
            all_regions.extend(region.split(","))
        regions_str = ",".join([validate_region(r) for r in all_regions if r])
    
    # Load data from BED file
    data, resolution = load_bed_data_regions(bed_file, regions_str, column_spec)
    
    # Calculate parameters based on input
    time_for_one_bin = resolution / fork_speed
    
    # No extra time for now
    extra_t = jnp.zeros(0)
    
    # Run simulation
    results = run_simulation(data, n_sim, fork_speed, distribution, extra_t, parallel)
    print("Done")
    
    # Scale results to match real time
   
    
    # Write results to output file
    write_custom_bedgraph_pandas(output_file, results)
    
    typer.echo(f"Simulation completed successfully. Results written to {output_file}")
    


if __name__ == "__main__":
    app()