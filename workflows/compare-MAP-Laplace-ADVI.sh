#!/bin/bash

# Define the chromosomes

# Loop over each model and fit_time combination
for model in Weibull Exponential; do
    for fit_time in "True" "False"; do
        # Set the fit_time argument if needed
        fit_time_arg=""
        if [ "$fit_time" = "True" ]; then
            fit_time_arg="--fit-time"
        fi

        # Loop over each chromosome
        for fit_mode in MAP Laplace ADVI; do
            # Construct the output file name with the chromosome
            output_file="comparison/comp_${fit_mode}-${model}_${fit_time}_bayesian.bed"
            bayesian  data/from_nfs_smv11.bed "$output_file" 2500 20 --fit-mode $fit_mode --regions chrI --model-type $model ${fit_time_arg} --smoothv 19 & 

        done
    done
done