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
        for fit_mode in Laplace; do
            # Construct the output file name with the chromosome
            for data in Weibull_True Weibull_False Exponential_True Exponential_False; do
                for noise in 0.2 0.1 0.05; do
                    common="2500 20  --signal-to-fit theo_rfd --regions chrI"
                    output_file="input${data}_${fit_mode}-${model}_${fit_time}_${noise}_bayesian.bed"
                    bayesian synthetic/chrI-${data}.bed analysis_models/$output_file $common --fit-mode $fit_mode --noise $noise --model-type $model  $fit_time_arg  & 
                done
            done

        done
    done
done