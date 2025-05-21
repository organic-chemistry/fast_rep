#!/bin/bash


# Loop over each model and fit_time combination
for model in Weibull Exponential; do
    for fit_time in "True" "False"; do
        # Set the fit_time argument if needed
        fit_time_arg=""
        if [ "$fit_time" = "True" ]; then
            fit_time_arg="--fit-time"
        fi
        # Loop over each chromosome
            # Construct the output file name with the chromosome
        output_file="synthetic/chrI-${model}_${fit_time}.bed"
        fit_rfd_ori data/from_nfs_smv11.bed $output_file 2500 20 --regions chrI --fit-mode MAP --model-type $model $fit_time_arg

    done
done