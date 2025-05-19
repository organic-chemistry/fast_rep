#!/bin/bash

# Define the chromosomes
chromosomes=(chrI chrII chrIII chrIV chrV chrVI chrVII chrVIII chrIX chrX chrXI chrXII chrXIII chrXIV chrXV chrXVI)

# Loop over each model and fit_time combination
for model in Laplace Exponential; do
    for fit_time in "True" "False"; do
        # Set the fit_time argument if needed
        fit_time_arg=""
        if [ "$fit_time" = "True" ]; then
            fit_time_arg="--fit-time"
        fi

        # Loop over each chromosome
        for chr in "${chromosomes[@]}"; do
            # Construct the output file name with the chromosome
            output_file="full-genome/Laplace-${model}_${fit_time}_${chr}_bayesian.bed"

            # Run the bayesian command with the current parameters
            bayesian data/from_nfs_smv11.bed "$output_file" 2500 20 \
                --fit-mode Laplace \
                --model-type "$model" \
                $fit_time_arg \
                --smoothv 19 \
                --regions "$chr" &
        done
    done
done