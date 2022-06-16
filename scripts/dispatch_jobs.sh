#!/bin/bash

# Inspired by https://github.com/y0ast/slurm-for-ml/blob/master/run_file.sh
# Expects to be in the same folder as generic.sh
# Edit this if you want more or fewer jobs in parallel
# First argument: generic script to use.
# Second argument: list of commands to run, which are used by generic.
jobs_in_parallel=48

if [ ! -f "$2" ]
then
    echo "Error: file passed does not exist"
    exit 1
fi

# This convoluted way of counting also works if a final EOL character is missing
n_lines=$(grep -c '^' "$2")

# Use file name for job name
job_name=$(basename "$2" .txt)

sbatch --array=1-${n_lines}%${jobs_in_parallel} --job-name ${job_name} "$1" "$2"