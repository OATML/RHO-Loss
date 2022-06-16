#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

# Inspired by https://github.com/y0ast/slurm-for-ml/blob/master/generic.sh

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --time=24:00:00 # currently assuming that job terminates when finished!

set -e # fail fully on first line failure
echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# echo "Dispatching singularity exec --nv --bind ${DATA}:${DATA} $SINGULARITY_CONTAINER_PATH python3 $JOB_CMD base_outdir=$BASE_OUTDIR $IRRED_LOSS_GENERATOR_COMMAND datamodule.data_dir=$DATA_DIR"

TIME_STR=$(date '+%m-%d_%H-%M-%S')
# Train the models
FILENAME=`echo "$JOB_CMD" | grep -o "name=.*\""`
FILENAME="${FILENAME}_${TIME_STR}.txt"
srun --output ${HOME}/slurm_outputs/${FILENAME}.out singularity exec --nv --bind ${DATA}:${DATA} $SINGULARITY_CONTAINER_PATH python3 $JOB_CMD base_outdir=$BASE_OUTDIR ++$IRRED_LOSS_GENERATOR_COMMAND datamodule.data_dir=$DATA_DIR