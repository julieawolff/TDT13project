#!/bin/sh
#SBATCH --account=share-ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=subreddit_baseline
#SBATCH --time=06:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output_subreddit_baseline.txt      # Specifying 'stdout'
#SBATCH --error=output_subreddit_baseline.err        # Specifying 'stderr'

#SBATCH --mail-user=julieawo@stud.ntnu.no
#SBATCH --mail-type=ALL


module purge

eval "$(conda shell.bash hook)"
conda activate tdt13idun

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"



python -m jupyter nbconvert --to notebook \
    --execute 03_subreddit_baseline.ipynb \
    --output 03_subreddit_baseline_executed.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=tdt13idun