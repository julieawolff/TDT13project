#!/bin/sh
#SBATCH --account=share-ie-idi  
#SBATCH --job-name=late_fusion

#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00

#SBATCH --output=output_late_fusion.txt  
#SBATCH --error=output_late_fusion.err  

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
    --execute C2_late_fusion.ipynb \
    --output C2_late_fusion_executed.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=tdt13idun