#!/bin/bash
#SBATCH --job-name="umudov_alt3 8"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:4
#SBATCH --ntasks=5
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aumudov@uci.edu
#SBATCH -t 1:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
srun --mpi=pmi2 --wait=0 bash run-dynamic.shared.sh
