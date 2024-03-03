#!/bin/bash
#SBATCH --job-name="relative 100"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks=5
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 22:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
srun --mpi=pmi2 --wait=0 bash run-dynamic.shared.sh
