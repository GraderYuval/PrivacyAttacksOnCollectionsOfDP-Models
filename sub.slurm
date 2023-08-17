#! /bin/sh

#SBATCH --job-name=sub
#SBATCH --output=sub_change.out # redirect stdout
#SBATCH --error=sub_change.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python LiRA_experiment.py