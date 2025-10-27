#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=128
#SBATCH -t 00:45:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
python3 optim.py
