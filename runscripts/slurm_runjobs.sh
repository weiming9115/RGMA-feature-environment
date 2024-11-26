#!/bin/bash

#SBATCH -J slurm_mcstrack_env
#SBATCH -A m4374
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

python run_feature_enviroment.py 1950
