#!/bin/bash
#SBATCH --job-name=ostendorf_test


#SBATCH --cpus-per-task=24
#SBATCH --mem=50000M
#SBATCH --time=00:05:00
#SBATCH --output=ostendorf_test_%j.out 
#SBATCH --error=ostendorf_test_%j.err
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=noah.ostendorfrwth-aachen.de

srun python3 slurm_test2.sh $1
