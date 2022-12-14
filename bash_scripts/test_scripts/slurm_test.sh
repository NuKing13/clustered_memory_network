#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=ostendorf_test_%j.out
#SBATCH --error=ostendorf_test_%j.err

#SBATCH --cpus-per-task=24
#SBATCH --mem=50000M
#SBATCH --time=01:00:00
#SBATCH --exclusive #?

#SBATCH --mail-type=ALL
#SBATCH --mail-user=noah.ostendorf@rwth-aachen.de

module load Stages/2022

module load GCCcore/.9.3.0
module load Python/3.8.5
module load GCC/9.3.0
module load SciPy-Stack/2020-Python-3.8.5
module load GSL/2.6
module load ParaStationMPI/5.4.7-1
module load mpi4py/3.0.3-Python-3.8.5

source /p/project/jinm60/users/ostendorf1/cm_nest/bin/activate

source /p/project/jinm60/users/ostendorf1/Thesis_Git/nest/bin/nest_vars.sh

#input order: n r g 
srun --exclusive python run_with_user_input.py $1 $2 $3 #$SLURM_ARRAY_TASK_ID to resolve stepsize loopless but with array
