#!/bin/bash
#input order: n r g cpus

#SBATCH --job-name=n_r_g_single_test_c128
#SBATCH --output=ostendorf_n_r_g_single_test_c128_%j.out
#SBATCH --error=ostendorf_n_r_g_single_test_c128_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --partition=dc-cpu

#SBATCH --mail-type=ALL
#SBATCH --mail-user=noah.ostendorf@rwth-aachen.de

cd ..

module load Stages/2022

module load GCC
module load Python
module load SciPy-Stack
module load GSL

#module load GCCcore
#module load ParaStationMPI
#module load mpi4py

source /p/project/jinm60/users/ostendorf1/Thesis_Git/venvs/cm_nest/bin/activate

source /p/project/jinm60/users/ostendorf1/Thesis_Git/venvs/cm_nest/bin/nest_vars.sh

NAME="n_r_g_single_test"

srun --exclusive python slurm_run_with_user_input.py $1 $2 $3 $4 $NAME & #order: n r g cpus_per_task name
