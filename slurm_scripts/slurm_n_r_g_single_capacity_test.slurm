#!/bin/bash
#input order: n r n_cl time/steps input_mode record_mode

#SBATCH --job-name=n_r_g_single
#SBATCH --output=ostendorf_n_r_g_single_test_c128_%j.out
#SBATCH --error=ostendorf_n_r_g_single_test_c128_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=00:30:00
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

NAME="single_n${1}_r${2}_g20.0_mod0.0_c0.2_cluster${3}_capacity_test"

srun --exclusive --ntasks=1 python run_with_user_input.py $1 $2 128 $NAME $3 $4 $5 $6 #order: n r cpus name n_cl time/steps input_mode record_mode
