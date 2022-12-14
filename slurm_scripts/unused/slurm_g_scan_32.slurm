#!/bin/bash
#input order: g_start g_stop stepsize n r

#SBATCH --job-name=g_scan_32
#SBATCH --output=ostendorf_g_scan_32_%j.out
#SBATCH --error=ostendorf_g_scan_32_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000M
#SBATCH --time=02:00:00
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

NAME="g_scan_32"
CPUS=8

STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$3")

if [ $STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0
	exit
fi

CURRENT_STRENGTH=$1
RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")
TOTAL_RUNS=$(($RUNS + 1))

if [ $TOTAL_RUNS -ne 32 ]
then 
	echo number of runs does not match selected script
	echo remember that g_stop is inclusive
	exit
fi

for (( c=0; c<=$RUNS; c++ ))
do
	srun --exclusive python slurm_run_with_user_input.py $4 $5 $CURRENT_STRENGTH $CPUS $NAME & #order: n r g cpus_per_task name
	CURRENT_STRENGTH=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_STRENGTH" "$3")
done 

wait
