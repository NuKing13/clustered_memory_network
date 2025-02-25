#!/bin/bash
#input order: r_start r_stop stepsize n g

#SBATCH --job-name=r_scan_64
#SBATCH --output=ostendorf_r_scan_64_%j.out
#SBATCH --error=ostendorf_r_scan_64_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=64
#SBATCH --cpus-per-task=4
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

NAME="r_scan_64"
CPUS=4

STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$3")

if [ $STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0
	exit
fi

CURRENT_RATE=$1
RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")
TOTAL_RUNS=$(($RUNS + 1))

if [ $TOTAL_RUNS -ne 64 ]
then 
	echo number of runs does not match selected script
	echo remember that r_stop is inclusive
	exit
fi

for (( c=0; c<=$RUNS; c++ ))
do
	srun --exclusive python slurm_run_with_user_input.py $4 $CURRENT_RATE $5 $CPUS $NAME & #order: n r g cpus_per_task name 
	CURRENT_RATE=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_RATE" "$3")
done 

wait
