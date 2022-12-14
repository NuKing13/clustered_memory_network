#!/bin/bash
#input order: r_start r_stop r_stepsize g_start g_stop g_stepsize n

#SBATCH --job-name=r_g_scan_32
#SBATCH --output=ostendorf_r_g_scan_32_%j.out
#SBATCH --error=ostendorf_r_g_scan_32_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --mem=200000M
#SBATCH --time=04:00:00
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

NAME="r_g_scan_32"
CPUS=8

R_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$3")
G_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$6")

if [ $R_STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0 \(r\)
	exit
elif [ $G_STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0 \(g\)
	exit
fi

CURRENT_RATE=$1
R_RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")
G_RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$4" "$5" "$6")
TOTAL_RUNS=$((($R_RUNS + 1) * ($G_RUNS + 1)))

if [ $TOTAL_RUNS -ne 32 ]
then 
	echo number of combined runs does not match selected script
	echo remember that r_stop and g_stop are inclusive
	exit
fi

for (( c=0; c<=$R_RUNS; c++ ))
do
CURRENT_STRENGTH=$4

	for (( d=0; d<=$G_RUNS; d++ ))
	do
		srun --exclusive python3 lurm_run_with_user_input.py $7 $CURRENT_RATE $CURRENT_STRENGTH $CPUS $NAME & #order: n r g cpus_per_task name
		CURRENT_STRENGTH=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_STRENGTH" "$6")
	done
	
	CURRENT_RATE=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_RATE" "$3")
done 

wait
