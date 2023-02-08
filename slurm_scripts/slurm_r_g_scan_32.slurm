#!/bin/bash
#input order: r_start r_stepsize r_num_steps g_start g_stepsize g_num_steps n mod cmod raw_flag n_cl

#SBATCH --job-name=r_g_scan_32
#SBATCH --output=ostendorf_r_g_scan_32_%j.out
#SBATCH --error=ostendorf_r_g_scan_32_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /p/project/jinm60/users/ostendorf1/Thesis_Git/venvs/cm_nest/bin/activate

source /p/project/jinm60/users/ostendorf1/Thesis_Git/venvs/cm_nest/bin/nest_vars.sh

N=$7
R_START=$1
#R_STOP=$2 deprecated usage
R_STEP=$2
G_START=$4
#G_STOP=$5 deprecated usage
G_STEP=$5
MOD=$8
CMOD=$9
SAVE=${10}
CLUSTER=${11}

R_STOP=$(($1 + ($2 * ($3 - 1))))
G_STOP=$(($4 + ($5 * ($6 - 1))))

echo r stops at ${R_STOP}
echo g stops at ${G_STOP}

NAME="scan_n${N}_r${R_START}-${R_STOP}_step${R_STEP}_g${G_START}-${G_STOP}_step${G_STEP}_mod${MOD}_c${CMOD}"

CPUS=8

R_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$2")
G_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$5")

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

#usage deprecated
#R_RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")
#G_RUNS=$(python -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$4" "$5" "$6")
TOTAL_RUNS=$(($3 * $6))

if [ $TOTAL_RUNS -ne 32 ]
then 
	echo number of combined runs does not match selected script
	#echo remember that r_stop and g_stop are inclusive
	exit
fi

R_RUNS=$(($3 - 1))
G_RUNS=$(($6 - 1))
for (( c=0; c<=$R_RUNS; c++ ))
do
CURRENT_STRENGTH=$4

	for (( d=0; d<=$G_RUNS; d++ ))
	do
		srun --exclusive --ntasks=1 python slurm_run_with_user_input.py $N $CURRENT_RATE $CURRENT_STRENGTH $CPUS $NAME $MOD $CMOD $SAVE $CLUSTER& #order: n r g cpus_per_task name mod c
		CURRENT_STRENGTH=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_STRENGTH" "$5")
	done
	
	CURRENT_RATE=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_RATE" "$2")
done 

wait
