#!/bin/bash
#input order: n mod c, running on input rate modulation

#SBATCH --job-name=r_scan_16
#SBATCH --output=ostendorf_r_scan_16_%j.out
#SBATCH --error=ostendorf_r_scan_16_%j.err

#SBATCH --nodes=1
#SBACTH --ntasks=11
#SBATCH --cpus-per-task=23
#SBATCH --mem=512G
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

source /p/project/jinm60/users/ostendorf1/Thesis_Git/venvs/cm_nest/bin/nest_vars.shrs.sh

N=$1
MOD=$2
C=$3

if [ $N -eq 2500 ]
then 
	CURRENT_RATE=10.0
	CLUSTERS=10
elif [ $N -eq 5000 ]
then
	CURRENT_RATE=7.0
	CLUSTERS=20
elif [ $N -eq 10000 ]
then
	CURRENT_RATE=5.0
	CLUSTERS=40
fi

NAME="n${N}_g20.0_mod${MOD}_c${C}_transition_rate_scan"
CPUS=23

for (( c=0; c<=10; c++ ))
do
	echo $N $CURRENT_RATE $CPUS $NAME $MOD $C true $CLUSTERS 100 nmda true false
	#srun --exclusive --ntasks=1 python slurm_run_with_user_input_rate_modulation.py $N $CURRENT_RATE $CPUS $NAME $MOD $C true $CLUSTERS 100 nmda true false & #order: n r g cpus_per_task name mod
																					 # c save_raw n_cl time_steps
																					 # post_proc_mode scale_std
																					 # raw_path/calc_cap
	CURRENT_RATE=$(python -c 'from sys import argv; print(round(float(argv[1]) + 0.1, 2))' "$CURRENT_RATE")
done 

wait

