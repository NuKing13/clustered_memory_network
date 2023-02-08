#!/bin/bash
#input order: mod_start mod_num_steps c_start c_num_steps cluster_start n raw_flag t_sim

N=$6
MOD_START=$1
C_START=$3
CLUSTER_START=$5
G=20.0
SAVE=$7
TIME=$8

if [ $N -eq 2500 ]
then 
	R=10.5
elif [ $N -eq 5000 ]
then 
	R=7.6
elif [ $N -eq 10000 ]
then 
	R=5.55
else
	echo unrecognized neuron number
	exit
fi

MOD_STOP=$(python -c 'from sys import argv; print(round(float(argv[1]) + float(argv[2]) * (float(argv[3]) - 1), 1))' "$MOD_START" "0.1" "$2")
C_STOP=$(python -c 'from sys import argv; print(round(float(argv[1]) + float(argv[2]) * (float(argv[3]) - 1), 1))' "$C_START" "0.1" "$4")
#CLUSTER_STOP=$(python -c 'from sys import argv; from math import pow; print(round(float(argv[1]) * pow(2, int(argv[2]) - 1), 0))' "$CLUSTER_START" "$6")

NAME="scan_n${N}_r${R}_g${G}_mod${MOD_START}-${MOD_STOP}_c${C_START}-${C_STOP}_n-cl${CLUSTER_START}" #-${CLUSTER_STOP}

CPUS=4

MOD_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$2")
C_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$4")
#CLUSTER_STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$6")

if [ $MOD_STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0 \(mod\)
	exit
elif [ $C_STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0 \(c\)
	exit
fi

#elif [ $CLUSTER_STEPSIZE_TEST -eq 0 ]
#then 
#	echo stepsize cannot be 0 \(cluster\)
#	exit


#TOTAL_RUNS=$(($2 * $4)) #*$6

#if [ $TOTAL_RUNS -ne 64 ]
#then 
#	echo number of combined runs does not match selected script
#	exit
#fi

MOD_RUNS=$(($2 - 1))
C_RUNS=$(($4 - 1))
#CLUSTER_RUNS=$(($6 - 1))

CURRENT_MOD=$1

for (( c=0; c<=$MOD_RUNS; c++ ))
do

CURRENT_C=$3

	for (( d=0; d<=$C_RUNS; d++ ))
	do
		CURRENT_CLUSTER=$5
		echo N: $N, R: $R, G: $G, CPUS: $CPUS, NAME: $NAME, MOD: $CURRENT_MOD, C: $CURRENT_C, SAVE: $SAVE, CLUSTERS: $CURRENT_CLUSTER, TIME: $TIME #order: n r g cpus_per_task name mod c save_raw n_cl t_sim
		
		#for (( cluster=0; cluster<=$CLUSTER_RUNS; cluster++ ))
		#do
		#	CURRENT_CLUSTER=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_CLUSTER" "$CURRENT_CLUSTER")
		#done
		
		CURRENT_C=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_C" "0.2") #0.1
	done
	
	CURRENT_MOD=$(python -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_MOD" "0.2") #0.1
done 

wait
