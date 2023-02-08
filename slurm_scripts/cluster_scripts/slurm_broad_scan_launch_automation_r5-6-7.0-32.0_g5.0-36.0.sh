#!/bin/bash

N=$1
if [ $# -eq 3 ]
then
	MOD=$2
	C=$3
elif [ $# -eq 1 ]
then
	MOD=0.0
	C=0.2
else
	echo unexpected number of arguments
	exit
fi


if [ $N -eq 2500 ]
then
	R_START=7
elif [ $N -eq 5000 ]
then
	R_START=6
elif [ $N -eq 10000 ]
then 
	R_START=5
else 
	echo unexpected neuron number
	exit
fi


for (( c=0; c<=13; c++ ))
do
	sbatch slurm_r_g_scan_32.slurm $R_START 1 2 5 1 16 $N $MOD $C
	sbatch slurm_r_g_scan_32.slurm $R_START 1 2 21 1 16 $N $MOD $C
	R_START=$(($R_START + 2))
done

