#!/bin/bash

cd ..

N=$1

cd slurm_scripts/

if [ $N -eq 2500 ]
then
	R_START=7
elif [ $N -eq 5000 ]
then
	R_START=6
elif [ $N -eq 7500 ]
then
	R_START=6
elif [ $N -eq 10000 ]
then 
	R_START=5
else 
	echo unexpected neuron number
	exit
fi

MOD=0.0
C=0.0

for (( a=0; a<=5; a++ ))
do 
	MOD=0.0
	for (( b=0; b<=5; b++ ))
	do
		R_START=5	
		for (( c=0; c<=1; c++ ))
		do
			sbatch slurm_r_g_scan_64.slurm $R_START 1 4 5 1 16 $N $MOD $C "false"
			sbatch slurm_r_g_scan_64.slurm $R_START 1 4 21 1 16 $N $MOD $C "false"
			R_START=$(($R_START + 4))
		done
		MOD=$(python -c 'from sys import argv; print(float(argv[1]) + 0.2)' "$MOD")
	done
	C=$(python -c 'from sys import argv; print(float(argv[1]) + 0.2)' "$C")
done

#outsourced to local host for synchronization
#if [ $MOD = 1.0 && $C = 1.0 ]
#then
#	N=$(( 2 * $N ))
#	MOD=0.0
#	C=0.0
#elif [ $C = 1.0 && $MOD != 1.0 ]
#then
#	MOD=$(python -c 'from sys import argv; res=(float(argv[1]) + 0.2); print(float(res))' "$MOD")
#	C=0.0
#else
#	C=$(python -c 'from sys import argv; res=(float(argv[1]) + 0.2); print(float(res))' "$C")
#fi
