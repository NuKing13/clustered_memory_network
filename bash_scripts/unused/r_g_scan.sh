#!/bin/bash
#input order: r_start r_stop r_stepsize g_start g_stop g_stepsize n

cd ..

CURRENT_RATE=$1
R_RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")

G_RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$4" "$5" "$6")

for (( c=0; c<=$R_RUNS; c++ ))
do
CURRENT_STRENGTH=$4

	for (( d=0; d<=$G_RUNS; d++ ))
	do
		python3 run_with_user_input.py $7 $CURRENT_RATE $CURRENT_STRENGTH #order: n r g
		CURRENT_STRENGTH=$(python3 -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_STRENGTH" "$6")
	done
	
	CURRENT_RATE=$(python3 -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_RATE" "$3")
done 
