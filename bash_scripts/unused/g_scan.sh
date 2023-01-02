#!/bin/bash
#input order: g_start g_stop stepsize n r

cd ..

CURRENT_STRENGTH=$1
RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")

for (( c=0; c<=$RUNS; c++ ))
do
	python3 run_with_user_input.py $4 $5 $CURRENT_STRENGTH #order: n r g
	CURRENT_STRENGTH=$(python3 -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_STRENGTH" "$3")
done 
