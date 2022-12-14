#!/bin/bash
#input order: r_start r_stop stepsize n g

cd ..

CURRENT_RATE=$1
RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")

for (( c=0; c<=$RUNS; c++ ))
do
	python3 run_with_user_input.py $4 $CURRENT_RATE $5 #order: n r g
	CURRENT_RATE=$(python3 -c 'from sys import argv; print(float(argv[1]) + float(argv[2]))' "$CURRENT_RATE" "$3")
done 
