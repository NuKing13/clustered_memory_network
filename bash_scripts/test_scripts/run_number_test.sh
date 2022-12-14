#!/bin/bash
#input order: r_start r_stop r_stepsize g_start g_stop g_stepsize n

cd ..

CURRENT_RATE=$1
R_RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$1" "$2" "$3")

G_RUNS=$(python3 -c 'from sys import argv; res=((float(argv[2]) - float(argv[1])) / float(argv[3])); print(int(res))' "$4" "$5" "$6")

echo r runs: $R_RUNS
echo g runs: $G_RUNS

OUTER_COUNTER=0
INNER_COUNTER=0
for (( c=0; c<=$R_RUNS; c++ ))
do
OUTER_COUNTER=$(($OUTER_COUNTER+1))
	for (( d=0; d<=$G_RUNS; d++ ))
	do
	INNER_COUNTER=$(($INNER_COUNTER+1))
	done
done 

echo outer loop runs: $OUTER_COUNTER
echo inner loop runs: $INNER_COUNTER
