#!/bin/bash
R_START=10.0
for (( c=0; c<=20; c++ ))
do
	echo $R_START
        R_START=$(python -c 'from sys import argv; print(round((float(argv[1]) + float(argv[2])), 2))' "$R_START" "0.05")
        
done
