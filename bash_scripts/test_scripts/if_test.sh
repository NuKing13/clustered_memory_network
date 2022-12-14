#!/bin/bash
#RES=$(($1 * $2))
#if [ $RES -ne 64 ]
#then
#	echo failed 
#	exit
#fi
#echo success 64

STEPSIZE_TEST=$(python -c 'from sys import argv; print(int(1)) if float(argv[1]) > 0.0 else print(int(0))' "$1")

if [ $STEPSIZE_TEST -eq 0 ]
then 
	echo stepsize cannot be 0
	exit
fi
