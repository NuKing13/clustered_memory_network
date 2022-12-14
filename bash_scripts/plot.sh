#!/bin/bash
#input order: mode + n r g/none

cd ..

if [ "$1" = "single" ]
then 
	MOD=0.0
	C=0.2
	STR=0.1
	STEP=50.0
	WEIGHT=$(python3 -c 'from sys import argv; from math import sqrt; print(0.00025 * sqrt(1250) / sqrt(int(argv[1])))' "$2")
	R=$3
	G=$4

	RAW="raw_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.p"
	RES="_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.json"

	python3 plot.py $RAW $RES
elif [ "$1" = "map" ]
then
	cd output/jureca_scans
	ls
	echo -n "Select directory to plot:"
	read -e DIRECTORY
	cd ../..
	python3 heatmap.py $DIRECTORY #usage: name_of_src_folder
else 
	echo unrecognized mode
	exit
fi
