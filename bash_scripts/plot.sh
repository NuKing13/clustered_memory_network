#!/bin/bash
#input order: mode

cd ..

if [ "$1" = "single" ]
then 
	PLOT_PATH=""
	cd output
	pwd
	ls
	echo Is this the directory to plot from? y/n
	read YN
	while [ "$YN" = "n" ]
	do
		echo please select subdirectory to plot from:
		read -e DIRECTORY
		cd $DIRECTORY
		PLOT_PATH="$PLOT_PATH$DIRECTORY"
		ls
		pwd
		echo Is this the directory to plot from? y/n
		read YN
	done
	
	cd raw
	ls
	
	echo please specifiy neuron number: 
	read N
	
	echo please specifiy rate:
	read R
	
	echo please specifiy inhibitory strength:
	read G
	
	echo please specifiy modularity mod:
	read MOD
	
	echo please specifiy modularity c:
	read C
	
	echo please specifiy input strength:
	read STR
	
	#MOD=0.0
	#C=0.2
	#STR=0.1
	STEP=50.0
	WEIGHT=$(python -c 'from sys import argv; from math import sqrt; print(0.00025 * sqrt(1250) / sqrt(int(argv[1])))' "$N")
	#R=$3
	#G=$4

	RAW="raw_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.p"
	RES="_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.json"
	
	cd /home/noah/Thesis_Git/clustered_memory_network

	python plot.py rasta_nmda $PLOT_PATH $RAW $RES
	
elif [ "$1" = "map" ]
then
	cd output/jureca_scans
	ls
	echo -n "Select directory to plot:"
	read -e DIRECTORY
	cd ../..
	
	if [ $# -eq 2 ]
	then 
		PLOT_MOD="mod_c"
	else
		PLOT_MOD="r-bg_g"
	fi
	
	python plot.py heatmap $DIRECTORY $PLOT_MOD #usage: name_of_src_folder plot_mod_c/r-bg_g
	
elif [ "$1" = "transition_rate" ]
then
	echo please input network size
	read -e N
	
	echo "plotting transiton rate graphs for network size: $N"
	echo "fixed g = 20, mod = 0.0, c = 0.2"
	echo "plotting ..."
	
	python plot.py transition_rate $N #usage: n

elif [ "$1" = "traces" ]
then 
	PLOT_PATH=""
	cd output
	pwd
	ls
	echo Is this the directory to plot from? y/n
	read YN
	while [ "$YN" = "n" ]
	do
		echo please select subdirectory to plot from:
		read -e DIRECTORY
		cd $DIRECTORY
		PLOT_PATH="$PLOT_PATH$DIRECTORY"
		ls
		pwd
		echo Is this the directory to plot from? y/n
		read YN
	done
	
	cd raw
	ls
	
	echo please specifiy neuron number: 
	read N
	
	echo please specifiy rate:
	read R
	
	echo please specifiy inhibitory strength:
	read G
	
	echo please specifiy modularity mod:
	read MOD
	
	echo please specifiy modularity c:
	read C
	
	echo please specifiy input strength:
	read STR
	
	STEP=50.0
	WEIGHT=$(python -c 'from sys import argv; from math import sqrt; print(0.00025 * sqrt(1250) / sqrt(int(argv[1])))' "$N")
	
	RAW="raw_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.p"
	
	cd /home/noah/Thesis_Git/clustered_memory_network

	python plot.py nmda_traces $PLOT_PATH $RAW 

elif [ "$1" = "nmda_and_traces" ]
then
	PLOT_PATH=""
	cd output
	pwd
	ls
	echo Is this the directory to plot from? y/n
	read YN
	while [ "$YN" = "n" ]
	do
		echo please select subdirectory to plot from:
		read -e DIRECTORY
		cd $DIRECTORY
		PLOT_PATH="$PLOT_PATH$DIRECTORY"
		ls
		pwd
		echo Is this the directory to plot from? y/n
		read YN
	done
	
	cd raw
	ls
	
	echo please specifiy neuron number: 
	read N
	
	echo please specifiy rate:
	read R
	
	echo please specifiy inhibitory strength:
	read G
	
	echo please specifiy modularity mod:
	read MOD
	
	echo please specifiy modularity c:
	read C
	
	echo please specifiy input strength:
	read STR
	
	STEP=50.0
	WEIGHT=$(python -c 'from sys import argv; from math import sqrt; print(0.00025 * sqrt(1250) / sqrt(int(argv[1])))' "$N")

	RAW="raw_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.p"
	RES="_r${R}_g${G}_mod${MOD}_c${C}_str${STR}_step${STEP}_weight${WEIGHT}.json"
	
	cd /home/noah/Thesis_Git/clustered_memory_network
	 
	python plot.py rasta_nmda_plus_traces $PLOT_PATH $RAW $RES

elif [ "$1" = "map4d" ]
then
	echo please implement
else 
	echo unrecognized mode
	exit
fi
