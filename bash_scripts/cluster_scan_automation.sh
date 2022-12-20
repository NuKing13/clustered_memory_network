#!/bin/bash
#input: n_start n_end mod_start c_start
#loops over n, mod and c (step size 2500/0.2 respectively) and fully scans neurons size up to and including n_end

#set n mod c local
N=$1
MOD=$3
C=$5

cd ..

set_param_files.py $N $MOD $C
rsync -a /home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters ostendorf1@jureca.fz-juelich.de:/p/project/jinm60/users/ostendorf1/Thesis_Git/clustered_memory_network/slurm_scripts/

while [ $N -le $2 ]
do
	#update n mod c on remote host
	set_param_files.py $N $MOD $C
	rsync -a /home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters ostendorf1@jureca.fz-juelich.de:/p/project/jinm60/users/ostendorf1/Thesis_Git/clustered_memory_network/slurm_scripts/
	
	ssh start_job
	
	#wait for all jobs to complete
	TEST_COMPLETION=$(ssh test_squeue | grep ostendorf1)
	while [ "$TEST_COMPLETION" != "" ]
	do 
		sleep 5m
		TEST_COMPLETION=$(ssh test_squeue | grep ostendorf1)
	done
	
	cd bash_scripts/
	
	#transfer results and store them properly on local machine
	transfer_results_and_cleanup.sh $N $MOD $C
	
	cd ..
	
	ssh clean_up
		
	#update n mod c on local host in preperation of next scan step
	if [ $MOD = 1.0 && $C = 1.0 ]
	then
		python update_param_file.py N
		python update_param_file.py MOD
		python update_param_file.py C
	elif [ $C = 1.0 && $MOD != 1.0 ]
	then
		python update_param_file.py MOD
		python update_param_file.py C
	else
		python update_param_file.py C
	fi
	
	#give possibility for early interrupt
	echo "Finished scan."
	echo "Parameters: N - ${N}; MOD - ${MOD}; C - ${C}"
	echo "Enter \"exit\" to quit now"
	read -t 30 BREAK
	
	if [ "${BREAK}" = "exit" ]
	then
		break
	fi
done
