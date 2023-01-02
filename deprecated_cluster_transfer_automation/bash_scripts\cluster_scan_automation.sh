#!/bin/bash
#input: n_start n_end mod_start c_start
#loops over n, mod and c (step size 2500/0.2 respectively) and fully scans neurons size up to and including n_end

#set n mod c local
N=$1
MOD=$3
C=$4

cd ..

python set_param_files.py $N $MOD $C
rsync -a /home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters ostendorf1@jureca.fz-juelich.de:/p/project/jinm60/users/ostendorf1/Thesis_Git/clustered_memory_network/slurm_scripts/

while [ $N -le $2 ]
do
	#update n mod c on local and remote host
	N=$(python read_param_from_file.py N)
	MOD=$(python read_param_from_file.py MOD)
	C=$(python read_param_from_file.py C)
	python set_param_files.py $N $MOD $C
	rsync -a /home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters ostendorf1@jureca.fz-juelich.de:/p/project/jinm60/users/ostendorf1/Thesis_Git/clustered_memory_network/slurm_scripts/
	
	echo "Starting new scan."
	ssh start_job
	echo "Jobs queued."
	
	#wait for all jobs to complete
	TEST_COMPLETION=$(ssh test_squeue | grep ostendor)
	while [ "$TEST_COMPLETION" != "" ]
	do 
		echo "Jobs are still queued." 
		echo "Going to sleep."
		sleep 5m
		echo "Testing empty queue again."
		TEST_COMPLETION=$(ssh test_squeue | grep ostendor)
	done
	
	cd bash_scripts/
	
	#transfer results and store them properly on local machine
	echo "Transferring results."
	./transfer_results_and_cleanup.sh $N $MOD $C
	echo "Transfer complete."
	
	cd ..
	
	echo "Cleaning up cluster."
	ssh clean_up
	echo "Clean up complete."
		
	#update n mod c on local host in preperation of next scan step
	echo "Updating param files."
	if [[ "${MOD}" = "1.0" ]] && [[ "${C}" = "1.0" ]]
	then
		python update_param_file.py N
		python update_param_file.py MOD
		python update_param_file.py C
	elif [[ "${MOD}" = "1.0" ]] && [[ "${C}" != "1.0" ]]
	then
		python update_param_file.py MOD
		python update_param_file.py C
	else
		python update_param_file.py MOD
	fi
	
	#give possibility for early interrupt
	echo "Finished scan."
	echo "Parameters: N - ${N}; MOD - ${MOD}; C - ${C}"
	echo "Enter \"exit\" to quit now"
	read -t 15 BREAK
	
	if [ "${BREAK}" = "exit" ]
	then
		break
	fi
done
