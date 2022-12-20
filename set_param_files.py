from sys import argv

write = int(argv[1])
input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_N", "w")
input.write(write)

write = float(argv[2])
input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_MOD", "w")
input.write(write)

write = float(argv[3])
input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_C", "w")
input.write(write)




