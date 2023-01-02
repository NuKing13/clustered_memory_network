from sys import argv

write = 0

if argv[1] == "N":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_N", "r")
    write = int(input.read(5))
    write = write * 2
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_N", "w")
    input.write(str(write))
elif argv[1] == "MOD":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_MOD", "r")
    write = float(input.read(3))

    if write < 1.0:
        write = write + 0.2
    else:
        write = 0.0
    
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_MOD", "w")
    input.write(str(write))
elif argv[1] == "C":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_C", "r")
    write = float(input.read(3))

    if write < 1.0:
        write = write + 0.2
    else:
        write = 0.0
    
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/param_C", "w")
    input.write(str(write))
else:
    print("invalid write input")
    exit()

