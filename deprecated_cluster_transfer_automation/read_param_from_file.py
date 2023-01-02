from sys import argv

if argv[1] == "N":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_N", "r")
    N = int(input.read(5))
    print(N)    
elif argv[1] == "MOD":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_MOD", "r")
    MOD = float(input.read(3))
    print(MOD)
elif argv[1] == "C":
    input = open("/home/noah/Thesis_Git/clustered_memory_network/slurm_scripts/automation_parameters/param_C", "r")
    C = float(input.read(3))
    print(C)
else:
    print("invalid read input")
    exit()



