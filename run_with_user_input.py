from sys import argv
from math import sqrt
from main import run, PostProcessing, Inhibition, Input
from neuron import L5pyr_simp_sym
from neuron import Single_comp

#usage: python3 script_test.py n r cpus name n_cl time/steps input_mode record_mode scale_std raw_path/calc_cap
n_neurons = int (argv[1])
r_bg = float (argv[2])
cpus = int (argv[3])
name = argv[4]
inhib_strength = 20.0
weight = 0.00025 * sqrt(1250) / sqrt(n_neurons)
cluster = int(argv[5])
time = int(argv[6])

input_mode = ""
if argv[7] == "rate":
    input_mode = Input.RATE_MODULATION
else:
    input_mode = Input.CONST_RATE
    
record_mode = ""
raw_path = False

if argv[8] == "capacity":
    record_mode = PostProcessing.CAPACITY
    raw_path = True
elif argv[8] == "external":
    record_mode = PostProcessing.EXTERNAL
    raw_path = (argv[10] == "true")
else:
    record_mode = PostProcessing.NETWORK_NMDA_STATS
    
scale_std = (argv[9] == "true")
inp_str=0.1
if scale_std:
    inp_str = 0.1 * (5 / cluster)

run(
    job_name= "n5000_indegree_fixed_to_200", #"n2500_transition_rate_cl10_no_scaling", #name
    n_cores=4,
    g=inhib_strength,   #14.00  #
    r=r_bg,          #15.00  #
    w=weight,
    rho=0.05,
    n_cl=cluster,
    n_dend=5,
    n=n_neurons,        #1250   #
    mod=0.9,
    c=0.2,
    iw_fac=1.19,
    inp_type=input_mode,
    t_sim=time,
    inp_str=inp_str,
    tstep=50.0,
    steps=time,
    exc_neuron=L5pyr_simp_sym(n_dend=5),
    inh_neuron=Single_comp(),
    inhib=Inhibition.RAND_DEND,
    post_proc=record_mode,
    rec_plottrace=True,
    rec_inp=False,
    raw_path=raw_path,
    scale_std=scale_std,
)
