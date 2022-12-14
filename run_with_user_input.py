from sys import argv
from math import sqrt
from main import run, PostProcessing, Inhibition, Input
from neuron import L5pyr_simp_sym
from neuron import Single_comp

#usage: python3 script_test.py n r g (name)
n_neurons = int (argv[1])
bg_rate = float (argv[2])
inhib_strength = float (argv[3])
weight = 0.00025 * sqrt(1250) / sqrt(n_neurons)
#name = argv[4]

run(
    job_name="test", #name
    n_cores=4,
    g=inhib_strength,   #14.00  #
    r=bg_rate,          #15.00  #
    w=weight,
    rho=0.1,
    n_cl=5,
    n_dend=5,
    n=n_neurons,        #1250   #
    mod=0.0,
    c=0.2,
    iw_fac=1.19,
    inp_type=Input.CONST_RATE,
    t_sim=1000,
    inp_str=0.1,
    tstep=50.0,
    steps=100,
    exc_neuron=L5pyr_simp_sym(),
    inh_neuron=Single_comp(),
    inhib=Inhibition.RAND_DEND,
    post_proc=PostProcessing.NETWORK_NMDA_STATS,
    rec_plottrace=False,
    rec_inp=False,
)
