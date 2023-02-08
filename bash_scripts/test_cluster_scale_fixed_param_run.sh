#!/bin/bash
#input order: n r g n_cl

cd ..

python slurm_run_with_user_input.py $1 $2 $3 4 dend_test 0.0 0.2 true $4
