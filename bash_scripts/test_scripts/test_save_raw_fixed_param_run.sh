#!/bin/bash
#input order: n r g 

cd ..

python slurm_run_with_user_input.py $1 $2 $3 4 test 0.0 0.2 false
