#!/bin/bash

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--));
    do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

env=aoi
# path to the python interpreter in the conda environment
pydir=/nfsd/signet4/burattoale/miniconda3/envs/$env/bin


echo "Running binary exponential simulations"
# number of parallel simulations
n_paral=2
open_sem $n_paral
# main loop for starting simulations
files=(
    experiments/binary_spawc.json
    experiments/binary_spawc_loc_aware.json
)
for i in "${files[@]}"; do
    echo "Running $i"
    run_with_lock $pydir/python runner.py --config $i
done
wait
