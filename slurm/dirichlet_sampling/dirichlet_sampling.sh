#!/bin/bash

module load anaconda/2023a-pytorch

d=${1}
T=${2}
h=${3}
i=${4}
n=${5}
store_loss=${6}

mydir="${HOME}/sampling-barrier"
script_name="${mydir}/scripts/DirichletMAMLA_sampling.py"

if [ ${store_loss} == "yes" ] ; then
    loss_progress_file="${mydir}/outputs/loss_progress_dirichlet.txt"
else
    loss_progress_file="NA"
fi

python ${script_name} --dimension ${d} \
       --num_iters ${T} \
       --stepsize ${h} \
       --run_index ${i} \
       --num_particles ${n} \
       --progress_file ${mydir}/outputs/progress_dirichlet.txt \
       --loss_progress_file ${loss_progress_file}
