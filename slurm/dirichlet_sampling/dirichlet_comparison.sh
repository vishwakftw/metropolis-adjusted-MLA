#!/bin/bash

module load anaconda/2023a-pytorch

d=${1}
T=${2}
hmamla=${3}
hmla=${4}
i=${5}
n=${6}
store_loss=${7}

mydir="${HOME}/sampling-barrier"
script_name="${mydir}/scripts/Dirichlet_MLA_MAMLA_comp.py"

if [ ${store_loss} == "yes" ] ; then
    loss_progress_file="${mydir}/outputs/loss_progress_dirichlet_comp.txt"
else
    loss_progress_file="NA"
fi

python ${script_name} --dimension ${d} \
       --num_iters ${T} \
       --stepsize_mamla ${hmamla} \
       --stepsize_mla ${hmla} \
       --run_index ${i} \
       --num_particles ${n} \
       --progress_file ${mydir}/outputs/progress_dirichlet_comp.txt \
       --loss_progress_file ${loss_progress_file}
