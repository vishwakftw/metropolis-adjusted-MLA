#!/bin/bash

module load anaconda/2023a-pytorch

dt=${1}
d=${2}
k=${3}
T=${4}
h=${5}
i=${6}
n=${7}
store_ratio=${8}

mydir="${HOME}/sampling-barrier"
script_name="${mydir}/scripts/UniformMMRW_sampling.py"

if [ ${store_ratio} == "yes" ] ; then
    proportion_progress_file="${mydir}/outputs/proportion_progress_${dt}.txt"
else
    proportion_progress_file="NA"
fi

python ${script_name} --domain_type ${dt} \
       --dimension ${d} \
       --condition_number ${k} \
       --num_iters ${T} \
       --stepsize ${h} \
       --run_index ${i} \
       --num_particles ${n} \
       --progress_file ${mydir}/outputs/progress_${dt}.txt \
       --proportion_progress_file ${proportion_progress_file}
