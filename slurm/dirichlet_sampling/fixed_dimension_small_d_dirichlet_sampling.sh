#!/bin/bash

dimension=${1}
inv_dimension_onehalf=`bc -l <<< "scale=8; 1/sqrt(${dimension}^3)"`
inv_dimension_sqr=`bc -l <<< "scale=8; 1/(${dimension}^2)"`

# parameters for runs
num_particles=2000
num_iterations=2000
store_loss="yes"
num_runs=10

SLURM_DIR="${HOME}/sampling-barrier/slurm/"

d_mult_list=(${inv_dimension_onehalf} ${inv_dimension_sqr})
c_mult=0.25

for d_mult in ${d_mult_list[@]};
  do
    h=`bc -l <<< "scale=8; ${c_mult} * ${d_mult}"`
    for i in $(seq 1 ${num_runs});
    do
      while [ $(squeue -u ${USER} -h -r | wc -l) -gt 60 ];
      do
        echo "Waiting for resources to be freed up"
        sleep 30;
      done
      echo "run index=${i}";
      bash ${SLURM_DIR}/dirichlet_sampling/launch_job_dirichlet_sampling.sh ${dimension} ${num_iterations} ${h} ${i} ${num_particles} ${store_loss};
    done
  done
