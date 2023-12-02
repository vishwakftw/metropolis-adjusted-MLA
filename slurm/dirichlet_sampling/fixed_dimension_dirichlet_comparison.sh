#!/bin/bash

dimension=${1}
stepsize_mamla=`bc -l <<< "scale=8; 0.25/sqrt(${dimension}^3)"`
stepsize_mla=`bc -l <<< "scale=8; 0.002/${dimension}"`

# parameters for runs
num_particles=2000
num_iterations=1000
store_loss="yes"
num_runs=10

SLURM_DIR="${HOME}/sampling-barrier/slurm/"

for i in $(seq 1 ${num_runs});
do
  while [ $(squeue -u ${USER} -h -r | wc -l) -gt 60 ];
  do
    echo "Waiting for resources to be freed up"
    sleep 30;
  done
  echo "run index=${i}";
  bash ${SLURM_DIR}/dirichlet_sampling/launch_job_dirichlet_comparison.sh ${dimension} ${num_iterations} ${stepsize_mamla} ${stepsize_mla} ${i} ${num_particles} ${store_loss};
done
