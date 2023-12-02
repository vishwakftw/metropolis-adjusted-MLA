#!/bin/bash

domain_type=${1}
dimension=${2}
inv_dimension=`bc -l <<< "scale=8; 1/${dimension}"`
inv_dimension_onehalf=`bc -l <<< "scale=8; 1/sqrt(${dimension}^3)"`

# parameters for runs
num_particles=2000
num_iterations=2000
store_ratio="yes"

SLURM_DIR="${HOME}/sampling-barrier/slurm/"

if [ "${domain_type}" = "simplex" ]; then
  cond_num_list=(1)
  num_runs=20
  c_mult=0.1
else
  cond_sqr=`bc -l <<< "scale=8; ${dimension}^2 / 4"`
  cond_exp=`bc -l <<< "scale=8; e(${dimension} / 4)"`
  cond_num_list=(${cond_sqr} ${cond_exp})
  num_runs=10
  if [ "${domain_type}" = "box" ]; then
    c_mult=0.25
  else
    c_mult=0.05
  fi
fi

d_mult_list=(${inv_dimension} ${inv_dimension_onehalf})

for k in ${cond_num_list[@]};
  do
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
        echo "condition number=${k}, step size=${h}, run index=${i}";
        bash ${SLURM_DIR}/uniform_sampling/launch_job_uniform.sh ${domain_type} ${dimension} ${k} ${num_iterations} ${h} ${i} ${num_particles} ${store_ratio};
      done
    done
  done
