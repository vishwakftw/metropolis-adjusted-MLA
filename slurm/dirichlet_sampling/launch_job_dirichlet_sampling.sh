#!/bin/bash

# set job name
script_name="T=${2}_h=${3}_i=${4}_n=${5}"
j_name="d=${1}_${script_name}"

# set slurm logging directory
j_dir="${HOME}/slurm/dirichlet_sampling/d=${1}/"

mkdir -p ${j_dir}

rm -f ${j_dir}/*.out # cleanup before running with this config
rm -f ${j_dir}/*.err # cleanup before running with this config

# set slurm script directory
SLURM_DIR="${HOME}/sampling-barrier/slurm/"

# this is the sbatch script

echo "#!/bin/bash
#SBATCH --output=${j_dir}/${script_name}.out
#SBATCH --error=${j_dir}/${script_name}.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=normal

export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

bash ${SLURM_DIR}/dirichlet_sampling/dirichlet_sampling.sh ${1} ${2} ${3} ${4} ${5} ${6};
" > ${j_dir}/${script_name}.slrm

sbatch ${j_dir}/${script_name}.slrm
