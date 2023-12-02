#!/bin/bash

# set job name
script_name="T=${4}_h=${5}_i=${6}_n=${7}"
j_name="dt=${1}_d=${2}_k=${3}_${script_name}"

# set slurm logging directory
j_dir="${HOME}/slurm/uniform_sampling/dt=${1}/d=${2}/k=${3}"

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

bash ${SLURM_DIR}/uniform_sampling/uniform_sampling.sh ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8};
" > ${j_dir}/${script_name}.slrm

sbatch ${j_dir}/${script_name}.slrm
