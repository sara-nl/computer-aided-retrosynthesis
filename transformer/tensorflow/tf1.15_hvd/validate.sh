#!/bin/bash
#Set job requirements
#SBATCH -p gpu_titanrtx
#SBATCH -t 8:00:00
#SBATCH -n 36
#SBATCH --gres=gpu:0
#SBATCH -J validate

module purge
module load 2019
module load Anaconda3/2018.12
module load cuDNN/7.6.3-CUDA-10.0.130
module load OpenMPI/3.1.4-GCC-8.3.0
module load NCCL/2.4.7-CUDA-10.0.130

VIRTENV=transformer_tf1.15

clear
module list

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI
#export HOROVOD_ALLOW_MIXED_GPU_IMPL=1

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

echo "Starting training"

source /sw/arch/Debian10/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
source activate $VIRTENV

python3 --version
mpirun -np 36 python3 transformer.py --validate data/retrosynthesis-test.smi \
       --model output/32_node_production_bs32_high_lr_20200915T002241/tr-14.h5 --beam 1 --run_name debug --temperature 1.3 \

mpirun -np 36 python3 transformer.py --validate data/retrosynthesis-test.smi \
       --model output/32_node_production_bs32_high_lr_20200915T002241/tr-14.h5 --beam 5 --run_name debug --temperature 1.3 \






