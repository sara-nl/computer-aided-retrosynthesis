module purge
module load 2019
module load Anaconda3/2018.12
module load Python/3.6.6-foss-2019b
module load cuDNN/7.6.3-CUDA-10.0.130
module load OpenMPI/3.1.4-GCC-8.3.0
module load NCCL/2.4.7-CUDA-10.0.130

VIRTENV=transformer_tf1.15

clear

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

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  yes | rm -r ~/.conda/envs/$VIRTENV
  conda create -y -n $VIRTENV python=3.6.10
  conda activate $VIRTENV
  conda install -y -c conda-forge --name $VIRTENV rdkit
  pip install --upgrade pip --no-cache-dir
  pip3 install tensorflow-gpu==1.15.3 --no-cache-dir
  pip3 install matplotlib --no-cache-dir
  pip3 install graphviz --no-cache-dir
  pip3 install pydot --no-cache-dir
  pip3 install gast==0.2.2 --no-cache-dir
  pip3 install horovod==0.19.5 --no-cache-dir
fi

# source /sw/arch/Debian10/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh

conda activate $VIRTENV

