#!/bin/bash
#PBS -N gpu_a4
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:20:00
#PBS -e errorfile.txt
#PBS -o logfile.txt
#PBS -q gpuq

# Work in a scratch directory of our own, so that two jobs never share one
tempdir=$HOME/scratch/job_$PBS_JOBID
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

# Use CUDA 12.8 (there is no module for it, so set the paths directly)
export PATH=/lfs/sware/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/lfs/sware/cuda-12.8/lib64:$LD_LIBRARY_PATH
nvcc -arch=sm_70 harness.cu wildfire.cu -o main

# Run the test script
./run_testcases.sh

# Copy generated files back to the submission directory
cp -r * $PBS_O_WORKDIR/
rm -r $tempdir

# Print job details
echo "Job completed at: $(date)"
echo "Executed on node: $(hostname)"
