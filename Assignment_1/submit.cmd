#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq

module load gcc/8.5.0
module load python/3.9.0

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .
./compile.sh
./run_tests.sh
cp logfile.log errorfile.err $PBS_O_WORKDIR/
rmdir $tempdir
