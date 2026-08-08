#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .
./compile.sh
./run_tests.sh
mv * $PBS_O_WORKDIR/.
rmdir $tempdir