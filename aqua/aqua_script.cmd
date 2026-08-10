#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq

# ---------------------------------------------------------------------------
#  Aqua job script for GPU Assignment 1.
#
#  This version copies your work to scratch, RUNS THERE, and copies only the
#  results back -- it never tries to move the inputs back onto themselves.
# ---------------------------------------------------------------------------

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir

cp -R $PBS_O_WORKDIR/* .

module load cuda11.4
module load gcc640
module load python385

chmod +x run_tests.sh compile.sh 2>/dev/null

./run_tests.sh

cp -f logfile.log  $PBS_O_WORKDIR/ 2>/dev/null
cp -f errorfile.err $PBS_O_WORKDIR/ 2>/dev/null

cd $HOME
rm -rf $tempdir
