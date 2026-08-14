#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq

# ---------------------------------------------------------------------------
#  Aqua job script for GPU Assignment 1.
#
#  This version copies your work to scratch, RUNS THERE, and copies only the
#  results back with job ID and timestamped filenames.
# ---------------------------------------------------------------------------

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir

# Generate timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")

cp -R $PBS_O_WORKDIR/* .

module load cuda11.4
module load gcc640
module load python385

chmod +x run_tests.sh compile.sh 2>/dev/null

# ./run_tests.sh

./run_tests.sh -s CS26M004.cu

# Copy files back with job ID and timestamped names
cp -f logfile.log  $PBS_O_WORKDIR/logfile_${tpdir}_${timestamp}.log 2>/dev/null
cp -f errorfile.err $PBS_O_WORKDIR/errorfile_${tpdir}_${timestamp}.err 2>/dev/null

cd $HOME
rm -rf $tempdir
