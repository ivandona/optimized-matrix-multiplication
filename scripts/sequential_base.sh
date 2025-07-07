#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -q short_cpuQ
#PBS -o logs/outputs/seq_o.txt
#PBS -e logs/outputs/seq_e.txt

# Load the MPI module
module load mpich-3.2

mpirun.actual -n 1 ./optimized-matrix-multiplication/build/bin/algorithms/base/sequential_base 1024
