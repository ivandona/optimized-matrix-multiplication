#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -q short_cpuQ
#PBS -o logs/outputs/seq_stras_o.txt
#PBS -e logs/outputs/seq_stras_e.txt

# Load the MPI module
module load mpich-3.2

mpiexec -n 1 ./optimized-matrix-multiplication/build/bin/algorithms/strassen/sequential_strassen 8

