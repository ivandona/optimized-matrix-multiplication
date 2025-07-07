#!/bin/bash
#PBS -l select=2:ncpus=32:mem=16gb
#PBS -l walltime=1:00:00
#PBS -q short_cpuQ
#PBS -o logs/outputs/mpi_strassen_o.txt
#PBS -e logs/errors/mpi_strassen_e.txt

# Load the MPI module
module load mpich-3.2

mpiexec -n 32 ./optimized-matrix-multiplication/build/bin/algorithms/strassen/mpi_strassen 1024
