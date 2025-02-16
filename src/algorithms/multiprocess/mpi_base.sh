#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb
#PBS -l walltime=0:05:00
#PBS -q short_cpuQ
#PBS -J 1-5

# Load the MPI module
module load mpich-3.2

# Define the matrix sizes as an array
matrix_sizes=(1024 2048 4096 8192 16384)

# Get the matrix size for this job array task
matrix_size=${matrix_sizes[$PBS_ARRAY_INDEX - 1]}

# Run the MPI program with the specified matrix size
mpirun.actual -n 4 ./optimized-matrix-multiplication/src/algorithms/multiprocess/mpi_base $matrix_size