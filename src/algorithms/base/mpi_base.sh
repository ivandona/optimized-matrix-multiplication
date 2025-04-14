#!/bin/bash
#PBS -l select=2:ncpus=32:mem=16gb
#PBS -l walltime=1:00:00
#PBS -q short_cpuQ
#PBS -J 1-12

# Load the MPI module
module load mpich-3.2

# Define the matrix sizes and process counts as arrays
matrix_sizes=(4096 8192 16384)
process_counts=(4 8 16 32)

# Compute the index for matrix size and process count
index=$((PBS_ARRAY_INDEX - 1))
matrix_index=$((index / 4))
process_index=$((index % 4))

matrix_size=${matrix_sizes[$matrix_index]}
procs=${process_counts[$process_index]}

# Run each configuration 10 times
for i in {1..10}; do
    echo "Running with matrix size $matrix_size and $procs processes (Iteration $i)"
    mpirun.actual -n $procs ./optimized-matrix-multiplication/src/algorithms/base/mpi_base $matrix_size
done
