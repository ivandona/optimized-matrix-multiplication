#!/bin/bash
#  PBS -J 1-5
#PBS -l select=4:ncpus=4:mem=8gb
#PBS -l walltime=0:10:00
#PBS -q short_cpuQ

module load mpich-3.2
cd $PBS_O_WORKDIR

# Definisci i valori di n per ogni job
# vals=(18000 18000 18000 18000 18000)

# n=${vals[$PBS_ARRAY_INDEX-1]}  # PBS_ARRAY_INDEX parte da 1

# Esegui il programma con MPI, passando n come argomento
mpiexec -n 4 ./a.out 4 # $n
