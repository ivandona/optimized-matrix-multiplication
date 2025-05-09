#!/bin/bash
#PBS -l select=1:ncpus=4:mem=1gb
#PBS -l walltime=00:30:00
#PBS -q short_cpuQ

./optimized-matrix-multiplication/src/algorithms/base/omp_base 1024 4