#!/bin/bash

#PBS -l select=1:ncpus=1:mem=1gb 
#PBS -l walltime=0:02:00
#PBS -q short_cpuQ
#PBS -o res/test_all.out
#PBS -e res/test_all.err

module load mpich-3.2
cd $PBS_O_WORKDIR

# function to launch a job
generate_job(){
    exec_name="$1"
    exec_path="$2"
    i="$3"
    N="$4"
    mode="$5"
    output_dir="$6"

    ncpus=$((N / i))
    jobfile="${output_dir}/${exec_name}.sh"

    cat > "$jobfile" <<EOF
#!/bin/bash
#PBS -l select=${i}:ncpus=${ncpus}:mem=4gb -l place=pack:excl
#PBS -l walltime=1:00:00
#PBS -q short_cpuQ
#PBS -o ${output_dir}/${exec_name}.out
#PBS -e ${output_dir}/${exec_name}.err

module load mpich-3.2

cd \$PBS_O_WORKDIR

case "$mode" in
  "SEQ")
    mpiexec -n $i ${exec_path}
    ;;
  "OMP")
    mpiexec -n $i ${exec_path} $ncpus
    ;;
  "MPI")
    mpiexec -n $i ${exec_path}
    ;;
  "HYB")
    mpiexec -n $i ${exec_path} $ncpus
    ;;
esac
EOF

    chmod +x "$jobfile"
    qsub "$jobfile"
}


total_cores_list=(64)
base_dir="src/algorithms/base"
strassen_dir="src/algorithms/strassen"
bin_dir="bin"
res_dir="res"

mkdir -p "$bin_dir"
mkdir -p "$res_dir"

# Collect source files by type
sequential_scripts=( "$base_dir"/sequential_*.c "$strassen_dir"/sequential_*.c )
openmp_scripts=( "$base_dir"/omp_*.c "$strassen_dir"/omp_*.c )
mpi_scripts=( "$base_dir"/mpi_*.c "$strassen_dir"/mpi_*.c )
hybrid_scripts=( "$base_dir"/hybrid_*.c "$strassen_dir"/hybrid_*.c )

# Launch sequential (can uncomment if needed)
for script in "${sequential_scripts[@]}"; do
    base=$(basename "$script" .c)
    exec_name="${base}"
    output_path="$bin_dir/$exec_name"
    echo "$script"
    mpicc -g -Wall -o "$output_path" "$script"
    generate_job "$exec_name" "$output_path" 1 1 "SEQ" "$res_dir"
done

# Launch parallel jobs
for N in "${total_cores_list[@]}"; do
    i=1
    while [ $i -le $N ]; do
        ncpus=$((N / i))

        if [ $i -eq 1 ]; then
            for script in "${openmp_scripts[@]}"; do
                base=$(basename "$script" .c)
                exec_name="${base}_N${N}_p${i}_t${ncpus}"
                output_path="$bin_dir/$exec_name"
                mpicc -g -Wall -fopenmp -o "$output_path" "$script"
                generate_job "$exec_name" "$output_path" $i $N "OMP" "$res_dir"
            done
        elif [ $i -eq $N ]; then
            for script in "${mpi_scripts[@]}"; do
                base=$(basename "$script" .c)
                exec_name="${base}_N${N}_p${i}_t${ncpus}"
                output_path="$bin_dir/$exec_name"
                mpicc -g -Wall -o "$output_path" "$script"
                generate_job "$exec_name" "$output_path" $i $N "MPI" "$res_dir"
            done
        elif [ $i -eq 2 ]; then
            for script in "${hybrid_scripts[@]}"; do
                base=$(basename "$script" .c)
                exec_name="${base}_N${N}_p${i}_t${ncpus}"
                output_path="$bin_dir/$exec_name"
                mpicc -g -Wall -fopenmp -o "$output_path" "$script"
                generate_job "$exec_name" "$output_path" $i $N "HYB" "$res_dir"
            done
        fi
        i=$((i * 2))
    done
done

echo "All scripts launched successfully"
