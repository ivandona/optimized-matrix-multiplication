#!/bin/bash

#PBS -l select=1:ncpus=1:mem=1gb 
#PBS -l walltime=0:02:00
#PBS -q short_cpuQ
#PBS -o res/test_reverse.out
#PBS -e res/test_reverse.err

module load mpich-3.2
cd $PBS_O_WORKDIR

# function to launch a job
generate_job(){
    exec_name="$1"
    exec_path="$2"
    processes="$3"
    total_cores="$4"
    mode="$5"
    output_dir="$6"
    nodes="$7"

    ncpus=$((total_cores / processes))
    jobfile="${output_dir}/${exec_name}.sh"


    cat > "$jobfile" <<EOF
#!/bin/bash
#PBS -l select=${nodes}:ncpus=${ncpus}:mem=4gb -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q short_cpuQ
#PBS -o ${output_dir}/${exec_name}.out
#PBS -e ${output_dir}/${exec_name}.err

module load mpich-3.2

cd \$PBS_O_WORKDIR

case "$mode" in
  "SEQ")
    mpiexec -n $processes ${exec_path}
    ;;
  "OMP")
    mpiexec -n $processes ${exec_path} $ncpus $nodes
    ;;
  "MPI")
    mpiexec -n $processes ${exec_path} $nodes
    ;;
  "HYB")
    mpiexec -n $processes ${exec_path} $ncpus $nodes
    ;;
esac
EOF

  chmod +x "$jobfile"
  qsub "$jobfile"
}

total_cores=256
total_nodes_list=(4) #4 8 16 32 64
max_processes=64
src_dir="src"
bin_dir="bin"
res_dir="res"

# Create folders
mkdir -p "$bin_dir"
mkdir -p "$res_dir"

# load src files
sequential_sripts=( "$src_dir"/*sequential.c )
openmp_scripts=( "$src_dir"/*reverse_OpenMP.c )
mpi_scripts=( "$src_dir"/*reverse_MPI.c )
hybrid_scripts=( "$src_dir"/*reverse_Hybrid.c )

# launch parallel
for N in "${total_nodes_list[@]}"; do
    mkdir -p "$res_dir/$N"
    i=$N
    while [ $i -le $max_processes ]; do
        ncpus=$((total_cores / i))
        for script in "${hybrid_scripts[@]}"; do
            base=$(basename "$script" .c)
            exec_name="${base}_cores${total_cores}_nodes${N}_p${i}_t${ncpus}"
            output_path="$bin_dir/$exec_name"
            mpicc -g -Wall -fopenmp -o "$output_path" "$script"
            generate_job "$exec_name" "$output_path" $i $total_cores "HYB" "$res_dir" $N
        done
        i=$((i * 2))
    done
done
echo "All scripts launched succesfully"
