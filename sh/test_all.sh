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


total_cores_list=(64) #64 128? 256?
src_dir="src"
bin_dir="bin"
res_dir="res"

# Create folders
mkdir -p "$bin_dir"
mkdir -p "$res_dir"

# load src files
sequential_sripts=( "$src_dir"/*sequential.c )
openmp_scripts=( "$src_dir"/*OpenMP.c )
mpi_scripts=( "$src_dir"/*MPI.c )
hybrid_scripts=( "$src_dir"/COO*reverse_Hybrid.c )

# launch sequential (need a larger wall time)
for script in "${sequential_sripts[@]}"; do
    base=$(basename "$script" .c)
    exec_name="${base}"
    output_path="$bin_dir/$exec_name"
    #mpicc -g -Wall -o "$output_path" "$script"
    #generate_job "$exec_name" "$output_path" 1 1 "SEQ" "$res_dir"
done

# launch parallel
for N in "${total_cores_list[@]}"; do
    i=1
    while [ $i -le $N ]; do
        ncpus=$((N / i))
        if [ $i -eq 1 ]; then
            for script in "${openmp_scripts[@]}"; do
                base=$(basename "$script" .c)
                exec_name="${base}_N${N}_p${i}_t${ncpus}"
                output_path="$bin_dir/$exec_name"
                #mpicc -g -Wall -fopenmp -o "$output_path" "$script"
                #generate_job "$exec_name" "$output_path" $i $N "OMP" "$res_dir"
            done
        elif [ $i -eq $N ]; then
            for script in "${mpi_scripts[@]}"; do
                base=$(basename "$script" .c)
                exec_name="${base}_N${N}_p${i}_t${ncpus}"
                output_path="$bin_dir/$exec_name"
                #mpicc -g -Wall -o "$output_path" "$script"
                #generate_job "$exec_name" "$output_path" $i $N "MPI" "$res_dir"
            done
        #else
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
echo "All scripts launched succesfully"
