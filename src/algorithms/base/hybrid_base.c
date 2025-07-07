#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../../../include/utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int rank, comm_sz, nthreads;
    int rows_per_process, matrix_size;
    double sum;
    double **A, **B, **C, **strip_A, **strip_C;
    double *A_data, *B_data, *C_data, *strip_A_data, *strip_C_data;
    double start_time, end_time, start_tot, end_tot, total_time;
    double cpu_time_generation, cpu_time_computation, cpu_time_multiplying;
    double comms_time_total, comms_time_dist, comms_time_aggr;
    MPI_Datatype matrix, strip;

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);

    if (rank == 0) {
        start_tot = MPI_Wtime();
    }

    srand(1);

    /* get matrix size */
    matrix_size = atoi(argv[1]);
    nthreads = atoi(argv[2]);

    /* calculate the strip size */
    rows_per_process = matrix_size / comm_sz;

    /* defining an MPI datatype for sub-matrix */
    MPI_Type_vector(rows_per_process, matrix_size, matrix_size, MPI_INT, &strip);
    MPI_Type_commit(&strip);

    /* defining an MPI datatype for matrix */
    MPI_Type_vector(matrix_size, matrix_size, matrix_size, MPI_INT, &matrix);
    MPI_Type_commit(&matrix);

    if(rank == 0) {
        start_time = MPI_Wtime();
        /* allocate and fill matrix A */
        alloc_matrix(&A, &A_data, matrix_size, matrix_size);
        generate_random_matrix(A, matrix_size, 0, 100);

        /* allocate matrix C (results matrix) */
        alloc_matrix(&C, &C_data, matrix_size, matrix_size);
    }
    
    /* allocate and fill matrix B */
    alloc_matrix(&B, &B_data, matrix_size, matrix_size);
    if(rank == 0) {
        generate_random_matrix(B, matrix_size, 0, 100);
        end_time = MPI_Wtime();
        cpu_time_generation = end_time - start_time;
    }

    /* allocate sub-matrices */
    alloc_matrix(&strip_A, &strip_A_data, rows_per_process, matrix_size);
    alloc_matrix(&strip_C, &strip_C_data, rows_per_process, matrix_size);
    init_matrix(strip_C, rows_per_process, matrix_size);

    if (rank == 0)
        start_time = MPI_Wtime();
    /* scatter matrix A to all processes */
    MPI_Scatter(A_data, 1, strip, &(strip_A[0][0]), 1, strip, 0, MPI_COMM_WORLD);

    // broadcast matrix B to all processes
    MPI_Bcast(B_data, 1, matrix, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        comms_data_distribution = end_time - start_time;

        // Start time for matrix multiplication
        start_time = MPI_Wtime();
    }

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads);
    
    #pragma omp parallel for shared(strip_A, B, strip_C)
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < matrix_size; j++) {
            sum = 0;
            for (int k = 0; k < matrix_size; k++) {
                sum += strip_A[i][k] * B[k][j];
            }
            strip_C[i][j] = sum;
        }
    }
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        cpu_time_multiplying = end_time - start_time;

        cpu_time_computation = cpu_time_generation + cpu_time_multiplying;
    }

    //print_matrix(strip_C, rows_per_process, matrix_size);

    if (rank == 0)
        start_time = MPI_Wtime();
    // gather results
    MPI_Gather(&(strip_C[0][0]), 1, strip, C_data, 1, strip, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        comms_data_aggregation = end_time - start_time;

        comms_time_total = comms_data_distribution + comms_data_aggregation;
    }

    /* if(rank == 0) {
        print_matrix(C, matrix_size, matrix_size);
    } */

    // Free resources
    MPI_Type_free(&strip);
    MPI_Type_free(&matrix);
    
    free(strip_A_data);
    free(strip_A);


    if (rank == 0) {
        free(A_data);
        free(A); 

        free(C_data);
        free(C); 
    }

    free(B_data);
    free(B);

    if (rank == 0) {
        end_tot = MPI_Wtime();
        total_time = end_tot - start_tot;
        printf("%lf\n", total_time);
    }

    MPI_Finalize();
    return 0;
}