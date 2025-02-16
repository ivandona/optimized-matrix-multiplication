#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>  // For seeding the random number generator

void generate_random_matrix(int *matrix, int rows, int cols, int min_val, int max_val) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            matrix[i * cols + j] = min_val + rand() % (max_val - min_val + 1);
        }
    }
}

void print_matrix(int *matrix, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, comm_sz;
    int *A, *B, *C;  // Matrices A, B, and C (result)
    int *local_A, *local_C;  // Local portions of A and C for each process
    int rows_per_process, remainder_rows;
    int i, j, k;
    int N;  // Matrix comm_sz (N x N)
    double start_time, end_time, total_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    N = atoi(argv[1]);

    // Seed the random number generator (only once in the root process)
    if (rank == 0) {
        srand(time(NULL));
    }

    rows_per_process = N / comm_sz;
    //remainder_rows = N % comm_sz;

    if (rank == 0) {
        A = (int *)malloc(N * N * sizeof(int));
        B = (int *)malloc(N * N * sizeof(int));
        C = (int *)malloc(N * N * sizeof(int));

        generate_random_matrix(A, N, N, 0, 100);
        generate_random_matrix(B, N, N, 0, 100);
    }

    // Allocate memory for local_A and local_C
    local_A = (int *)malloc(rows_per_process * N * sizeof(int));
    local_C = (int *)malloc(rows_per_process * N * sizeof(int));
    
    if (rank == 0)
        start_time = MPI_Wtime();

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, rows_per_process * N, MPI_INT, local_A, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    //perform vector multiplication by all processes
    /* for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = sum + local_A[j] * B[j][i];            
        }
        local_C[i] = sum;
        sum = 0;
    } */

    if (rank == 0 && remainder_rows > 0) {
        for (i = comm_sz * rows_per_process; i < N; i++) {
            for (j = 0; j < N; j++) {
                C[i * N + j] = 0;
                for (k = 0; k < N; k++) {
                    C[i * N + j] += A[i * N + k] * B[k * N + j];
                }
            }
        }
    }

    // Gather results from all processes
    MPI_Gather(local_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
        printf("Time: %lf\n", total_time);

        // Free memory
        free(A);
        free(B);
        free(C);
    }
    
    // Free local memory
    free(local_A);
    free(local_C);
    
    MPI_Finalize();
    return 0;
}

