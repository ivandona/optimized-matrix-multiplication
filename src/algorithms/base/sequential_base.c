#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

int main(int argc, char** argv) {
    int matrix_size;
    int **A, **B, **C, **strip_A, **strip_C;
    int *A_data, *B_data, *C_data, *strip_A_data, *strip_C_data;
    clock_t start_time, end_time;
    double total_time;

    start_time = clock();

    srand(1);

    matrix_size = atoi(argv[1]);

    /* allocate and fill matrix A */
    alloc_matrix(&A, &A_data, matrix_size, matrix_size);
    generate_random_matrix(A, matrix_size, 0, 100);

    /* allocate matrix C (results matrix) */
    alloc_matrix(&C, &C_data, matrix_size, matrix_size);
    
    /* allocate and fill matrix B */
    alloc_matrix(&B, &B_data, matrix_size, matrix_size);
    generate_random_matrix(B, matrix_size, 0, 100);

    /* allocate sub-matrices */
    alloc_matrix(&strip_A, &strip_A_data, rows_per_process, matrix_size);
    alloc_matrix(&strip_C, &strip_C_data, rows_per_process, matrix_size);
    init_matrix(strip_C, rows_per_process, matrix_size);
    
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            for (int k = 0; k < matrix_size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }  

    //print_matrix(strip_C, rows_per_process, matrix_size);

    // gather results
    MPI_Gather(&(strip_C[0][0]), 1, strip, C_data, 1, strip, 0, MPI_COMM_WORLD);
    
    /* if(rank == 0) {
        print_matrix(C, matrix_size, matrix_size);
    } */

    // Free resources
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

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCK_PER_SEC;
    printf("%lf ", total_time);

    return 0;
}