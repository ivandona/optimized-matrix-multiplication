#include <stdio.h>
#include <stdlib.h>
#include "../../../include/utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int tid, nthreads;
    int matrix_size;
    int **A, **B, **C;
    int *A_data, *B_data, *C_data;
    double start_time, end_time, others_time, calculation_time, total_time;

    start_time = omp_get_wtime();

    /* get matrix size and thread nr*/
    matrix_size = atoi(argv[1]);
    nthreads = atoi(argv[2]);

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads);

    #ifdef _OPENMP
    #pragma omp parallel shared(A, B, C, A_data, B_data, C_data, nthreads, start_time, end_time, others_time) private(tid) 
    #endif
    {
        tid = omp_get_thread_num();

        alloc_matrix(&A, &A_data, matrix_size, matrix_size);
        alloc_matrix(&B, &B_data, matrix_size, matrix_size);
        alloc_matrix(&C, &C_data, matrix_size, matrix_size);
        
        generate_random_matrix(A, matrix_size, 0, 100);
        generate_random_matrix(B, matrix_size, 0, 100);
        init_matrix(C, matrix_size, matrix_size);
        
        end_time = omp_get_wtime();
        others_time = end_time - start_time;

        start_time = omp_get_wtime();
        
        #pragma omp for
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                for (int k = 0; k < matrix_size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        end_time = omp_get_wtime();
    }

    calculation_time = end_time - start_time;

    total_time = others_time + calculation_time;
    printf("Others time: %lf\n", others_time);
    printf("Calculation time: %lf\n", calculation_time);
    printf("Totale time: %lf\n", total_time);

}