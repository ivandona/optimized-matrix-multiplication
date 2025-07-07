#include <stdio.h>
#include <stdlib.h>
#include "../../../include/utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int tid, nthreads;
    int matrix_size;
    double **A, **B, **C;
    double *A_data, *B_data, *C_data;
    double start_time, end_time, others_time, calculation_time, total_time;

    start_time = omp_get_wtime();

    /* get matrix size and thread nr*/
    matrix_size = 4096;
    nthreads = atoi(argv[1]);

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

    // Writing timing into CSV file
    char filename[128];
    if(argc==2){
        snprintf(filename, sizeof(filename),"res/%s/output_base_omp_%d_%d.csv", argv[2], 1, nthreads);
    }
    else snprintf(filename, sizeof(filename),"res/output_base_omp_%d_%d.csv", 1, nthreads);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,1,%d,%.3f,%.3f,%.3f,%.3f,0,0,0\n",
            "base_mpi",               // algorithm
            n,                        // matrix dimension
            nthreads,                 // nr of threads
            total_time,               // total time
            cpu_time_computation,     // total computation time
            cpu_time_multiplying,     // comp. multiplication time
            others_time,      // generation time
        );
        fclose(fp);
    }
}