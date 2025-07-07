#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../../include/utils.h"

int main(int argc, char** argv) {
    int matrix_size;
    double **A, **B, **C;
    double *A_data, *B_data, *C_data;
    clock_t start_time, end_time;
    double total_time;

    start_time = clock();

    srand(1);

    matrix_size = atoi(argv[1]);

    /* allocate and fill matrix A */
    alloc_matrix(&A, &A_data, matrix_size, matrix_size);
    generate_random_matrix(A, matrix_size, 0, 100);
    
    /* allocate and fill matrix B */
    alloc_matrix(&B, &B_data, matrix_size, matrix_size);
    generate_random_matrix(B, matrix_size, 0, 100);

    /* allocate matrix C (results matrix) */
    alloc_matrix(&C, &C_data, matrix_size, matrix_size);
    init_matrix(C, matrix_size, matrix_size);
    
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            for (int k = 0; k < matrix_size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }  

    //print_matrix(strip_C, rows_per_process, matrix_size);

    
    /* if(rank == 0) {
        print_matrix(C, matrix_size, matrix_size);
    } */

    // Free resources
    free(A_data);
    free(A); 

    free(B_data);
    free(B);

    free(C_data);
    free(C); 

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("%lf\n", total_time);

    char filename[128];
    snprintf(filename, sizeof(filename),"res/output_base_seq_%d_%d.csv", 1, 1);
    FILE *fp = fopen(filename, "a");
    if (fp != NULL){
        fprintf(fp,
            "%s,%d,1,1,%.3f\n",
            "base_seq",              // algorithm
            n,                        // matrix dimension
            total_time,               // total time
        );
        fclose(fp);
    }

    return 0;
}