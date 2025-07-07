#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../../include/utils.h"

double **strassen(double **A, double **B, int n) {
    int k;
    double **result, **a00, **a01, **a10, **a11, **b00, **b01, **b10, **b11;
    double *result_data, *a00_data, *a01_data, *a10_data, *a11_data, *b00_data, *b01_data, *b10_data, *b11_data;

    alloc_matrix(&result, &result_data, n, n);

    if (n == 1) {
        result[0][0] = A[0][0] * B[0][0];
        return result;
    }

    k = n / 2;

    alloc_matrix(&a00, &a00_data, k, k);
    alloc_matrix(&a01, &a01_data, k, k);
    alloc_matrix(&a10, &a10_data, k, k);
    alloc_matrix(&a11, &a11_data, k, k);
    alloc_matrix(&b00, &b00_data, k, k);
    alloc_matrix(&b01, &b01_data, k, k);
    alloc_matrix(&b10, &b10_data, k, k);
    alloc_matrix(&b11, &b11_data, k, k);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a00[i][j] = A[i][j];
            a01[i][j] = A[i][j + k];
            a10[i][j] = A[i + k][j];
            a11[i][j] = A[i + k][j + k];

            b00[i][j] = B[i][j];
            b01[i][j] = B[i][j + k];
            b10[i][j] = B[i + k][j];
            b11[i][j] = B[i + k][j + k];
        }
    }

    double **p1 = strassen(a00, add(b01, b11, k, -1), k);
    double **p2 = strassen(add(a00, a01, k, 1), b11, k);
    double **p3 = strassen(add(a10, a11, k, 1), b00, k);
    double **p4 = strassen(a11, add(b10, b00, k, -1), k);
    double **p5 = strassen(add(a00, a11, k, 1), add(b00, b11, k, 1), k);
    double **p6 = strassen(add(a01, a11, k, -1), add(b10, b11, k, 1), k);
    double **p7 = strassen(add(a00, a10, k, -1), add(b00, b01, k, 1), k);

    double **c00 = add(add(add(p5, p4, k, 1), p6, k, 1), p2, k, -1);
    double **c01 = add(p1, p2, k, 1);
    double **c10 = add(p3, p4, k, 1);
    double **c11 = add(add(add(p5, p1, k, 1), p3, k, -1), p7, k, -1);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            result[i][j] = c00[i][j];
            result[i][j + k] = c01[i][j];
            result[i + k][j] = c10[i][j];
            result[i + k][j + k] = c11[i][j];
        }
    }

    // Free all allocated memory
    free_matrix(a00); free_matrix(a01); free_matrix(a10); free_matrix(a11);
    free_matrix(b00); free_matrix(b01); free_matrix(b10); free_matrix(b11);
    free_matrix(p1); free_matrix(p2); free_matrix(p3); free_matrix(p4);
    free_matrix(p5); free_matrix(p6); free_matrix(p7);
    free_matrix(c00); free_matrix(c01); free_matrix(c10); free_matrix(c11);

    return result;
}

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

    C = strassen(A, B, matrix_size);

    printf("Result:\n");
    print_matrix(C, matrix_size, matrix_size);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("%lf\n", total_time);

    return 0;
}
