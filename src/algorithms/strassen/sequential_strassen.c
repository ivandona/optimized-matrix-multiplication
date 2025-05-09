#include <stdio.h>
#include <stdlib.h>

#define SIZE 4

void print_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%d ", matrix[i][j]);
        printf("\n");
    }
}

int **allocate_matrix(int n) {
    int **matrix = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        matrix[i] = (int *)malloc(n * sizeof(int));
    return matrix;
}

void free_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++)
        free(matrix[i]);
    free(matrix);
}

int **add(int **A, int **B, int n, int multiplier) {
    int **result = allocate_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = A[i][j] + multiplier * B[i][j];
    return result;
}

int **strassen(int **A, int **B, int n) {
    int **result = allocate_matrix(n);

    if (n == 1) {
        result[0][0] = A[0][0] * B[0][0];
        return result;
    }

    int k = n / 2;

    int **a00 = allocate_matrix(k);
    int **a01 = allocate_matrix(k);
    int **a10 = allocate_matrix(k);
    int **a11 = allocate_matrix(k);
    int **b00 = allocate_matrix(k);
    int **b01 = allocate_matrix(k);
    int **b10 = allocate_matrix(k);
    int **b11 = allocate_matrix(k);

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

    int **p1 = strassen(a00, add(b01, b11, k, -1), k);
    int **p2 = strassen(add(a00, a01, k, 1), b11, k);
    int **p3 = strassen(add(a10, a11, k, 1), b00, k);
    int **p4 = strassen(a11, add(b10, b00, k, -1), k);
    int **p5 = strassen(add(a00, a11, k, 1), add(b00, b11, k, 1), k);
    int **p6 = strassen(add(a01, a11, k, -1), add(b10, b11, k, 1), k);
    int **p7 = strassen(add(a00, a10, k, -1), add(b00, b01, k, 1), k);

    int **c00 = add(add(add(p5, p4, k, 1), p6, k, 1), p2, k, -1);
    int **c01 = add(p1, p2, k, 1);
    int **c10 = add(p3, p4, k, 1);
    int **c11 = add(add(add(p5, p1, k, 1), p3, k, -1), p7, k, -1);

    for (int i = 0; i < k; i++) {-
        for (int j = 0; j < k; j++) {
            result[i][j] = c00[i][j];
            result[i][j + k] = c01[i][j];
            result[i + k][j] = c10[i][j];
            result[i + k][j + k] = c11[i][j];
        }
    }

    // Free all allocated memory
    free_matrix(a00, k); free_matrix(a01, k); free_matrix(a10, k); free_matrix(a11, k);
    free_matrix(b00, k); free_matrix(b01, k); free_matrix(b10, k); free_matrix(b11, k);
    free_matrix(p1, k); free_matrix(p2, k); free_matrix(p3, k); free_matrix(p4, k);
    free_matrix(p5, k); free_matrix(p6, k); free_matrix(p7, k);
    free_matrix(c00, k); free_matrix(c01, k); free_matrix(c10, k); free_matrix(c11, k);

    return result;
}

int main() {
    int **A = allocate_matrix(SIZE);
    int **B = allocate_matrix(SIZE);

    int a_data[SIZE][SIZE] = {
        {2, 2, 3, 1},
        {1, 4, 1, 2},
        {2, 3, 1, 1},
        {1, 3, 1, 2}
    };

    int b_data[SIZE][SIZE] = {
        {2, 1, 2, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {1, 4, 3, 2}
    };

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = a_data[i][j];
            B[i][j] = b_data[i][j];
        }

    int **C = strassen(A, B, SIZE);

    printf("Result:\n");
    print_matrix(C, SIZE);

    free_matrix(A, SIZE);
    free_matrix(B, SIZE);
    free_matrix(C, SIZE);

    return 0;
}
