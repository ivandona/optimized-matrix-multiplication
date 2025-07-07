void alloc_matrix(double*** matrix, double** matrix_data, int rows, int cols) {
    // Allocate contiguous memory for matrix data
    *matrix_data = (double *)malloc(sizeof(double) * rows * cols);
    // Allocate memory for row pointers
    *matrix = (double **)malloc(sizeof(double *) * rows);

    if (*matrix_data == NULL || *matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Set row pointers to the correct positions
    for (int i = 0; i < rows; i++) {
        (*matrix)[i] = &((*matrix_data)[i * cols]);
    }
}

void alloc_matrix_array(double ****matrices, double ***matrices_data, int count, int rows, int cols) {
    *matrices = (double ***)malloc(sizeof(double **) * count);       // Array of matrix pointers
    *matrices_data = (double **)malloc(sizeof(double *) * count);   // Array of data blocks

    if (*matrices == NULL || *matrices_data == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix array\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < count; i++) {
        alloc_matrix(&((*matrices)[i]), &((*matrices_data)[i]), rows, cols);
    }
}

void free_matrix(double **matrix) {
    /* free the memory - the first element of the array is at the start */
    free(matrix[0]);

    /* free the pointers into the memory */
    free(matrix);
}

void free_matrix_array(double ***matrices, double **matrices_data, int count) {
    for (int i = 0; i < count; i++) {
        free(matrices[i]);       // Free row pointer array
        free(matrices_data[i]);  // Free contiguous data block
    }
    free(matrices);       // Free array of row-pointer arrays
    free(matrices_data);  // Free array of data blocks
}

void init_matrix(double ** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0;
        }
    }
}

void generate_random_matrix(double** matrix, int matrix_size, int min_val, int max_val) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            matrix[i][j] = min_val + rand() % (max_val - min_val + 1);
        }
    }
}

void print_matrix(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

double **add(double **A, double **B, int n, int multiplier) {
    double **result, *result_data;
    alloc_matrix(&result, &result_data, n, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = A[i][j] + multiplier * B[i][j];
    return result;
}