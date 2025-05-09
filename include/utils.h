void alloc_matrix(int*** matrix, int** matrix_data, int rows, int cols) {
    // Allocate contiguous memory for matrix data
    *matrix_data = (int *)malloc(sizeof(int) * rows * cols);
    // Allocate memory for row pointers
    *matrix = (int **)malloc(sizeof(int *) * rows);

    if (*matrix_data == NULL || *matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Set row pointers to the correct positions
    for (int i = 0; i < rows; i++) {
        (*matrix)[i] = &((*matrix_data)[i * cols]);
    }
}

void free_matrix(int **matrix) {
    /* free the memory - the first element of the array is at the start */
    free(matrix[0]);

    /* free the pointers into the memory */
    free(matrix);
}

void init_matrix(int ** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0;
        }
    }
}

void generate_random_matrix(int** matrix, int matrix_size, int min_val, int max_val) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            matrix[i][j] = min_val + rand() % (max_val - min_val + 1);
        }
    }
}

void print_matrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}