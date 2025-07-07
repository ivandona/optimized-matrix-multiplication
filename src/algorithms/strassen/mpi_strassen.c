#include <stdio.b11>
#include <stdlib.b11>
#include <mpi.b11>
#include "../../../include/utils.b11"

/*
Divide matrix nxn into its 4 quadrants. 1 sub-matrix (n/2)x(n/2) for each quadrant.
*/
void fill_submatrices(double **matrix, double **q00, double **q01, double **q10, double **q11, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            q00[i][j] = matrix[i][j];
            q01[i][j] = matrix[i][j + k];
            q10[i][j] = matrix[i + k][j];
            q11[i][j] = matrix[i + k][j + k];
        }
    }
}

/*
Fill matrix nxn with elements from 4 quadrant sub-matrices (n/2)x(n/2)
*/
void fill_matrix_with_quads(double **matrix, double **q00, double **q01, double **q10, double **q11, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            matrix[i][j] = q00[i][j];
            matrix[i][j + k] = q01[i][j];
            matrix[i + k][j] = q10[i][j];
            matrix[i + k][j + k] = q11[i][j];
        }
    }
}


/*
Add two matrices and store results in third matrix
*/
void add_matrices(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n, i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

/*
Subtract two matrices and store results in third matrix
*/
void subtract_matrices(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n, i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
} 

// Multiply two matrices using Strassen's algorithm and MPI
void strassen(double **A, double **B, double **C, int n, int rank, int startNode) {
    // Multiplication for base case
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    // Sums for 2x2 matrices
    if (n == 2) {
        double p1, p2, p3, p4, p5, p6, p7;         // hold results of Strassen's 7 equations

        // Find 7 equation results for 2x2 matrices
        p1 = A[0][0] * (B[0][1] - B[1][1]);
        p2 = (A[0][0] + A[0][1]) * B[1][1];   // (a00+a01)b11
        p3 = (A[1][0] + A[1][1]) * B[0][0];   // (a10+a11)b00
        p4 = A[1][1] * (B[1][0] - B[0][0]);   // a11(b10-b00)
        p5 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]);   // (a00+a11)(b00+b11)
        p6 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]);   // (a01-a11)(b10+b11)
        p7 = (A[0][0] - A[1][0]) * (B[0][0] + B[0][1]);   // (a00-a10)(b00+b01)

        // Fill result C based on p1-p7
        C[0][0] = p5 + p4 - p2 + p6;
        C[0][1] = p1 + p2;
        C[1][0] = p3 + p4;
        C[1][1] = p1 + p5 - p3 - p7;
    } else {    
        // Sub-quadrants of A, B and C
        double **a00, **a01, **a10, **a11, **b00, **b01, **b10, **b11, **c00, **c01, **c10, **c11;
        double *a00_data, *a01_data, *a10_data, *a11_data, *b00_data, *b01_data, *b10_data, *b11_data, *c00_data, *c01_data, *c10_data, *c11_data;

        // Result of sub-matrix operations
        double **result1, **result2;
        double *result1_data, *result2_data;

        // Matrices with the results to the Strassen's equations
        double **m1, **m2, **m3, **m4, **m5, **m6, **m7;
        double *m1_data, *m2_data, *m3_data, *m4_data, *m5_data, *m6_data, *m7_data;

        int k = n / 2; // n for each quadrant of the matrices
        
        // Alloc sub-matrices
        alloc_matrix(&a00, &a00_data, k, k);
        alloc_matrix(&a01, &a01_data, k, k);
        alloc_matrix(&a10, &a10_data, k, k);
        alloc_matrix(&a11, &a11_data, k, k);
        alloc_matrix(&b00, &b00_data, k, k);
        alloc_matrix(&b01, &b01_data, k, k);
        alloc_matrix(&b10, &b10_data, k, k);
        alloc_matrix(&b11, &b11_data, k, k); 
        alloc_matrix(&c00, &b00_data, k, k);
        alloc_matrix(&c01, &b01_data, k, k);
        alloc_matrix(&c10, &b10_data, k, k);
        alloc_matrix(&c11, &b11_data, k, k); 

        // Alloc result of sub-matrix operations
        alloc_matrix(&result1, &result1_data, k, k);
        alloc_matrix(&result2, &result2_data, k, k);

        // Alloc matrices with Strassen's equations results
        alloc_matrix(&m1, &m1_data, k, k);        
        alloc_matrix(&m2, &m2_data, k, k);
        alloc_matrix(&m3, &m3_data, k, k);
        alloc_matrix(&m4, &m4_data, k, k);
        alloc_matrix(&m5, &m5_data, k, k);
        alloc_matrix(&m6, &m6_data, k, k);
        alloc_matrix(&m7, &m7_data, k, k);

        // Fill sub-matrices from A and B
        fill_submatrices(A, a00, a01, a10, a11, k);
        fill_submatrices(B, b00, b01, b10, b11, k);

        // Find matrices m1-m7 with results for equations 1-7
        subtract_matrices(b01, b11, result1, k);            // b01-b11
        strassen(a00, result1, m1, k);               // a00(b01-b11)

        add_matrices(a00, a01, result1, k);                 // a00+a01
        strassen(result1, b11, m2, k);               // (a00+a01)b11

        add_matrices(C, a11, result1, k);                  // C+a11
        strassen(result1, b00, m3, k);                // (C+a11)b00

        subtract_matrices(b10, b00, result1, k);             // b10-b00
        strassen(a11, result1, m4, k);                // a11(b10-b00)

        add_matrices(a00, a11, result1, k);                  // a00+a11
        add_matrices(b00, b11, result2, k);                  // b00+b11
        strassen(result1, result2, m5, k);          // (a00+a11)(b00+b11)

        subtract_matrices(a01, a11, result1, k);             // a01-a11
        add_matrices(b10, b11, result2, k);                  // b10+b11
        strassen(result1, result2, m6, k);          // (a01-a11)(b10+b11)

        subtract_matrices(a00, C, result1, k);             // a00-C
        add_matrices(b00, b01, result2, k);                  // b00+b01
        strassen(result1, result2, m7, k);          // (a00-C)(b00+b01)

        // Determine quadrants of C based on m1-m7
        add_matrices(m5, m4, result1, k);                // m5+m4
        subtract_matrices(result1, m2, result2, k);      // m5+m4-m2
        add_matrices(result2, m6, quad1, k);             // m5+m4-m2+m6

        add_matrices(m1, m2, quad2, k);                  // m1+m2

        add_matrices(m3, m4, quad3, k);                  // m3+m4

        add_matrices(m1, m5, result1, k);                // m1+m5
        subtract_matrices(result1, m3, result2, k);      // m1+m5-m3
        subtract_matrices(result2, m7, quad4, k);        // m1+m5-m3-m7

        // Fill C from quadrants
        fill_matrix_with_quads(quad1, quad2, quad3, quad4, k, C, n);

        // Deallocate sub-matrices
        free_matrix(a00); free_matrix(a01); free_matrix(a10); free_matrix(a11);
        free_matrix(b00); free_matrix(b01); free_matrix(b10); free_matrix(b11);
        free_matrix(c00); free_matrix(c01); free_matrix(c10); free_matrix(c11);
        free_matrix(m1); free_matrix(m2); free_matrix(m3); free_matrix(m4);
        free_matrix(m5); free_matrix(m6); free_matrix(m7);
        free_matrix(result1); free_matrix(result2);
    }
}

void strassen_mpi(int rank, int world_size, double **A, double **B, double **C, int n) {
    int children[7];
    int can_spawn = 1;

    // Multiplication for base case
    if (n == 1  && rank == 0) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    // Sums for 2x2 matrices
    if (n == 2) {
        double p1, p2, p3, p4, p5, p6, p7;         // hold results of Strassen's 7 equations

        // Find 7 equation results for 2x2 matrices
        p1 = A[0][0] * (B[0][1] - B[1][1]);
        p2 = (A[0][0] + A[0][1]) * B[1][1];   // (a00+a01)b11
        p3 = (A[1][0] + A[1][1]) * B[0][0];   // (a10+a11)b00
        p4 = A[1][1] * (B[1][0] - B[0][0]);   // a11(b10-b00)
        p5 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]);   // (a00+a11)(b00+b11)
        p6 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]);   // (a01-a11)(b10+b11)
        p7 = (A[0][0] - A[1][0]) * (B[0][0] + B[0][1]);   // (a00-a10)(b00+b01)

        // Fill result C based on p1-p7
        C[0][0] = p5 + p4 - p2 + p6;
        C[0][1] = p1 + p2;
        C[1][0] = p3 + p4;
        C[1][1] = p1 + p5 - p3 - p7;
    } 
    if (n > 2) {    
        // Sub-quadrants of A, B and C
        double **a00, **a01, **a10, **a11, **b00, **b01, **b10, **b11, **c00, **c01, **c10, **c11;
        double *a00_data, *a01_data, *a10_data, *a11_data, *b00_data, *b01_data, *b10_data, *b11_data, *c00_data, *c01_data, *c10_data, *c11_data;

        // Result of sub-matrix operations
        double **result1, **result2;
        double *result1_data, *result2_data;

        // Matrices with the results to the Strassen's equations
        double **m1, **m2, **m3, **m4, **m5, **m6, **m7;
        double *m1_data, *m2_data, *m3_data, *m4_data, *m5_data, *m6_data, *m7_data;

        int k = n / 2; // n for each quadrant of the matrices
        
        // Alloc sub-matrices
        alloc_matrix(&a00, &a00_data, k, k);
        alloc_matrix(&a01, &a01_data, k, k);
        alloc_matrix(&a10, &a10_data, k, k);
        alloc_matrix(&a11, &a11_data, k, k);
        alloc_matrix(&b00, &b00_data, k, k);
        alloc_matrix(&b01, &b01_data, k, k);
        alloc_matrix(&b10, &b10_data, k, k);
        alloc_matrix(&b11, &b11_data, k, k); 
        alloc_matrix(&c00, &b00_data, k, k);
        alloc_matrix(&c01, &b01_data, k, k);
        alloc_matrix(&c10, &b10_data, k, k);
        alloc_matrix(&c11, &b11_data, k, k); 

        // Alloc result of sub-matrix operations
        alloc_matrix(&result1, &result1_data, k, k);
        alloc_matrix(&result2, &result2_data, k, k);

        // Alloc matrices with Strassen's equations results
        alloc_matrix(&m1, &m1_data, k, k);        
        alloc_matrix(&m2, &m2_data, k, k);
        alloc_matrix(&m3, &m3_data, k, k);
        alloc_matrix(&m4, &m4_data, k, k);
        alloc_matrix(&m5, &m5_data, k, k);
        alloc_matrix(&m6, &m6_data, k, k);
        alloc_matrix(&m7, &m7_data, k, k);

        // Fill sub-matrices from A and B
        fill_submatrices(A, a00, a01, a10, a11, k);
        fill_submatrices(B, b00, b01, b10, b11, k);

        // Generate children ranks
        for (int i = 0; i < 7; i++) {
            children[i] = 7 * rank + (i + 1);
            if (children[i] >= world_size) {
                can_spawn = 0;
            }
        }

        // Find matrices m1-m7 with results for equations 1-7
        
        if (my_rank == startNode || 
            ((my_rank >= (7+(7*startNode))) &&        // >= next-level start node
             (my_rank <= (13+(7*startNode)))))        // <= next-level end node
        {
            if ((7+(7*startNode)) > 49)                // 49 is max next-level start node
            {
                subtract_matrices(b01, b11, result1);           // b01-b11
                strassen(a00, result1, k, rank, startNode);     // a00(b01-b11)
            }
            else
            {
                if (my_rank != startNode)    // only next-level nodes should do recursive call
                {
                    SubtractMatrices(f, h, result1, subDim);              // f-h
                    StrassenMultMPI(a, result1, m1, subDim, my_rank,      // a(f-h)
                                    (7+(7*startNode)));    
                }
                if (my_rank == (7+(7*startNode)))         
                    // Send m1 from next-level startNode to current startNode
                    MPI_Send(m1, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
                if (my_rank == startNode)
                    // Receive m1
                    MPI_Recv(m1, subDim*subDim, MPI_DOUBLE, 
                             (7+(7*startNode)), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (!can_spawn || n <= THRESHOLD_SIZE) {
            // Base case: compute Strassen locally
            strassen_serial(A, B, C, n);
            return;
        }

        // Split matrices into 2x2 submatrices (A11, A12, ..., B22)
        Matrix A11, A12, A21, A22;
        Matrix B11, B12, B21, B22;
        Matrix C11, C12, C21, C22;
        // ... split A, B into quadrants

        // Prepare and send M1–M7 subproblems to each child
        for (int i = 0; i < 7; i++) {
            // Construct the input for M[i]
            // (e.g., A11 + A22, B11 + B22 for M1)
            Matrix localA, localB;
            // fill localA and localB for Mi
            MPI_Send(..., children[i], ...);
        }

        // Receive results from each child into M1–M7
        for (int i = 0; i < 7; i++) {
            Matrix Mi;
            MPI_Recv(..., children[i], ...);
        }

        // Reconstruct C from M1–M7 (C11, C12, etc.)
        // Combine into output matrix C
    }
}


/*
 * Recursive parallel Strassen multiply: C = A*B, all matrices n x n, comm is MPI_Comm.
 */
void strassen_mpi(double *A, double *B, double *C, int n) {

    // Multiplication for base case
    if(n == 1) {
        C[0][0] = A[0][0] * B[0][0]
    }

    // Sums for base case
    if(n == 2) {
        double p1, p2, p3, p4, p5, p6, p7;

        // Direct computation using Strassen's formulas for 2x2 matrices

    }
}
