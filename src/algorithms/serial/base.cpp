#include "base.h"
#include <stdexcept>  // For handling errors (if needed)

void BaseMatrix::multiply(const std::vector<std::vector<int>>& A,
                          const std::vector<std::vector<int>>& B,
                          std::vector<std::vector<int>>& C) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // Ensure matrix dimensions are compatible for multiplication
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // Resize C to correct dimensions
    C.assign(rowsA, std::vector<int>(colsB, 0));

    // Naive O(n³) matrix multiplication
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
