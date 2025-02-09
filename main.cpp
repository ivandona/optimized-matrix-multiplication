#include <iostream>
#include "algorithms/base.h"

int main() {
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{5, 6}, {7, 8}};
    std::vector<std::vector<int>> C;

    try {
        BaseMatrix::multiply(A, B, C);

        // Print result
        for (const auto& row : C) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
