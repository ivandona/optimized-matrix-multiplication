#ifndef BASE_H
#define BASE_H

#include <vector>  // Standard library for matrix representation

class BaseMatrix {
public:
    // Function to multiply two matrices A and B, storing result in C
    static void multiply(const std::vector<std::vector<int>>& A,
                         const std::vector<std::vector<int>>& B,
                         std::vector<std::vector<int>>& C);
};

#endif  // BASE_H
