#include <iostream>
#include "Matrix.hpp"

int main() {
    std::cout << "Hello NN-ab-ovo!" << std::endl;
    Matrix mat(3, 3);
    Matrix mat2(3, 3);
    Matrix mat3(3, 3);
    mat.randomize();
    mat3 = mat + mat2;
    mat3.print();
    return 0;
}
