#include <iostream>
#include "Matrix.hpp"
#include "DenseLayer.hpp"

int main() {
    std::cout << "Hello NN-ab-ovo!" << std::endl;
    Matrix mat(1, 1);
    Matrix mat2(3, 3);
    Matrix mat3(3, 3);
    DenseLayer dl(1, 1);
    mat.randomize();
    dl.forward(mat);
    dl.print();
    return 0;
}
