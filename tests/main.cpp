#include <iostream>
#include "Matrix.hpp"
#include "DenseLayer.hpp"
#include "Sequential.hpp"

int main() {
    Sequential model;
    model.add(DenseLayer(2, 3));
    model.add(DenseLayer(3, 1));

    Matrix input(1, 2);
    input(0, 0) = 0.5;
    input(0, 1) = -1.5;

    Matrix output = model.forward(input);

    output.print();
}

