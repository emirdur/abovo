#include <iostream>
#include "Matrix.hpp"
#include "DenseLayer.hpp"
#include "Sequential.hpp"

int main() {
    Matrix X(4, 2);
    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 0; X(1, 1) = 1;
    X(2, 0) = 1; X(2, 1) = 0;
    X(3, 0) = 1; X(3, 1) = 1;

    Matrix y(4, 1);
    y(0, 0) = 0;
    y(1, 0) = 1;
    y(2, 0) = 1;
    y(3, 0) = 0;

    Sequential model;
    model.add(DenseLayer(2, 4));
    model.add(DenseLayer(4, 1));

    // epochs, batch_size, learning_rate
    model.train(X, y, 500, 2, 0.01);

    std::cout << std::endl << "Predictions after training:" << std::endl;
    Matrix predictions = model.forward(X);
    predictions.print();

    return 0;
}