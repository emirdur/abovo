#include <iostream>

#include "nn/Activation.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Matrix.hpp"
#include "nn/Sequential.hpp"

using namespace nn;

int main() {
  Matrix X(4, 2);
  X(0, 0) = 0;
  X(0, 1) = 0;
  X(1, 0) = 0;
  X(1, 1) = 1;
  X(2, 0) = 1;
  X(2, 1) = 0;
  X(3, 0) = 1;
  X(3, 1) = 1;

  Matrix y(4, 1);
  y(0, 0) = 0;
  y(1, 0) = 1;
  y(2, 0) = 1;
  y(3, 0) = 0;

  Sequential model;
  model.add(DenseLayer(2, 8, ActivationType::LEAKY_RELU));
  model.add(DenseLayer(8, 1, ActivationType::SIGMOID));

  // epochs, batch_size, learning_rate
  model.train(X, y, 1000, 2, 0.1, LossType::MSE);

  std::cout << std::endl << "Predictions after training:" << std::endl;
  Matrix predictions = model.forward(X);
  predictions.print();

  return 0;
}