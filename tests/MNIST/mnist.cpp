#include <iomanip>
#include <iostream>
#include <string>

#include "Loader.cpp"
#include "nn/Activation.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Matrix.hpp"
#include "nn/Sequential.hpp"

using namespace nn;

void evaluate_model(Sequential &model, const Matrix &X_test,
                    const Matrix &y_test, const std::string &description) {
  double accuracy = model.evaluate(X_test, y_test);
  std::cout << description << ": " << std::fixed << std::setprecision(2)
            << accuracy * 100 << "%" << std::endl;
}

int main() {
  int num_train, num_labels;
  int num_test, num_test_labels;
  try {
    // Load MNIST dataset
    auto images = load_mnist_images("train-images.idx3-ubyte", num_train);
    auto labels = load_mnist_labels("train-labels.idx1-ubyte", num_labels);

    auto test_images = load_mnist_images("t10k-images.idx3-ubyte", num_test);
    auto test_labels =
        load_mnist_labels("t10k-labels.idx1-ubyte", num_test_labels);

    std::clog << "[DEBUG]: Loaded " << num_train << " training images and "
              << num_labels << " labels." << std::endl;

    if (num_train != num_labels) {
      std::cerr << "[Error]: Number of images and labels don't match!"
                << std::endl;
      return 1;
    }

    Matrix X_train(images);
    Matrix y_train(labels);
    Matrix X_test(test_images);
    Matrix y_test(test_labels);

    // We try quantization...
    std::cout << "\nQuantization-Aware Training:" << std::endl;

    Sequential qat_model;
    qat_model.add(DenseLayer(784, 512, ActivationType::RELU));
    qat_model.add(DenseLayer(512, 256, ActivationType::RELU));
    qat_model.add(DenseLayer(256, 128, ActivationType::RELU));
    qat_model.add(DenseLayer(128, 10, ActivationType::SOFTMAX));

    std::clog << "[DEBUG]: Starting training..." << std::endl;
    qat_model.enableAdam(true);
    qat_model.train(X_train, y_train, 25, 64, 0.001, LossType::CROSS_ENTROPY);

    // qat_model.enableQAT(true);

    // std::clog << "[DEBUG]: Starting QAT refinement..." << std::endl;
    // qat_model.train(X_train, y_train, 15, 64, 0.0001,
    // LossType::CROSS_ENTROPY); std::clog << "[DEBUG]: QAT training complete!"
    // << std::endl;

    evaluate_model(qat_model, X_test, y_test, "QAT model in FP32");

    qat_model.quantizeAll(true);
    evaluate_model(qat_model, X_test, y_test, "QAT model in INT8");

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    return 1;
  }
}