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
  std::cout << "[RESULT] "
            << description
            << ": " << std::setprecision(2)
            << accuracy * 100 << "%" << std::endl;
}


void log_info(const std::string &msg) {
  std::cout << "[INFO] " << msg << std::endl;
}

void log_debug(const std::string &msg) {
  std::clog << "[DEBUG] " << msg << std::endl;
}

void log_error(const std::string &msg) {
  std::cerr << "[ERROR] " << msg << std::endl;
}


int main() {
  int num_train, num_labels;
  int num_test, num_test_labels;
  try {
    log_info("Loading MNIST dataset...");
    auto images = load_mnist_images("train-images.idx3-ubyte", num_train);
    auto labels = load_mnist_labels("train-labels.idx1-ubyte", num_labels);

    auto test_images = load_mnist_images("t10k-images.idx3-ubyte", num_test);
    auto test_labels =
        load_mnist_labels("t10k-labels.idx1-ubyte", num_test_labels);

    log_debug("Loaded " + std::to_string(num_train) + " training images and " +
              std::to_string(num_labels) + " labels.");
    log_debug("Loaded " + std::to_string(num_test) + " test images and " +
              std::to_string(num_test_labels) + " labels.");

    if (num_train != num_labels) {
      log_error("Number of images and labels don't match!");
      return 1;
    }

    Matrix X_train(images), y_train(labels);
    Matrix X_test(test_images), y_test(test_labels);

    Sequential qat_model;
    qat_model.add(DenseLayer(784, 512, ActivationType::RELU));
    qat_model.add(DenseLayer(512, 256, ActivationType::RELU));
    qat_model.add(DenseLayer(256, 128, ActivationType::RELU));
    qat_model.add(DenseLayer(128, 10, ActivationType::SOFTMAX));

    log_info("Starting training (epochs=25, batch=64, lr=0.001)");
    qat_model.enableAdam(true);
    qat_model.train(X_train, y_train, 25, 64, 0.001, LossType::CROSS_ENTROPY);

    evaluate_model(qat_model, X_test, y_test, "Sequential model in FP32");

    qat_model.quantizeAll(true);
    evaluate_model(qat_model, X_test, y_test, "PTQ sequential model in INT8");

    return 0;

  } catch (const std::exception &e) {
    log_error(e.what());
    return 1;
  }
}
