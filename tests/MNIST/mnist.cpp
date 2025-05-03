#include <iostream>

#include "nn/Matrix.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Sequential.hpp"
#include "nn/Activation.hpp"
#include "Loader.cpp"

using namespace nn;

int main() {
    int num_train, num_labels;
    int num_test, num_test_labels;
    try {
        auto images = load_mnist_images("train-images.idx3-ubyte", num_train);
        auto labels = load_mnist_labels("train-labels.idx1-ubyte", num_labels);
    
        auto test_images = load_mnist_images("t10k-images.idx3-ubyte", num_test);
        auto test_labels = load_mnist_labels("t10k-labels.idx1-ubyte", num_test_labels);

        std::clog << "[DEBUG]: Loaded " << num_train << " training images and " << num_labels << " labels." << std::endl;
        
        if (num_train != num_labels) {
            std::cerr << "[Error]: Number of images and labels don't match!" << std::endl;
            return 1;
        }
        
        Sequential model;
        model.add(DenseLayer(784, 512, ActivationType::LEAKY_RELU));
        model.add(DenseLayer(512, 256, ActivationType::LEAKY_RELU));
        model.add(DenseLayer(256, 128, ActivationType::LEAKY_RELU));
        model.add(DenseLayer(128, 10, ActivationType::SIGMOID));
        
        Matrix X_train(images);
        Matrix y_train(labels);
        Matrix X_test(test_images);
        Matrix y_test(test_labels);
        
        std::clog << "[DEBUG]: Starting training..." << std::endl;
        model.train(X_train, y_train, 15, 64, 0.01);
        
        std::clog << "[DEBUG]: Training complete!" << std::endl;
        
        double test_accuracy = model.evaluate(X_test, y_test);
        std::cout << "Final test accuracy: " << test_accuracy * 100 << "%" << std::endl;

        model.dequantizeAll();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR]: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}