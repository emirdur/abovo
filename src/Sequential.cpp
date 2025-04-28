#include "Sequential.hpp"

Sequential::Sequential() {}

Matrix Sequential::forward(const Matrix& X) const {
    Matrix out = X;

    for (const auto& layer : layers) {
        out = layer.forward(out);
        out = layer.activation(out);
    }

    return out;
} 

void Sequential::add(const DenseLayer& layer) {
    layers.push_back(layer);
}

void Sequential::print() const {
    for (auto& layer : layers) {
        layer.print();
    }
}

