#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

#include "nn/Sequential.hpp"

namespace nn {

Sequential::Sequential() {}

Matrix Sequential::forward(const Matrix& X) {
    Matrix out = X;

    for (auto& layer : layers) {
        out = layer.forward(out);
        out = layer.activation(out);
    }

    return out;
} 

void Sequential::backward(const Matrix& y_pred, const Matrix& y_true, double learning_rate) {
    Matrix gradient = loss.loss_derivative(y_pred, y_true, LossType::MSE);
    
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i].backward(gradient, learning_rate);
    }
}

void Sequential::add(const DenseLayer& layer) {
    layers.push_back(layer);
}

void Sequential::print() const {
    for (auto& layer : layers) {
        layer.print();
    }
}

// mini-batch implementation
void Sequential::train(const Matrix& X, const Matrix& y, int epochs, int batch_size, double learning_rate) {
    int num_samples = X.getRows();
    int X_cols = X.getCols();
    int y_cols = y.getCols();
    
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), g);
        
        double epoch_loss = 0.0;
        int num_batches = ceil(static_cast<double>(num_samples) / batch_size);
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min((batch + 1) * batch_size, num_samples);
            int current_batch_size = end_idx - start_idx;
            
            Matrix batch_X(current_batch_size, X_cols);
            Matrix batch_y(current_batch_size, y_cols);
            
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start_idx + i];

                for (int j = 0; j < X_cols; ++j) {
                    batch_X(i, j) = X(idx, j);
                }
                for (int j = 0; j < y_cols; ++j) {
                    batch_y(i, j) = y(idx, j);
                }
            }
            
            Matrix preds = forward(batch_X);
            
            double batch_loss = loss.loss(preds, batch_y, LossType::MSE);
            epoch_loss += batch_loss;
            
            backward(preds, batch_y, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
        }
    }
}

// finds max value = prediction and label
double Sequential::evaluate(const Matrix& X_test, const Matrix& y_test) {
    Matrix y_pred = forward(X_test);
    int correct = 0;

    for (int i = 0; i < y_pred.getRows(); ++i) {
        int pred_label = 0;
        double max_pred = y_pred(i, 0);
        for (int j = 1; j < y_pred.getCols(); ++j) {
            if (y_pred(i, j) > max_pred) {
                max_pred = y_pred(i, j);
                pred_label = j;
            }
        }
        
        int true_label = 0;
        double max_true = y_test(i, 0);
        for (int j = 1; j < y_test.getCols(); ++j) {
            if (y_test(i, j) > max_true) {
                max_true = y_test(i, j);
                true_label = j;
            }
        }
        
        if (pred_label == true_label) ++correct;
    }

    return static_cast<double>(correct) / X_test.getRows();
}

void Sequential::quantizeAll() {
    for (auto& layer : layers) {
        layer.quantize();
    }
}

void Sequential::dequantizeAll() {
    for (auto& layer : layers) {
        layer.dequantize();
    }
}

bool Sequential::isQuantized() const {
    for (const auto& layer : layers) {
        if (!layer.isQuantized()) {
            return false;
        }
    }
    return true;
}

}