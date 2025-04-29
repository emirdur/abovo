#include "Sequential.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

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
    Matrix gradient = loss.d_mse(y_pred, y_true);
    
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
            
            double batch_loss = loss.mse(preds, batch_y);
            epoch_loss += batch_loss;
            
            backward(preds, batch_y, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
        }
    }
}
