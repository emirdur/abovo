#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP
#include <vector>
#include "DenseLayer.hpp"
#include "Loss.hpp"

class Sequential {
private:
    std::vector<DenseLayer> layers;
    Loss loss;

public:
    Sequential();

    void add(const DenseLayer& layer);
    Matrix forward(const Matrix& X);
    void print() const;

    void backward(const Matrix& y_pred, const Matrix& y_true, double eta); 

    // stochastic-like implementation of gradient descent just with more input points in batches
    void train(const Matrix& X, const Matrix& y, int epochs, int batch_size, double learning_rate);
};

#endif