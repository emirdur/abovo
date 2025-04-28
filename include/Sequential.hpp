#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP
#include <vector>
#include "DenseLayer.hpp"

class Sequential {
private:
    std::vector<DenseLayer> layers;

public:
    Sequential();

    void add(const DenseLayer& layer);
    Matrix forward(const Matrix& X) const;
    void print() const;
};

#endif