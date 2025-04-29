#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include "Matrix.hpp"

Matrix::Matrix(int r, int c): rows(r), cols(c) {
    data = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new double[cols];
    }
}

Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

Matrix::~Matrix() {
    for (int i = 0; i < rows; ++i) {
        delete[] data[i];
    }
    delete[] data;
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    data = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            data[i][j] = other.data[i][j];
        }
    }
}

void Matrix::randomize(double fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std::sqrt(2.0 / fan_in)); // He weight initialization

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = dist(gen);
        }
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

// pass by reference so no need to copy
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows && other.rows != 1) {
        throw std::invalid_argument("Matrix dimensions must match or be broadcasted.");
    }
    if (cols != other.cols) {
        throw std::invalid_argument("Column dimensions must match.");
    }
    
    Matrix res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.data[i][j] = data[i][j] + other.data[other.rows == 1 ? 0 : i][j];
        }
    }    

    return res;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication.");
    }

    Matrix res(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            res.data[i][j] = 0;

            for (int k = 0; k < cols; ++k) {
                res.data[i][j] += data[i][k] * other.data[k][j]; // because inner dimensions must match
            }
        }
    }

    return res;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix res(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.data[i][j] = data[i][j] * scalar;
        }
    }

    return res;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }

    if (rows != other.rows || cols != other.cols) {
        // Deallocate current memory
        for (int i = 0; i < rows; ++i) {
            delete[] data[i];
        }
        delete[] data;

        // Allocate new memory
        rows = other.rows;
        cols = other.cols;
        data = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new double[cols];
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = other.data[i][j];
        }
    }

    return *this;
}

double& Matrix::operator()(int row, int col) const {
    return data[row][col];
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    
    return result;
}

Matrix Matrix::hadamard_product(const Matrix& other) const {
	if (rows != other.rows || cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions must match for hadamard multiplication.");
	}

    Matrix res(rows, cols);

	for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.data[i][j] = this->data[i][j] * other.data[i][j];
        }
    }

    return res;
}