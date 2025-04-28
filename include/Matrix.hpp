#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix {
private:
    int rows, cols;
    double** data;

public:
    Matrix(int r, int c);
    ~Matrix();

    Matrix(const Matrix& other);

    void randomize();
    void print();
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator=(const Matrix& other);
    Matrix transpose() const;
};

#endif