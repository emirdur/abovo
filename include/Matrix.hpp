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
    void print() const;
    int getRows() const;
    int getCols() const;
    // won't change any member data within the function
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator=(const Matrix& other);
    double& operator()(int row, int col) const;
    Matrix transpose() const;
};

#endif