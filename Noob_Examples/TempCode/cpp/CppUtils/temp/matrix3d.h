#pragma once

#include <vector>
#include <iostream>
#include "matrix.h"

class Matrix3D
{
protected:
    unsigned int _matrix2d_count;
    std::vector<Matrix2D>  _data;
public:
    Matrix3D(std::vector<int> sizes);
    Matrix3D();
    ~Matrix3D();

    void randInit();
    void zeros();

    std::vector<Matrix2D> GetData() const;
    unsigned int GetMatrix2DCount() const;
    Matrix2D GetMatrix2D(unsigned int i);

    void SetData(unsigned int i, Matrix2D m);

    Matrix3D add(Matrix3D m);
    Matrix3D sub(Matrix3D m);
    Matrix3D multiplyByScalar(double s);

    void printData();
    void writeData(std::string name);
};
