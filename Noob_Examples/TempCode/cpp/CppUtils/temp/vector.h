#pragma once

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>

class Vector
{
protected:
    unsigned int _count;
    std::vector<double> _data;
public:
    Vector();
    Vector(unsigned int count);
    ~Vector();

    void randInit();
    void zeros();

    unsigned int GetCount() const;
    std::vector<double> GetData() const;
    double GetItem(unsigned int i);
    double GetMax();

    void SetData(std::vector<double> data);
    void SetData(Vector v);
    void SetItem(unsigned int i,double value);

    void printData();
    void writeData(std::string name);

    Vector add(Vector v);
    Vector sub(Vector v);
    Vector multiplyByScalar(double s);
    Vector HadamardProd(Vector v);

};

