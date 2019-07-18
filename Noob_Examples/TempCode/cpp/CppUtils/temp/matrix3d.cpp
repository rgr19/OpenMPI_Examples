#include "matrix3d.h"

Matrix3D::Matrix3D(std::vector<int> sizes)
{
    _matrix2d_count = sizes.size()-1;

    _data.resize(_matrix2d_count);
    for (int i = 0; i<_matrix2d_count;i++){
        _data[i] = Matrix2D(sizes[i+1],sizes[i]);
    }
}

Matrix3D::Matrix3D()
{

}

Matrix3D::~Matrix3D()
{
    _data.clear();
}

void Matrix3D::randInit()
{
    for (int i = 0; i<_matrix2d_count;i++){
        _data[i].randInit();
    }
}

void Matrix3D::zeros()
{
    for (int i = 0; i<_matrix2d_count;i++){
        _data[i].zeros();
    }
}

unsigned int Matrix3D::GetMatrix2DCount() const
{
    return _matrix2d_count;
}

std::vector<Matrix2D> Matrix3D::GetData() const
{
    return _data;
}

Matrix2D Matrix3D::GetMatrix2D(unsigned int i)
{
    return _data[i];
}

void Matrix3D::SetData(unsigned int i, Matrix2D m)
{
    _data[i] = m;
}


void Matrix3D::printData()
{
    for (unsigned int i = 0; i < _matrix2d_count; i++)
        _data[i].printData();
}

void Matrix3D::writeData(std::string name)
{
    std::ofstream ofs;
    ofs.open (name.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << _matrix2d_count << std::endl;

    for (unsigned int i = 0; i < _matrix2d_count; i++)
        _data[i].writeData(name);

    ofs.close();
}

Matrix3D Matrix3D::add(Matrix3D m)
{
    Matrix3D ret = m;
    for(unsigned int i = 0; i < m.GetMatrix2DCount(); i++){
        Matrix2D v = _data[i].add(m.GetData()[i]);
        ret.SetData(i,v);
    }
    return ret;
}

Matrix3D Matrix3D::sub(Matrix3D m)
{
    Matrix3D ret = m;
    for(unsigned int i = 0; i < m.GetMatrix2DCount(); i++){
        Matrix2D v = _data[i].sub(m.GetData()[i]);
        ret.SetData(i,v);
    }
    return ret;
}

Matrix3D Matrix3D::multiplyByScalar(double s)
{
    Matrix3D ret = *this;
    for(unsigned int i = 0; i < _matrix2d_count; i++){
        Matrix2D v = _data[i].multiplyByScalar(s);
        ret.SetData(i,v);
    }
    return ret;
}
