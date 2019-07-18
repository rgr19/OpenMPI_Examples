//#include "StdAfx.h"
#include "Vector2.h"
#include "Quaternion.h"
#include <stdio.h>

OVector2::OVector2()
{
	x_ = 0.0f;
	y_ = 0.0f;
}

OVector2::OVector2(float x, float y)
{
	x_ = x;
	y_ = y;
}

OVector2::OVector2(const OVector2& v)
{
	x_ = v.x_;
	y_ = v.y_;
}

OVector2::~OVector2()
{

}

OVector2& OVector2::operator=(const OVector2& v)
{
	x_ = v.x_;
	y_ = v.y_;
	return *this;
}

float& OVector2::operator[](int i)
{
	i = (i < 0) ? 0 : (i > 1 ? 1 : i);
	return vec_[i];
}

void OVector2::Print()
{
	printf("\n[%.2f, %.2f] \n", x_, y_);
}

OVector2 OVector2::operator+(const OVector2&v) const
{
	OVector2 r;
	for(int i = 0; i < 2; i++) {
		r.vec_[i] = vec_[i] + v.vec_[i];
	}
	return r;
}

OVector2 OVector2::operator-(const OVector2&v) const
{
	OVector2 r;
	for(int i = 0; i < 2; i++) {
		r.vec_[i] = vec_[i] - v.vec_[i];
	}
	return r;
}

OVector2 OVector2::operator*(const float& f) const
{
	OVector2 r;
	for(int i = 0; i < 2; i++) {
		r.vec_[i] = vec_[i] * f;
	}
	return r;
}

OVector2 operator*(const float& f, const OVector2& v)
{
	OVector2 r;
	r = v * f;
	return r;
}

float OVector2::operator*(const OVector2& v) const
{
	return x_ * v.y_ - y_ * v.x_;
}


float OVector2::Dot(const OVector2& v) const
{
	float r;
	r = x_ * v.x_ + y_ * v.y_;
	return r;
}

bool OVector2::normalize()
{
    const float err = 0.0001f;
    float r = sqrt(x_ * x_ +  y_ * y_);
    if(r < err)
    {
        return false;
    }
    x_ /= r;
    y_ /= r;

    return true;
}

bool OVector2::shortest_rotation(OVector2 axis_1, OVector2 axis_2)
{
	if(!axis_1.normalize() || !axis_2.normalize())
	{
		return true;
	}
	float rotation_axis_z = axis_1 * axis_2;
	float cos_angle = axis_1.Dot(axis_2);
	cos_angle = cos_angle > 1.0f ? 1.0f : (cos_angle < -1.0f ? -1.0f : cos_angle);
	float rotation_angle = acosf(cos_angle);
	float sn = sinf(rotation_angle);
	float cs = cosf(rotation_angle);

	*this = OVector2(axis_2.x_ * cs - axis_2.y_ * sn, axis_2.x_ * sn + axis_2.y_ * cs);

	return true;
}