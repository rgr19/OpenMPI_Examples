#pragma once
#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>
#include <random>
#include <algorithm>
#include "const.h"

#define T2OM(T) double(2 * PI / T)
#define OM2T(om) double(2*PI/om)
#define NYQUIST(dt) (1/(2*dt))



double calc_poly(int n, double x, double * a);

int polyfit(
		const double * const X, //x
		const double * const Y, //y
		unsigned int n,
		unsigned int order,
		double * coefficients
);

double dround(double v, int n);

int iround(double r);

template <class T>
int line_side(T x0, T y0, T x1, T y1, T x, T y) {
	//-1 = under, 1 = over when x0 < x1
	double v = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0);
	return int(v / fabs(v));
}




long double simpshy(const double & h, const std::vector <double> & y);

long double simpsxy(const std::vector <double> & X, const std::vector <double> & Y);


double mind(std::vector <double> x);

double maxd(std::vector <double> x);

long arg_mind(std::vector <double> x);

long arg_maxd(std::vector <double> x);


void randn(const size_t N, double * A);

void randn2(const size_t N, double * A);

void randn2pi(const size_t N, double * A);

//Very fast sqrt
float SQRTF(float number);

double SQRT(double number);

//Very fast 1/sqrt
float RSQRTF(float number);

double RSQRT(double number);


inline int maximum(int a, int b) {
	return a > b ? a : b;
}

inline int maximum(int a, int b, int c) {
	return maximum(maximum(a, b), c);
}

inline int maximum(int a, int b, int c, int d, int e, int f) {
	return maximum(maximum(a, b, c), maximum(d, e, f));
}

inline float maximum(float a, float b) {
	return a > b ? a : b;
}

inline float maximum(float a, float b, float c) {
	return maximum(maximum(a, b), c);
}

inline float maximum(float a, float b, float c, float d, float e, float f) {
	return maximum(maximum(a, b, c), maximum(d, e, f));
}

inline double maximum(double a, double b) {
	return a > b ? a : b;
}

inline double maximum(double a, double b, double c) {
	return maximum(maximum(a, b), c);
}

inline double maximum(double a, double b, double c, double d, double e, double f) {
	return maximum(maximum(a, b, c), maximum(d, e, f));
}

inline int minimum(int a, int b) {
	return a < b ? a : b;
}

inline int minimum(int a, int b, int c) {
	return minimum(minimum(a, b), c);
}

inline int minimum(int a, int b, int c, int d, int e, int f) {
	return minimum(minimum(a, b, c), minimum(d, e, f));
}

inline float minimum(float a, float b) {
	return a < b ? a : b;
}

inline float minimum(float a, float b, float c) {
	return minimum(minimum(a, b), c);
}

inline float minimum(float a, float b, float c, float d, float e, float f) {
	return minimum(minimum(a, b, c), minimum(d, e, f));
}

inline double minimum(double a, double b) {
	return a < b ? a : b;
}

inline double minimum(double a, double b, double c) {
	return minimum(minimum(a, b), c);
}

inline double minimum(double a, double b, double c, double d, double e, double f) {
	return minimum(minimum(a, b, c), minimum(d, e, f));
}

inline int screen_coordinate(float c, int l, float a, float b) {
	int x = (c - a) * l / (b - a);
	
	if (x < 0) {
		x = 0;
		
	} else if (x > l) {
		x = l;
	}
	
	return x;
}

inline float norm_coordinate(float x, float l, float a, float b) {
	float c = a + (b - a) * x / l;
	
	if (c < a) {
		c = a;
		
	} else if (c > b) {
		c = b;
	}
	
	return c;
};


