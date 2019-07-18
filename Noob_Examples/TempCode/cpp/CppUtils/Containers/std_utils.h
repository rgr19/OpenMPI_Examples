//
// Created by robgrzel on 17.05.17.
//
#pragma once

#include <vector>
#include <algorithm>

#ifndef STD_VECTOR_UTILS_H
#define STD_VECTOR_UTILS_H

#endif //STD_VECTOR_UTILS_H

template <typename T> std::vector<T> arange( T start, T stop, T step ) {
	std::vector<T> values;
	for ( T        value = start; value < stop; value += step ) {
		values.push_back( value );
	}
	return values;
}

template <class T>
void fill_zeros(std::vector<T> &v) {
	std::fill(v.begin(), v.end(), 0);
	
}

template <class T>
void operator+=(std::vector<T> &a, const std::vector<T> &b) {
	std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
	
}
template <class T>
void operator*=(std::vector<T> &a, const T &b) {
	std::transform(a.begin(), a.end(), a.begin(),
	               std::bind1st(std::multiplies<T>(), b));
	
}
template <class T>
std::vector<T> operator*(std::vector<T> a, const T &b) {
	std::transform(a.begin(), a.end(), a.begin(),
	               std::bind1st(std::multiplies<T>(), b));
	return a;
	
}

