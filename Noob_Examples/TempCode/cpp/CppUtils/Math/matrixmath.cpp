//******************************************************************************
//** UTILS: Polarized Light Scattering C++ Class Library
//**
//** File: matrixmath.cpp
//**
//** Thomas A. Germer
//** Sensor Science Division, National Institute of Standards and Technology
//** 100 Bureau Dr. Stop 8443; Gaithersburg, MD 20899-8443
//** Phone: (301) 975-2876
//** Email: thomas.germer@nist.gov
//**
//** Version: 7.00 (January 2015)
//**
//******************************************************************************
#include <vector>
#include <cmath>
#include "../utils.h"

#include "matrixmath.h"

using namespace std;

#define THROW(x) throw SCATMECH_exception(x);

namespace UTILS {


	namespace FFTW {

		static double  CABS1(std::complex<double> ZDUM) {
			return fabs(real(ZDUM)) + fabs(imag(ZDUM));
		}

		template <class T> inline T mIn(const T a,const T b) {
			return (a>b)? b : a;
		}
		template <class T> inline T mAx(const T a,const T b) {
			return (a<b)? b : a;
		}





	} // namespace UTILS::FFTW


} // namespace UTILS
