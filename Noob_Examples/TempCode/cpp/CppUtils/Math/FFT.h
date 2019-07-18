//
// Created by robgrzel on 11.06.17.
//

#ifndef UTILS_FFT_H
#define UTILS_FFT_H

#include <iostream>
#include <valarray>
#include <map>
#include <omp.h>

#include "../utils.h"
#include "../Containers/std_utils.h"

#include "matrixmath.h"


namespace UTILS {
	
	
	namespace {
		// The following map stores work arrays for CFFTB and CFFTF...
		std::map<int, std::vector<double> > fftwork;
	}
	
	namespace FFTW {
		
		void CFFTI(int &N, DFARRAY WSAVE);
		
		void CFFTI1(int &N, DFARRAY WA, DFARRAY IFAC);
		
		void CFFTF(int &N, DFARRAY C, DFARRAY WSAVE);
		
		void CFFTF1(int &N, DFARRAY C, DFARRAY CH, DFARRAY WA, DFARRAY IFAC);
		
		void PASSF(int &NAC, int &IDO, int &IP, int &L1, int &IDL1, DFARRAY CC, DFARRAY C1, DFARRAY C2, DFARRAY CH,
		           DFARRAY CH2, DFARRAY WA);
		
		void PASSF2(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1);
		
		void PASSF3(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2);
		
		void PASSF4(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2, DFARRAY WA3);
		
		void PASSF5(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2, DFARRAY WA3, DFARRAY WA4);
		
		void CFFTB(int &N, DFARRAY C, DFARRAY WSAVE);
		
		void CFFTB1(int &N, DFARRAY C, DFARRAY CH, DFARRAY WA, DFARRAY IFAC);
		
		void PASSB(int &NAC, int &IDO, int &IP, int &L1, int &IDL1, DFARRAY CC, DFARRAY C1, DFARRAY C2, DFARRAY CH,
		           DFARRAY CH2, DFARRAY WA);
		
		void PASSB2(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1);
		
		void PASSB3(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2);
		
		void PASSB4(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2, DFARRAY WA3);
		
		void PASSB5(int &IDO, int &L1, DFARRAY CC, DFARRAY CH, DFARRAY WA1, DFARRAY WA2, DFARRAY WA3, DFARRAY WA4);
		
	} // FFTW
	
	
	void fft1d(std::valarray<std::complex<double>> &data, int isign = -1);
	
	void fft1d(CFARRAY data, int N, int isign = -1);
	
	void fft(const std::vector<double> &In, std::vector<std::complex<double>> &Out);
	
	void ifft(const std::vector<double> &In, std::vector<std::complex<double>> &Out);
	
	void fftfreq(std::vector<long double> &f, const long double d);
	
	size_t fftfreq_pos(std::vector<double> &f, const double d);
	
	double iamp2psd(std::complex<double> iamp, size_t n);
	
	void fft_pos(std::vector<double> &Out, const std::vector<double> &In, const size_t &Npos);
	
	double spc2amp(double dom, double spc);
	
	
} // UTILS


#endif //UTILS_FFT_H
