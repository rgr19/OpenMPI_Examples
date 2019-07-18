#pragma once


#ifndef MYMPICUDATEST_CUDA_MUL_AV_H
#define MYMPICUDATEST_CUDA_MUL_AV_H

#include "cuda_kernels.cuh"

template<typename T1, typename T2, cuint SHM>
__global__ void cuda_mul_AV(cuint nRows, cuint nCols, const T1 *__restrict__ dA, const T2 *__restrict__ dx, T1 *__restrict__ dy) {
	cuint tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ volatile T2 x_shared[SHM];
	//__shared__ volatile int x_shared[BLOCK_SIZE];
	
	T1 y_val = 0;
	
	#pragma unroll
	for (uint m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
			x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
		else x_shared[threadIdx.x] = 0;
		__syncthreads();
		
		#pragma unroll
		for (uint e = 0; e < BLOCK_SIZE; ++e) {
			// --- Column-major ordering - faster
			y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
			// --- Row-major ordering - slower
			//y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		
		__syncthreads();
	}
	
	if (tid < nRows) dy[tid] = y_val;
	
}

template<typename T1, typename T2, cuint SHM>
__global__ void cuda_mul_AVsign(cuint nRows, cuint nCols, const T1 *__restrict__ dA, const T2 *__restrict__ dx, T1 *__restrict__ dy) {
	cuint tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ volatile T2 x_shared[SHM];
	//__shared__ volatile int x_shared[BLOCK_SIZE];
	
	T1 y_val = 0;
	
	#pragma unroll
	for (uint m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
			x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
		else x_shared[threadIdx.x] = 0;
		__syncthreads();
		
		#pragma unroll
		for (uint e = 0; e < BLOCK_SIZE; ++e) {
			// --- Column-major ordering - faster
			y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
			// --- Row-major ordering - slower
			//y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		
		__syncthreads();
	}
	
	if (tid < nRows) dy[tid] = sign<float>(y_val);
	
}



__global__ void cuda_mul_AVsign_fii(cuint SM, cuint nRows, cuint nCols, const float *__restrict__ dA, const int *__restrict__ dx, int *__restrict__ dy) {
	cuint tid = threadIdx.x;
	cuint gid = tid + blockIdx.x * blockDim.x;
	
	extern __shared__ volatile int x_shared[];
	//__shared__ volatile int x_shared[BLOCK_SIZE];
	
	float y_val = 0;
	
	cuint nBlc = ((nCols + SM - 1) / SM);
	uint colid;
	
	//printf("nBlc=%d, nRows=%d, nCols=%d \n", nBlc, nRows, nCols);
	
	#pragma unroll
	for (uint m = 0; m < nBlc; ++m) {
		colid = m * SM + tid;
		if (colid < nCols)
			x_shared[tid] = dx[colid];
		else
			x_shared[tid] = 0;
		
		__syncthreads();
		
		#pragma unroll
		for (uint e = 0; e < SM; ++e) {
			//printf("tid=%d, m=%d, e=%d, x[e]=%d, dA[...]=%f\n",
			//       gid, m, e, x_shared[e], dA[gid + (e + SM * m) * nRows]);
			
			// --- Column-major ordering - faster
			y_val += dA[gid + (e + SM * m) * nRows] * x_shared[e];
			
			// --- Row-major ordering - slower
			//y_val += dA[tid * nCols + (e + SM * m)] * x_shared[e];
		}
		
		__syncthreads();
	}
	
	if (tid < nRows) {
		//printf("tid=%d, yval=%f\n",tid,y_val);
		dy[tid] = sign<float>(y_val);
	}
	
}

__global__ void cuda_mul_AV_fif(cuint SM, cuint nRows, cuint nCols, const float *__restrict__ dA, const int *__restrict__ dx, float *__restrict__ dy) {
	cuint tid = threadIdx.x;
	cuint gid = tid + blockIdx.x * blockDim.x;
	
	extern __shared__ volatile int x_shared[];
	//__shared__ volatile int x_shared[BLOCK_SIZE];
	
	float y_val = 0;
	
	cuint nBlc = ((nCols + SM - 1) / SM);
	uint colid;
	
	#pragma unroll
	for (uint m = 0; m < nBlc; ++m) {
		colid = m * SM + tid;
		if (colid < nCols)
			x_shared[tid] = dx[colid];
		else
			x_shared[tid] = 0;
		
		__syncthreads();
		
		#pragma unroll
		for (uint e = 0; e < SM; ++e) {
			// --- Column-major ordering - faster
			y_val += dA[gid + (e + SM * m) * nRows] * x_shared[e];
			// --- Row-major ordering - slower
			//y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		
		__syncthreads();
	}
	
	if (tid < nRows) {
		dy[tid] = y_val;
	}
	
}



#endif //MYMPICUDATEST_CUDA_MUL_AV_H
