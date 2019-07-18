#pragma once

#ifndef REDUCE_CUH_H
#define REDUCE_CUH_H

#include "cuda_macros.h"
#include "std_utils.h"

enum ReduceType {
	REDUCE_INT,
	REDUCE_FLOAT,
	REDUCE_DOUBLE
};

const char *getReduceTypeString(const ReduceType type) {
	switch (type) {
		case REDUCE_INT:
			return "int";
		case REDUCE_FLOAT:
			return "float";
		case REDUCE_DOUBLE:
			return "double";
		default:
			return "unknown";
	}
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads) {
	
	//get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	
	if (whichKernel < 3) {
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	} else {
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}
	
	if ((float) threads * blocks > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
		printf("n is too large, please choose a smaller number!\n");
	}
	
	if (blocks > prop.maxGridSize[0]) {
		printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
		       blocks, prop.maxGridSize[0], threads * 2, threads);
		
		blocks /= 2;
		threads *= 2;
	}
	
	if (whichKernel == 6) {
		blocks = MIN(maxBlocks, blocks);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size) {
	T sum = data[0];
	T c = (T) 0.0;
	
	for (int i = 1; i < size; i++) {
		T y = data[i] - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	
	return sum;
}



//	reduce1 << < blcPerGrd, thrPerBlc, thrPerBlc*sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce1i(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdatai[tid] = g_x[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdatai[tid] += sdatai[tid + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}
//	reduce1 << < blcPerGrd, thrPerBlc, thrPerBlc*sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce1f(float *g_A, const float *g_x, float *g_y, int g_size) {
	extern __shared__ volatile float sdataf[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdataf[tid] = g_x[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdataf[tid] += sdataf[tid + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdataf[0];
}

//	reduce2 << < blcPerGrd, thrPerBlc, thrPerBlc*sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce2i(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdatai[tid] = g_x[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		
		if (index < blockDim.x) {
			sdatai[index] += sdatai[index + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}
//	reduce2 << < blcPerGrd, thrPerBlc, thrPerBlc*sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce2f(float *g_A, const float *g_x, float *g_y, int g_size) {
	extern __shared__ volatile float sdataf[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdataf[tid] = g_x[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		
		if (index < blockDim.x) {
			sdataf[index] += sdataf[index + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdataf[0];
}
//	reduce3 << < blcPerGrd, thrPerBlc,thrPerBlc*sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce3(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdatai[tid] = g_x[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdatai[tid] += sdatai[tid + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}

//	reduce4 << < blcPerGrd, thrPerBlc / 2,thrPerBlc*sizeof(int)  >> > (g_x, g_y, grdSize);
__global__ void reduce4(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdatai[tid] = g_x[i] + g_x[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdatai[tid] += sdatai[tid + s];
		}
		
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}

//	reduce5 << < blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int) >> > (g_x, g_y, grdSize);
__global__ void reduce5(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdatai[tid] = g_x[i] + g_x[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdatai[tid] += sdatai[tid + s];
		}
		
		__syncthreads();
	}
	if (tid < 32) {
		sdatai[tid] += sdatai[tid + 32];
		sdatai[tid] += sdatai[tid + 16];
		sdatai[tid] += sdatai[tid + 8];
		sdatai[tid] += sdatai[tid + 4];
		sdatai[tid] += sdatai[tid + 2];
		sdatai[tid] += sdatai[tid + 1];
	}
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}

template<unsigned int blockSize>
__global__ void reduce6(const int *g_x, int *g_y, int g_size) {
	extern __shared__ volatile int sdatai[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdatai[tid] = g_x[i] + g_x[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	
	if (blockSize >= 512) {
		if (tid < 256) { sdatai[tid] += sdatai[tid + 256]; }
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) { sdatai[tid] += sdatai[tid + 128]; }
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) { sdatai[tid] += sdatai[tid + 64]; }
		__syncthreads();
	}
	if (tid < 32) {
		if (blockSize >= 64) sdatai[tid] += sdatai[tid + 32];
		if (blockSize >= 32) sdatai[tid] += sdatai[tid + 16];
		if (blockSize >= 16) sdatai[tid] += sdatai[tid + 8];
		if (blockSize >= 8) sdatai[tid] += sdatai[tid + 4];
		if (blockSize >= 4) sdatai[tid] += sdatai[tid + 2];
		if (blockSize >= 2) sdatai[tid] += sdatai[tid + 1];
	}
	
	// write result for this block to global mem
	if (tid == 0) g_y[blockIdx.x] = sdatai[0];
}

template<cuint nThreads>
__global__ void reduce7i_xy(const int *g_x, int *g_y, cuint n) {
	extern __shared__ volatile int sdatai[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (nThreads * 2) + threadIdx.x;
	unsigned int gridSize = nThreads * 2 * gridDim.x;
	sdatai[tid] = 0;
	while (i < n) {
		sdatai[tid] += g_x[i] * g_x[i] + g_x[i + nThreads] * g_x[i + nThreads];
		i += gridSize;
	}
	__syncthreads();
	// reduction in shared memory
	if (nThreads >= 512) {
		if (tid < 256) { sdatai[tid] += sdatai[tid + 256]; }
		__syncthreads();
	}
	if (nThreads >= 256) {
		if (tid < 128) { sdatai[tid] += sdatai[tid + 128]; }
		__syncthreads();
	}
	if (nThreads >= 128) {
		if (tid < 64) { sdatai[tid] += sdatai[tid + 64]; }
		__syncthreads();
	}
	if (tid < 32) {
		if (nThreads >= 64) sdatai[tid] += sdatai[tid + 32];
		if (nThreads >= 32) sdatai[tid] += sdatai[tid + 16];
		if (nThreads >= 16) sdatai[tid] += sdatai[tid + 8];
		if (nThreads >= 8) sdatai[tid] += sdatai[tid + 4];
		if (nThreads >= 4) sdatai[tid] += sdatai[tid + 2];
		if (nThreads >= 2) sdatai[tid] += sdatai[tid + 1];
		// transfer of the result to global memory
		if (tid == 0) g_y[blockIdx.x] = sdatai[0];
	}
}

template<cuint nThreads>
__global__ void hamming_distance(const int *g_A, const int *g_x, int *g_y, cuint n) {
	extern __shared__ volatile int sdatai[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (nThreads * 2) + threadIdx.x;
	unsigned int gridSize = nThreads * 2 * gridDim.x;
	sdatai[tid] = 0;
	while (i < n) {
		sdatai[tid] += (g_x[i] != g_A[i]) + (g_x[i + nThreads] != g_A[i + nThreads]);
		i += gridSize;
	}
	__syncthreads();
	// reduction in shared memory
	if (nThreads >= 512) {
		if (tid < 256) { sdatai[tid] += sdatai[tid + 256]; }
		__syncthreads();
	}
	if (nThreads >= 256) {
		if (tid < 128) { sdatai[tid] += sdatai[tid + 128]; }
		__syncthreads();
	}
	if (nThreads >= 128) {
		if (tid < 64) { sdatai[tid] += sdatai[tid + 64]; }
		__syncthreads();
	}
	if (tid < 32) {
		if (nThreads >= 64) sdatai[tid] += sdatai[tid + 32];
		if (nThreads >= 32) sdatai[tid] += sdatai[tid + 16];
		if (nThreads >= 16) sdatai[tid] += sdatai[tid + 8];
		if (nThreads >= 8) sdatai[tid] += sdatai[tid + 4];
		if (nThreads >= 4) sdatai[tid] += sdatai[tid + 2];
		if (nThreads >= 2) sdatai[tid] += sdatai[tid + 1];
		// transfer of the result to global memory
		if (tid == 0) g_y[blockIdx.x] = sdatai[0];
	}
	
	if (tid < n) printf("tid=%d, sdatai[tid]=%d, gA[tid]=%d, gx[tid]=%d\n",tid, sdatai[tid],g_A[tid], g_x[tid]);
	
}

template<cuint nThreads>
__global__ void reduce7i(const int *g_A, const int *g_x, int *g_y, cuint n) {
	extern __shared__ volatile int sdatai[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (nThreads * 2) + threadIdx.x;
	unsigned int gridSize = nThreads * 2 * gridDim.x;
	sdatai[tid] = 0;
	while (i < n) {
		sdatai[tid] += g_x[i] * g_A[i] + g_x[i + nThreads] * g_A[i + nThreads];
		i += gridSize;
	}
	__syncthreads();
	// reduction in shared memory
	if (nThreads >= 512) {
		if (tid < 256) { sdatai[tid] += sdatai[tid + 256]; }
		__syncthreads();
	}
	if (nThreads >= 256) {
		if (tid < 128) { sdatai[tid] += sdatai[tid + 128]; }
		__syncthreads();
	}
	if (nThreads >= 128) {
		if (tid < 64) { sdatai[tid] += sdatai[tid + 64]; }
		__syncthreads();
	}
	if (tid < 32) {
		if (nThreads >= 64) sdatai[tid] += sdatai[tid + 32];
		if (nThreads >= 32) sdatai[tid] += sdatai[tid + 16];
		if (nThreads >= 16) sdatai[tid] += sdatai[tid + 8];
		if (nThreads >= 8) sdatai[tid] += sdatai[tid + 4];
		if (nThreads >= 4) sdatai[tid] += sdatai[tid + 2];
		if (nThreads >= 2) sdatai[tid] += sdatai[tid + 1];
		// transfer of the result to global memory
		if (tid == 0) g_y[blockIdx.x] = sdatai[0];
	}
}

template<cuint nThreads>
__global__ void reduce7f(const float *g_A, const float *g_x, float *g_y, cuint n) {
	extern __shared__ volatile float sdataf[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (nThreads * 2) + threadIdx.x;
	unsigned int gridSize = nThreads * 2 * gridDim.x;
	sdataf[tid] = 0;
	while (i < n) {
		sdataf[tid] += g_x[i] * g_A[i] + g_x[i + nThreads] * g_A[i + nThreads];
		i += gridSize;
	}
	__syncthreads();
	// reduction in shared memory
	if (nThreads >= 512) {
		if (tid < 256) { sdataf[tid] += sdataf[tid + 256]; }
		__syncthreads();
	}
	if (nThreads >= 256) {
		if (tid < 128) { sdataf[tid] += sdataf[tid + 128]; }
		__syncthreads();
	}
	if (nThreads >= 128) {
		if (tid < 64) { sdataf[tid] += sdataf[tid + 64]; }
		__syncthreads();
	}
	if (tid < 32) {
		if (nThreads >= 64) sdataf[tid] += sdataf[tid + 32];
		if (nThreads >= 32) sdataf[tid] += sdataf[tid + 16];
		if (nThreads >= 16) sdataf[tid] += sdataf[tid + 8];
		if (nThreads >= 8) sdataf[tid] += sdataf[tid + 4];
		if (nThreads >= 4) sdataf[tid] += sdataf[tid + 2];
		if (nThreads >= 2) sdataf[tid] += sdataf[tid + 1];
		// transfer of the result to global memory
		if (tid == 0) g_y[blockIdx.x] = sdataf[0];
	}
}



inline int cuda_reduce6_int(cuint thrPerBlc, cuint blcPerGrd, cuint grdSize, const int *g_x, int *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint threads = thrPerBlc / 2;
	
	bool ispow2 = is_power_of_2(threads);
	
	if (!ispow2) return 1;
	
	switch (threads) {
		case 512: {
			CUDA_KERNEL_T_DYN(reduce6, 512, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN(reduce6, 256, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN(reduce6, 128, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN(reduce6, 64, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN(reduce6, 32, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN(reduce6, 16, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y,
			                  grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN(reduce6, 8, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN(reduce6, 4, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN(reduce6, 2, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN(reduce6, 1, blcPerGrd, thrPerBlc / 2, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		default:
			return 1;
	}
	
	return 0;
}

inline int cuda_reduce7i(cuint thrPerBlc, cuint blcPerGrd, cuint grdSize, const int *g_x, int *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint threads = thrPerBlc / 2;
	
	cuint gsize = blcPerGrd / 8;
	
	const bool isNotPow2 = !is_power_of_2(threads);
	
	switch (threads) {
		case 512: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 512, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 256, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 128, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 64, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 32, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 16, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 8, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 4, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 2, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN(reduce7i_xy, 1, gsize, threads, thrPerBlc * sizeof(int), g_x, g_y, grdSize);
			
			break;
		}
		default :
			return 1;
	}
	
	return 0;
}


inline int cuda_hamming_distance7i(cuint blcPerGrd, cuint thrPerBlc, cuint grdSize, int *g_A, const int *g_x, int *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint nThr = thrPerBlc / 2;
	
	cuint nBlc = MAX(1, blcPerGrd / 8);
	
	const bool isNotPow2 = !is_power_of_2(nThr);
	
	
	switch (nThr) {
		case 512: {
			CUDA_KERNEL_T_DYN(reduce7i, 512, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN(reduce7i, 256, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN(reduce7i, 128, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN(reduce7i, 64, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN(reduce7i, 32, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN(reduce7i, 16, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN(reduce7i, 8, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN(reduce7i, 4, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN(reduce7i, 2, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN(reduce7i, 1, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			
			break;
		}
		default :
			return 1;
	}
	
	return 0;
}

inline int cuda_reduce7i(cuint blcPerGrd, cuint thrPerBlc, cuint grdSize, int *g_A, const int *g_x, int *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint nThr = thrPerBlc / 2;
	
	cuint nBlc = MAX(1, blcPerGrd / 8);
	
	const bool isNotPow2 = !is_power_of_2(nThr);
	
	
	switch (nThr) {
		case 512: {
			CUDA_KERNEL_T_DYN(reduce7i, 512, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN(reduce7i, 256, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN(reduce7i, 128, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN(reduce7i, 64, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN(reduce7i, 32, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN(reduce7i, 16, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN(reduce7i, 8, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN(reduce7i, 4, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN(reduce7i, 2, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN(reduce7i, 1, nBlc, nThr, thrPerBlc * sizeof(int),  g_A, g_x, g_y, grdSize);
			
			break;
		}
		default :
			return 1;
	}
	
	return 0;
}

inline int cuda_async_reduce7i(cuint blcPerGrd, cuint thrPerBlc, cuint grdSize, cudaStream_t stream, int *g_A, const int *g_x, int *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint nThr = thrPerBlc / 2;
	
	cuint nBlc = blcPerGrd / 8;
	
	const bool isNotPow2 = !is_power_of_2(nThr);
	
	
	switch (nThr) {
		case 512: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 512, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 256, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 128, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 64, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 32, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 16, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 8, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 4, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 2, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN_STR(reduce7i, 1, nBlc, nThr, thrPerBlc * sizeof(int), stream, g_A, g_x, g_y, grdSize);
			
			break;
		}
		default :
			return 1;
	}
	
	return 0;
}

inline int cuda_async_reduce7f(cuint thrPerBlc, cuint blcPerGrd, cuint grdSize, cudaStream_t stream, float *g_A, const float *g_x, float *g_y) {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint threads = thrPerBlc / 2;
	
	cuint gsize = blcPerGrd / 8;
	
	const bool isNotPow2 = !is_power_of_2(threads);
	
	
	switch (threads) {
		case 1024: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 1024, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 512: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 512, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 256: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 256, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 128: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 128, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 64: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 64, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 32: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 32, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 16: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 16, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 8: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 8, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 4: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 4, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 2: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 2, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			break;
		}
		case 1: {
			CUDA_KERNEL_T_DYN_STR(reduce7f, 1, gsize, threads, thrPerBlc * sizeof(float), stream, g_A, g_x, g_y, grdSize);
			
			break;
		}
		default :
			return 1;
	}
	
	return 0;
}


#endif //MYMPICUDATEST_REDUCE_CUH_H
