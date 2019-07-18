#pragma once


#ifndef MYMPICUDATEST_CUDA_KERNELS_CUH
#define MYMPICUDATEST_CUDA_KERNELS_CUH


#include "cuda_reduce.cuh"

#define BLOCK_SIZE 32


template<typename T>
__host__ __device__ int sign(T x) {
	return (x > 0) - (x < 0);
}


//TIMEIT_CUDA(CUDA_KERNEL_T_DYN, cuda_mul_AV, int, dim_grid, dim_block, sShared, sShared, nrows, ncols, dev_rand_data + ncols, dev_rand_data, dev_y);
__global__ void saxpy_gpu(size_t n, const int alpha, const int *vecX, int *vecY) {
	int i;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	vecY[i] = alpha * vecX[i] + vecY[i];
}

__host__ void saxpy_cpu(size_t n, const int alpha, const int *vecX, int *vecY) {
	
	for (int i = 0; i < n; i++) {
		vecY[i] = alpha * vecX[i] + vecY[i];
	}
}


template<cuint TILE_DIM>
__global__ void cuda_mul_AAT(float *a, float *c, int M) {
	cuint tidx = threadIdx.x;
	cuint tidy = threadIdx.y;
	cuint row = blockIdx.y * blockDim.y;
	cuint col = blockIdx.x * blockDim.x;
	
	__shared__ float aTile[TILE_DIM][TILE_DIM];
	__shared__ float aTileTrans[TILE_DIM][TILE_DIM];
	
	int rowid = row + tidy;
	int colid = col + tidx;
	float sum = 0.0f;
	
	aTile[tidy][tidx] = a[rowid * TILE_DIM + tidx];
	
	aTileTrans[tidx][tidy] = a[(col + tidy) * TILE_DIM + tidx];
	__syncthreads();
	
	for (int i = 0; i < TILE_DIM; i++) {
		sum += aTile[tidy][i] * aTileTrans[i][tidx];
	}
	
	c[rowid * M + colid] = sum;
}


template<typename T>
__global__ void cuda_mat_ediv(T *a, const T x) {
	
	cuint tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	a[tid] /= x;
	
};

template<typename T, cuint TILE_DIM>
__global__ void cuda_mul_VVT(T *a, T *c) {
	cuint tidx = threadIdx.x;
	cuint col = blockIdx.x * blockDim.x;
	
	__shared__ T aTile[TILE_DIM];
	
	int colid = col + tidx;
	
	aTile[tidx] = a[colid];
	
	__syncthreads();
	
	for (int i = 0; i < TILE_DIM; i++) {
		c[i * TILE_DIM + colid] += aTile[i] * aTile[colid];
	}
	
}

template<typename Tin, typename Tout, cuint TILE_DIM>
__global__ void cuda_upd_weights(const Tin *p, Tout *W, const Tout x, cuint P) {
	cuint tidx = threadIdx.x;
	cuint col = blockIdx.x * blockDim.x;
	
	
	__shared__ Tin aTile[TILE_DIM];
	
	//set W to zero
	for (int i = 0; i < TILE_DIM; i++) {
		W[i * TILE_DIM + col + tidx] = 0;
	}
	
	__syncthreads();
	
	#pragma unroll
	for (int j = 0; j < P; j++) {
		
		int colid = j*TILE_DIM + col + tidx;
		
		if (tidx < TILE_DIM)
			aTile[tidx] = p[colid];
		
		__syncthreads();
		
		#pragma unroll
		for (int i = 0; i < TILE_DIM; i++) {
			W[i * TILE_DIM + colid] += x * aTile[i] * aTile[tidx];
		}
		
	}
}

template<typename Tin, typename Tout, cuint TILE_DIM>
__global__ void cuda_mul_VVTx(const Tin *a, Tout *c, const Tout x) {
	cuint tidx = threadIdx.x;
	cuint col = blockIdx.x * blockDim.x;
	
	
	__shared__ Tin aTile[TILE_DIM];
	
	int colid = col + tidx;
	
	if (tidx < TILE_DIM)
		aTile[tidx] = a[colid];
	
	__syncthreads();
	
	#pragma unroll
	for (int i = 0; i < TILE_DIM; i++) {
		c[i * TILE_DIM + colid] += x * aTile[i] * aTile[colid];
	}
	
}

__global__ void cuda_hamming_dist(cuint N, const int *x, const int *y, int * d) {
	cuint tidx = threadIdx.x;
	cuint col = blockIdx.x * blockDim.x;
	
	
	extern __shared__ volatile int xTile[];
	extern __shared__ volatile int yTile[];
	
	int colid = col + tidx;
	
	if (tidx < N) {
		xTile[tidx] = x[colid];
		yTile[tidx] = y[colid];
	}
	
	__syncthreads();
	
	d[0] += xTile[colid] != yTile[colid];
	
}

__device__ cuint getGlobalIdx_2D_1D() {
	cuint blockId = blockIdx.y * gridDim.x + blockIdx.x;
	cuint threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

__device__ cuint getGlobalIdx_2D_2D() {
	cuint blockId = blockIdx.x + blockIdx.y * gridDim.x;
	cuint threadId = blockId * (blockDim.x * blockDim.y)
	                 + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ cuint getGlobalIdx_1D_2D() {
	return blockIdx.x * blockDim.x * blockDim.y
	       + threadIdx.y * blockDim.x + threadIdx.x;
}


__global__ void cuda_populate_mat_rows(cuint nRows, cuint nCols, const int *__restrict__ dx, int *__restrict__ dA) {
	
	//TODO
	cuint tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	fori(0, nRows) dA[tid + i * nCols] = dx[tid];
	
	
}

#endif