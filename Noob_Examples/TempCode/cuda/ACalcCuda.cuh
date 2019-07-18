#pragma once

#ifndef STRUCTS_CUH
#define STRUCTS_CUH

#include <cstring>

#include "AData.h"

#include "CudaUtils/containers/thrust_vector_2d.cuh"


#include "CudaUtils/cuda_dummy.h"


typedef struct ACalcCudaStep_t {
	
	int myrank = -1;
	int elemId = -1;
	double ans = 0;
	
	__host__ __device__ ACalcCudaStep_t & operator=(const ACalcCudaStep_t & a);
	
	__host__ __device__ ACalcCudaStep_t operator+(const ACalcCudaStep_t & a) const;
	
	__host__ __device__ ACalcCudaStep_t & operator+=(const ACalcCudaStep_t & a);
	
	__host__ __device__ void calc( const int i, const ACalcCudaStep_t *, const ADataIdemnItem *, const AData * );
	
} ACalcStep;


__host__ std::ostream & operator<<(std::ostream & os, const ACalcCudaStep_t & o);


typedef struct ACalcCuda_t {
	int myrank;
	const size_t sizeself = sizeof(ACalcCuda_t);
	const size_t sizeitem = sizeof(ACalcCudaStep_t);
	const size_t sizebase = sizeself
	                        - 5 * sizeof(vecSum)
	                        - sizeof(vecAns)
	                        - sizeof(d_vecConst);
	
	bool empty = true;
	
	size_t nIdem;
	size_t nConst;
	size_t nSteps;
	size_t size2d; //nConst * nSteps
	
	double t0, t1, dt;
	
	cudaError_t err = cudaSuccess;
	
	idemn size_t blocksPerGrid;
	const size_t threadsPerBlock = 256;
	
	thrust::host_vector2d<ACalcStep> vecAns;
	thrust::host_vector<ACalcStep> vecSum;
	thrust::host_vector<ACalcStep> vecAnsBlocks;
	thrust::host_vector<ACalcStep> vecAnsInit;
	thrust::device_vector<ACalcStep> d_vecAnsBlocks;
	thrust::device_vector<ACalcStep> d_vecAnsInit;
	thrust::device_vector<ADataConstItem> d_vecConst;
	
	
	inline int size_self() const { return (int) sizeself; };
	
	inline int size_item() const { return (int) sizeitem; };
	
	inline int size_base() const { return (int) sizebase; };
	
	inline int size_items() const { return static_cast<int>(sizeitem * nConst); };
	
	inline int size_items2d() const { return static_cast<int>(sizeitem * nConst * nSteps); };
	
	~ACalcCuda_t();
	
	ACalcCuda_t(int myrank, size_t nConst, size_t nSteps, size_t nIdem, double t0, double t1);
	
	
	void compute_cuda(const AData_t *, const ADataIdemnItem_t *, int myrank);
	
	
} AResultSystem;


__device__ void printElemDevice(int id, ACalcCudaStep_t & res);

__host__ void printElemHost(ACalcCudaStep_t & res);

__global__ void kernelPerstep(const int, const ADataIdemnItem_t *, const AData_t *, ACalcCudaStep_t *);


#define STRUCTS_ARGS                                    \
    AData_t *oDataGen,                             \
    ACalcCuda_t *oResults

#define STRUCTS_INIT                                    \
    oDataGen(oDataGen),                                   \
    oResults(oResults)


#endif //CUDA_TEST_STRUCTS_CUH
