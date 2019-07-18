//
// Created by robgrzel on 05.06.17.
//

#ifndef CUDA_DUMMY_H
#define CUDA_DUMMY_H
#pragma once


#include <cstdio>

#ifdef USE_CUDA_DUMMY

/*! \brief Macros/inlines to assist CLion to parse Cuda files (*.cu, *.cuh) */
#ifdef __JETBRAINS_IDE__

#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__

#define __noinline__
#define __constant__
#define __managed__
#define __restrict__


// CUDA Synchronization
//inline void __syncthreads() { };

inline void __threadfence_block() {};

inline void __threadfence() {};

inline void __threadfence_system();

inline int __syncthreads_count(int predicate) { return predicate; };

inline int __syncthreads_and(int predicate) { return predicate; };

inline int __syncthreads_or(int predicate) { return predicate; };

template<class T>
inline T __clz(const T val) { return val; }

template<class T>
inline T __ldg(const T *address) { return *address; };
// CUDA TYPES
typedef unsigned short     uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;
typedef unsigned long      ulong;
typedef unsigned long long ulonglong;
typedef long long          longlong;

// CUDA TYPES
typedef unsigned short     uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;
typedef unsigned long      ulong;
typedef unsigned long long ulonglong;
typedef long long          longlong;


extern int warpsize;

struct __cuda_fake_struct {
    size_t x;
    size_t y;
};


//extern __cuda_fake_struct blockDim;
//extern __cuda_fake_struct threadIdx;
//extern __cuda_fake_struct blockIdx;


#endif //jetbrains


#endif //USE_CUDA

#if defined(USE_CUDA_DUMMY) || defined(USE_CUDA)

#include "std_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cuda_profiler_api.h>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef USE_CURAND
#include <curand.h>
#endif

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>

#define __CUDA_INTERNAL_COMPILATION__

#include <math_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#undef __CUDA_INTERNAL_COMPILATION__


#define MAX_BLOCK_DIM_SIZE 65535


typedef struct gpu_device_t {

    CUdevice dev       = -1;
    int      gpuCap[2] = {-1, -1};

} gpu_device;

int set_cuda(int myrank);


inline void gpuAssert(const cudaError_t &code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#define CUDA_ECHO_ERROR(e, strs) { \
    char a[255];                                                    \
	if (e != cudaSuccess) {                                         \
		strncpy(a, strs, 255);                                      \
		fprintf(stderr,                                                     \
		"CUDA Failed to run %s at file %s line %d :: errorCode : %s\n",     \
		a, __FILE__, __LINE__, cudaGetErrorString(e) );                     \
		exit(EXIT_FAILURE);                                                 \
	}                                                                       \
}

inline void cuda_get_device_prop(cudaDeviceProp & prop, int device){
		cudaGetDeviceProperties(&prop, device);
		printf("...Device Number: %d\n", device);
		printf("   Device name: %s\n", prop.name);
		printf("   Memory Clock Rate (KHz): %d\n",
		       prop.memoryClockRate);
		printf("   Memory Bus Width (bits): %d\n",
		       prop.memoryBusWidth);
		printf("   Peak Memory Bandwidth (GB/s): %f\n\n",
		       2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

inline cuint cuda_count_devices(){
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	
	printf("CUDA nDevices : %d\n", nDevices);

	return (cuint)(nDevices);
}

inline cuint init_cuda_gpu(cuint device){
	// Init GPU Device

	cuint nGpu = cuda_count_devices();

	if (!nGpu) {
		fprintf(stderr, "CUDA no devices ERROR, cant init device : %d\n", device);
		return 1;
	}

	printf("CUDA init device : %d\n", device);

	CUdevice dev;
	CUcontext context;
	cuInit(device);
	cuDeviceGet(& dev, device);
	cuCtxCreate(& context, device, dev);
	cudaError_t err = cudaSetDevice(device);
	CUDA_ECHO_ERROR(err, "init_cuda_gpu");



	return nGpu;

}

__global__ void dummy_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("dummy kernel i:%d\n", tid);

}

void call_dummy_kernel() {
    dummy_kernel << < 1, 32 >> > ();
}

#ifdef CURAND_H_

/*!
 * \brief Get string representation of cuRAND errors.
 * \param status The status.
 * \return String representation.
 */

inline const char *CurandGetErrorString(curandStatus_t status) {
    switch (status) {
        case CURAND_STATUS_SUCCESS:return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown cuRAND status";
}

#endif //CURAND_H_

#ifdef CUBLAS_V2_H_

inline const char *CublasGetErrorString(cublasStatus_t &error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:break;
    }
    return "Unknown cuBLAS status";
}

#endif //CUBLAS_V2_H_


#endif //use_cuda

#endif //CUDA_DUMMY_H
