//
// Created by robgrzel on 05.06.17.
//

#ifndef CUDA_DUMMY_H
#define CUDA_DUMMY_H

#ifdef USE_CUDA


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
inline void __syncthreads() { };

inline void __threadfence_block() { };

inline void __threadfence() { };

inline void __threadfence_system();

inline int __syncthreads_count(int predicate) { return predicate; };

inline int __syncthreads_and(int predicate) { return predicate; };

inline int __syncthreads_or(int predicate) { return predicate; };

template<class T>
inline T __clz(const T val) { return val; }

template<class T>
inline T __ldg(const T *address) { return * address; };
// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;

// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;


extern int warpsize;

struct __cuda_fake_struct {
	size_t x;
	size_t y;
};


//extern __cuda_fake_struct blockDim;
//extern __cuda_fake_struct threadIdx;
//extern __cuda_fake_struct blockIdx;
#endif


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdio>


#define __CUDA_INTERNAL_COMPILATION__

#include <math_functions.h>
#include <device_functions.h>

#undef __CUDA_INTERNAL_COMPILATION__


typedef struct gpu_device_t {
	
	CUdevice dev = -1;
	int gpuCap[2] = {-1, -1};
	
} gpu_device;

int set_cuda(int myrank);

void echoError(cudaError_t e, const char *strs);

void gpuAssert(cudaError_t code, const char *file, int line, bool abort);


#ifdef CURAND_H_

/*!
 * \brief Get string representation of cuRAND errors.
 * \param status The status.
 * \return String representation.
 */

const char *CurandGetErrorString(curandStatus_t status);

#endif

#ifdef CUBLAS_V2_H_

const char *CublasGetErrorString(cublasStatus_t error);

#endif

#endif  // USE_CUDA


#define CHECK_CU_RESULT(err, cufunc){                                     \
    if (err != CUDA_SUCCESS) {                                            \
        printf ("Error %d for CUDA Driver API function '%s'.\n",          \
                err, cufunc);                                             \
        exit(-1);                                                         \
    }                                                                     \
}


#define CHECK_CU_ERROR(err, strs) {                                        \
    if (err != cudaSuccess) {                                             \
        char a[255];                                                       \
        strncpy( a, strs, 255 );                                         \
        fprintf( stderr, "Failed to %s,errorCode %s",                      \
                    a, cudaGetErrorString( err ) );                       \
        exit( EXIT_FAILURE );                                              \
    }                                                                      \
}


/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    cudaError_t e = cudaGetLastError();                                      \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  }

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Protected cuBLAS call.
 * \param func Expression to call.
 *
 * It checks for cuBLAS errors after invocation of the expression.
 */
#define CUBLAS_CALL(func)                                       \
  {                                                             \
    cublasStatus_t e = (func);                                  \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS)                          \
        << "cuBLAS: " << CublasGetErrorString(e); \
  }

/*!
 * \brief Protected cuRAND call.
 * \param func Expression to call.
 *
 * It checks for cuRAND errors after invocation of the expression.
 */
#define CURAND_CALL(func) {                                     \
    curandStatus_t e = (func);                                  \
    CHECK_EQ(e, CURAND_STATUS_SUCCESS)                          \
        << "cuRAND: " << CurandGetErrorString(e); \
  }

#define CUDA_CPMALDEV(harr, darr, T, n){                                                               \
		err = cudaMalloc( ( void** ) &darr, n* sizeof( T ) );                                           \
        echoError( err, "cudaMalloc the " #darr );                                                      \
        err = cudaMemcpy(darr, &harr, n*sizeof( T ), cudaMemcpyHostToDevice);                             \
        echoError( err, "cudaMemcpy from " #harr " to " #darr );                                        \
        }
  
#define CUDA_MALDEV(harr, darr, T, n){                                                               \
		err = cudaMalloc( ( void** ) &darr, n* sizeof( T ) );                                           \
        echoError( err, "cudaMalloc the " #darr );                                                      \
    }
  
#define CUDA_CP2DEV(harr, darr, T, n){                                                               \
        err = cudaMemcpy(darr, &harr, n*sizeof( T ), cudaMemcpyHostToDevice);                             \
        echoError( err, "cudaMemcpy from " #harr " to " #darr );                                        \
    }

#define KER2(bpg, tpb) <<<bpg,tpb>>>
#define KER3(bpg, tpb, shd ) <<<bpg,tpb, shd>>>
#define KER4(bpg, tpb, shd, stream) <<<bpg,tpb, shd, stream>>>

#define CUDA_KERNEL(kernel, bpg, tpb, ...){                     \
    kernel<<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err = cudaGetLastError();                                    \
    echoError(err, #kernel);                              \
}

#define CUDA_KERNEL_DYN(kernel, bpg, tpb, shd, ...){                     \
    kernel<<<bpg,tpb,shd>>>( __VA_ARGS__ );                     \
    cudaError_t err = cudaGetLastError();                                    \
    echoError(err, #kernel);                              \
}

#define THRUST_COPY(vecSrc, vecDst){                                                \
    try {                                                                           \
        thrust::copy((vecSrc).begin(), (vecSrc).end(), (vecDst).begin());           \
    } catch (thrust::system_error e) {                                              \
        fprintf(stderr, "thrust::copy from %s to %s ERROR:", #vecSrc, #vecDst);     \
        std::cerr << e.what() << std::endl;                                         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_HVEC2D(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::host_vector2d<T>(size, init);                                 \
    } catch (thrust::system_error e) {                                              \
        std::cerr << "Error creating "  #var ": " << e.what() << std::endl;         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_HVEC(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::host_vector<T>(size, init);                                 \
    } catch (thrust::system_error e) {                                              \
        std::cerr << "Error creating "  #var ": " << e.what() << std::endl;         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_DVEC(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::device_vector<T>(size, init);                               \
    } catch (thrust::system_error e) {                                              \
        std::cerr << "Error creating "  #var  ": " << e.what() << std::endl;        \
        exit(-1);                                                                   \
    }                                                                               \
}

#ifdef USE_CUDNN

#include <cudnn.h>

#define CUDNN_CALL( func )                                                      \
  {                                                                           \
	cudnnStatus_t e = (func);                                                 \
	CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


// Overload atomicAdd to work for floats on all architectures
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// From CUDA Programming Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long* address_as_ull =                  // NOLINT(*)
	reinterpret_cast<unsigned long long*>(address);     // NOLINT(*)
  unsigned long long old = *address_as_ull;             // NOLINT(*)
  unsigned long long assumed;                           // NOLINT(*)

  do {
	assumed = old;
	old = atomicCAS(address_as_ull, assumed,
					__double_as_longlong(val +
					__longlong_as_double(assumed)));

	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#endif //USE_CUDNN


#ifdef USE_CUDA

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void echoError(cudaError_t e, const char *strs) {
	char a[255];
	if (e != cudaSuccess) {
		strncpy(a, strs, 255);
		fprintf(stderr, "Failed to %s,errorCode %s",
		        a, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}


inline int set_cuda(int myrank) {
	
	int gpuNum;
	
	unsigned int Flags = 0;
	
	CHECK_CU_RESULT(cuInit(Flags), "check cuInit");
	CHECK_CU_RESULT(cuDeviceGetCount(& gpuNum), "check cuDeviceGetCount");
	
	gpu_device gpuDevices[gpuNum];
	
	for (int i = 0; i < gpuNum; i++) {
		gpuDevices[i].dev = i;
		
		CHECK_CU_RESULT(cuDeviceComputeCapability(
				& gpuDevices[i].gpuCap[0], & gpuDevices[i].gpuCap[1], i),
		                "check cuDeviceComputeCapability");
		
		printf("myrank : %d, gpu info: gpuNum=%d, gpuId=%d, gpuCap(%d, %d)\n",
		       myrank, gpuNum, gpuDevices[i].dev, gpuDevices[i].gpuCap[0], gpuDevices[i].gpuCap[1]);
		
	}
	
	if (gpuNum == 1)
		cudaSetDevice(0);
	else if (gpuNum == 2) {
		if (myrank % 2) { cudaSetDevice(0); }
		else { cudaSetDevice(1); }
	}
	
	return gpuNum;
};


#ifdef CURAND_H_

/*!
 * \brief Get string representation of cuRAND errors.
 * \param status The status.
 * \return String representation.
 */

inline const char *CurandGetErrorString(curandStatus_t status) {
	switch (status) {
		case CURAND_STATUS_SUCCESS:
			return "CURAND_STATUS_SUCCESS";
		case CURAND_STATUS_VERSION_MISMATCH:
			return "CURAND_STATUS_VERSION_MISMATCH";
		case CURAND_STATUS_NOT_INITIALIZED:
			return "CURAND_STATUS_NOT_INITIALIZED";
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "CURAND_STATUS_ALLOCATION_FAILED";
		case CURAND_STATUS_TYPE_ERROR:
			return "CURAND_STATUS_TYPE_ERROR";
		case CURAND_STATUS_OUT_OF_RANGE:
			return "CURAND_STATUS_OUT_OF_RANGE";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "CURAND_STATUS_LAUNCH_FAILURE";
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "CURAND_STATUS_PREEXISTING_FAILURE";
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "CURAND_STATUS_INITIALIZATION_FAILED";
		case CURAND_STATUS_ARCH_MISMATCH:
			return "CURAND_STATUS_ARCH_MISMATCH";
		case CURAND_STATUS_INTERNAL_ERROR:
			return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "Unknown cuRAND status";
}

#endif

#ifdef CUBLAS_V2_H_

inline const char *CublasGetErrorString(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		default:
			break;
	}
	return "Unknown cuBLAS status";
}

#endif //CUBLAS_V2_H_

#endif //USE_CUDA

#endif //CUDA_DUMMY_H
