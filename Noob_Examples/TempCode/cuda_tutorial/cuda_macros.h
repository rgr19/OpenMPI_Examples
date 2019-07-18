#pragma once


#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H


#if defined(USE_CUDA_DUMMY) || defined(USE_CUDA)

#include "cuda_dummy.h"


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


#define CUDA_CPMALDEV(harr, darr, T, n){                                                  \
       cudaError_t err = cudaMalloc( ( void** ) &darr, n* sizeof( T ) );                              \
        CUDA_ECHO_ERROR( err, "cudaMalloc the " #darr );                                         \
      cudaError_t  err = cudaMemcpy(darr, &harr, n*sizeof( T ), cudaMemcpyHostToDevice);                \
        CUDA_ECHO_ERROR( err, "cudaMemcpy from " #harr " to " #darr );                           \
        }

#define CUDA_FREEDEV(darr){                     \
       cudaError_t err = cudaFree(darr );               \
        CUDA_ECHO_ERROR( err, "cudaFree the " #darr );   \
}

#define CUDA_FREEHST(harr){                     \
       cudaError_t err = cudaFreeHost(harr );               \
        CUDA_ECHO_ERROR( err, "cudaFreeHost the " #harr );   \
}



#define CUDA_MALDEV(darr, T, n){                                                   \
       cudaError_t err = cudaMalloc( ( void** ) &darr, n* sizeof( T ) );                 \
        CUDA_ECHO_ERROR( err, "cudaMalloc the " #darr );                                          \
    }

#define CUDA_MALHST(harr, T, n){                                                               \
       cudaError_t err = cudaMallocHost( ( void** ) &harr, n* sizeof( T ) );                                 \
        CUDA_ECHO_ERROR( err, "cudaMallocHost the " #harr );                                                      \
    }

#define CUDA_CP2DEV(harr, darr, T, n){                                                               \
       cudaError_t err = cudaMemcpy(darr, harr, n*sizeof( T ), cudaMemcpyHostToDevice);                  \
        CUDA_ECHO_ERROR( err, "cudaMemcpy from " #harr " to " #darr );                                        \
    }

#define CUDA_CPD2D(darr1, darr2, T, n){   \
       cudaError_t err = cudaMemcpy(darr2, darr1, n*sizeof( T ), cudaMemcpyDeviceToDevice);  \
        CUDA_ECHO_ERROR( err, "cudaMemcpy from " #darr1 " to " #darr2 );    \
    }

#define CUDA_ACPD2D(darr1, darr2, T, n, stream){   \
       cudaError_t err = cudaMemcpyAsync(darr2, darr1, n*sizeof( T ), cudaMemcpyDeviceToDevice, stream);  \
        CUDA_ECHO_ERROR( err, "cudaMemcpyAsync from " #darr1 " to " #darr2 );    \
    }
    
#define CUDA_ACP2DEV(harr, darr, T, n, stream){                                                               \
       cudaError_t err = cudaMemcpyAsync(darr, harr, n*sizeof( T ), cudaMemcpyHostToDevice, stream);         \
        CUDA_ECHO_ERROR( err, "cudaMemcpyAsync from " #harr " to " #darr " on stream " #stream );                  \
    }

#define CUDA_STREAM(stream) {                                                                                       \
        cudaError_t err = cudaStreamCreate(&stream) ;                                                               \
        CUDA_ECHO_ERROR( err, "cudaStreamCreate : " #stream );                                        \
}

#define CUDA_CP2HST(darr, harr, T, n){                                                               \
       cudaError_t err = cudaMemcpy(harr, darr, n*sizeof( T ), cudaMemcpyDeviceToHost);                             \
        CUDA_ECHO_ERROR( err, "cudaMemcpy from " #darr " to " #harr );                                        \
    }

#define CUDA_ACP2HST(darr, harr, T, n, stream){                                                               \
       cudaError_t err = cudaMemcpyAsync(harr, darr, n*sizeof( T ), cudaMemcpyDeviceToHost, stream);                             \
        CUDA_ECHO_ERROR( err, "cudaMemcpyAsync from " #darr " to " #harr " on stream " #stream );                                        \
    }

#define CUDA_MEMSET(darr, v, T, n){                                                               \
       cudaError_t err = cudaMemset(darr, v, n * sizeof(T));                             \
        CUDA_ECHO_ERROR( err, "cudaMemset in " #darr );                                        \
    }

#define KER2(bpg, tpb) <<<bpg,tpb>>>
#define KER3(bpg, tpb, shd) <<<bpg,tpb, shd>>>
#define KER4(bpg, tpb, shd, stream) <<<bpg,tpb, shd, stream>>>


#define CUDA_KERNEL(kernel, bpg, tpb, ...){                     \
    kernel<<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
    }


#define CUDA_KERNEL_T(kernel, T, bpg, tpb, ...){                     \
    kernel<T><<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
}


#define CUDA_KERNEL_DYN(kernel, bpg, tpb, shd, ...){                     \
    kernel<<<bpg,tpb,shd>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                                 \
}

#define CUDA_KERNEL_STR(kernel, bpg, tpb, str, ...){                     \
    kernel<<<bpg,tpb,0,str>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_TT(kernel, T1, T2, bpg, tpb, ...){                     \
    kernel<T1,T2><<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_TTT(kernel, T1, T2, T3, bpg, tpb, ...){                     \
    kernel<T1,T2, T3><<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_TTT_STR(kernel, T1, T2, T3, bpg, tpb, str, ...){                     \
    kernel<T1,T2, T3><<<bpg,tpb,1,str>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }
 
#define CUDA_KERNEL_T_DYN(kernel, T, bpg, tpb, shd, ...){                     \
    kernel<T><<<bpg,tpb,shd>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_TT_DYN(kernel, T1, T2, bpg, tpb, shd, ...){                     \
    kernel<T1,T2><<<bpg,tpb,shd>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_DYN_STR(kernel, bpg, tpb, shd, str, ...){                     \
    kernel<<<bpg,tpb,shd, str>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }

#define CUDA_KERNEL_T_DYN_STR(kernel, T, bpg, tpb, shd, str, ...){                     \
    kernel<T><<<bpg,tpb,shd, str>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
 }


#define CUDA_START_TIMER(start, stop) {   \
    cudaEventCreate(&start);  \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
}

#define CUDA_END_TIMER(start, stop, function){   \
    float ms;  \
    cudaEventRecord(stop);  \
    cudaEventSynchronize(stop);  \
    cudaEventElapsedTime(&ms, start, stop);   \
    cudaEventDestroy(start);    \
    cudaEventDestroy(stop);   \
    printf("TIMEIT_CUDA : " #function " t[msec]: %f \n", ms );     \
}


#define CUDA_END_TIMER_ANS(start, stop, function, ans){   \
    float ms;  \
    cudaEventRecord(stop);  \
    cudaEventSynchronize(stop);  \
    cudaEventElapsedTime(&ms, start, stop);   \
    cudaEventDestroy(start);    \
    cudaEventDestroy(stop);   \
    printf("TIMEIT_CUDA : " #function " t[msec]: %f, ans: %f\n", ms, ans );     \
}


#define TIMEIT_CUDA(function, ...){                                                                           \
    cudaEvent_t start,stop;                                                                             \
    float ms;                                                                                  \
    cudaEventCreate(&start);                                                                             \
    cudaEventCreate(&stop);                                                                              \
    cudaEventRecord(start);                                                                               \
    function(__VA_ARGS__);                                                                  \
    cudaEventRecord(stop);                                                                                  \
    cudaEventSynchronize(stop);                                                                            \
    cudaEventElapsedTime(&ms, start, stop);                                                       \
    cudaEventDestroy(start);                                                                                \
    cudaEventDestroy(stop);                                                                                 \
    std::cout<< "TIMEIT_CUDA : " #function << "[microseconds]" << ms*1000 << std::endl;       \
}


#define TIMEIT_10_CUDA(function, ...){                                                                           \
    cudaEvent_t start,stop;                                                                             \
    float miliseconds;                                                                                  \
    cudaEventCreate(&start);                                                                             \
    cudaEventCreate(&stop);                                                                              \
    cudaEventRecord(start);                                                                               \
    fori(0,10) function(__VA_ARGS__);                                                                  \
    cudaEventRecord(stop);                                                                                  \
    cudaEventSynchronize(stop);                                                                            \
    cudaEventElapsedTime(&miliseconds, start, stop);                                                       \
    cudaEventDestroy(start);                                                                                \
    cudaEventDestroy(stop);                                                                                 \
    std::cout<< "TIMEIT_10_CUDA : " #function << "[microseconds]" << miliseconds*1000 << std::endl;       \
}

#define TIMEIT_100_CUDA(function, ...){                                                                           \
    cudaEvent_t start,stop;                                                                             \
    float miliseconds;                                                                                  \
    cudaEventCreate(&start);                                                                             \
    cudaEventCreate(&stop);                                                                              \
    cudaEventRecord(start);                                                                               \
    fori(0,100) function(__VA_ARGS__);                                                                  \
    cudaEventRecord(stop);                                                                                  \
    cudaEventSynchronize(stop);                                                                            \
    cudaEventElapsedTime(&miliseconds, start, stop);                                                       \
    cudaEventDestroy(start);                                                                                \
    cudaEventDestroy(stop);                                                                                 \
    std::cout<< "TIMEIT_100_CUDA : " #function << "[microseconds]" << miliseconds*1000 << std::endl;       \
}


#define TIMEIT_1000_CUDA(function, ...){                                                                   \
    cudaEvent_t start,stop;                                                                             \
    float miliseconds;                                                                                  \
    cudaEventCreate(&start);                                                                             \
    cudaEventCreate(&stop);                                                                              \
    cudaEventRecord(start);                                                                               \
    fori(0,1000) function(__VA_ARGS__);                                                                  \
    cudaEventRecord(stop);                                                                                  \
    cudaEventSynchronize(stop);                                                                            \
    cudaEventElapsedTime(&miliseconds, start, stop);                                                       \
    cudaEventDestroy(start);                                                                                \
    cudaEventDestroy(stop);                                                                                 \
    std::cout<< "TIMEIT_1000_CUDA : " #function << "[microseconds]" << miliseconds*1000 << std::endl;       \
}


//	TIMEIT_CUDA(CUDA_KERNEL_T_DYN, cuda_mul_AV, float, dim_grid, dim_block, sShared, sShared, nrows, ncols,
//	            dev_rand_data + ncols, dev_rand_data, dev_y);

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
    } catch (thrust::system_error & e) {                                              \
        std::cerr << "Error creating "  #var ": " << e.what() << std::endl;         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_HVEC(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::host_vector<T>(size, init);                                 \
    } catch (thrust::system_error & e) {                                              \
        std::cerr << "Error creating "  #var ": " << e.what() << std::endl;         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_HPVEC(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::hostpin_vector<T>(size, init);                                 \
    } catch (thrust::system_error & e) {                                              \
        std::cerr << "Error creating "  #var ": " << e.what() << std::endl;         \
        exit(-1);                                                                   \
    }                                                                               \
}

#define THRUST_DVEC(var, T, size, init) {                                           \
    try {                                                                           \
        (var) = thrust::device_vector<T>(size, init);                               \
    } catch (thrust::system_error & e) {                                              \
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


#define CUBLAS(kernel, handle, ...){                                               \
    cublasStatus_t error = kernel( handle, __VA_ARGS__);                                     \
    printf("Cublas Error : %s at [%s line %d]\n", CublasGetErrorString( error), __FILE__,__LINE__);                       \
}


#endif //USE CUDA

#endif //MYMPICUDATEST_CUDA_MACROS_H
