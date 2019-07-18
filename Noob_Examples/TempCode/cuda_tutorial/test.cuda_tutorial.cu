
//#include "cuda_reduce_benchmark.cuh"
//#include "hopfield_deterministic.cuh"

#ifdef USE_CUDA_DUMMY

#include "cuda_dummy.h"
//#include "std_utils.h"

#else
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
#endif


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>


#define BTOMB(x) (x)/1024/1024
#define MBTOB(x) (x)*1024*1024

#ifdef USE_CUDA_DUMMY
#define cuint const unsigned int
#define uint unsigned int
#else
typedef const unsigned int cuint;
typedef unsigned int uint;
#endif

#define fori(s, n) for(int i=s; i<n; ++i)
#define forj(s, n) for(int j=s; j<n; ++j)
#define fork(s, n) for(int k=s; k<n; ++k)
#define forl(s, n) for(int l=s; l<n; ++l)
#define forh(s, n) for(int h=s; h<n; ++h)
#define forauto(vec) for(auto & i : vec)

template<typename T>
T nextpow2(T x) {
	T n = x;
	--n;
	
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	
	return n + 1;
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

#define CUDA_KERNEL_DYN(kernel, bpg, tpb, shd, ...){                     \
    kernel<<<bpg,tpb,shd>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                                 \
}

#define CUDA_KERNEL(kernel, bpg, tpb, ...){                     \
    kernel<<<bpg,tpb>>>( __VA_ARGS__ );                     \
    cudaError_t err= cudaGetLastError();                   \
    CUDA_ECHO_ERROR(err, #kernel);                              \
    }


#define CUDA_MEMSET(darr, v, T, n){                                                               \
       cudaError_t err = cudaMemset(darr, v, n * sizeof(T));                             \
        CUDA_ECHO_ERROR( err, "cudaMemset in " #darr );                                        \
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


#define CUDA_CP2HST(darr, harr, T, n){                                                               \
       cudaError_t err = cudaMemcpy(harr, darr, n*sizeof( T ), cudaMemcpyDeviceToHost);                             \
        CUDA_ECHO_ERROR( err, "cudaMemcpy from " #darr " to " #harr );                                        \
    }

inline void cuda_get_device_prop(cudaDeviceProp &prop, int device) {
	cudaGetDeviceProperties(&prop, device);
	printf("...Device Number: %d\n", device);
	printf("   Device name: %s\n", prop.name);
	printf("   Memory Clock Rate (KHz): %d\n",
	       prop.memoryClockRate);
	printf("   Memory Bus Width (bits): %d\n",
	       prop.memoryBusWidth);
	printf("   Peak Memory Bandwidth (GB/s): %f\n\n",
	       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

inline cuint cuda_count_devices() {
	int nDevices;
	
	cudaGetDeviceCount(&nDevices);
	
	printf("CUDA nDevices : %d\n", nDevices);
	
	return (cuint) (nDevices);
}

inline cuint init_cuda_gpu(cuint device) {
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
	cuDeviceGet(&dev, device);
	cuCtxCreate(&context, device, dev);
	cudaError_t err = cudaSetDevice(device);
	CUDA_ECHO_ERROR(err, "init_cuda_gpu");
	
	
	return nGpu;
	
}

/*
EACH BLOCK IN GRID: contains one time step to compute error by
 W = P * P
 F = P[randIdx]
 S = W * F
 error = S != F
 EACH THREAD IN BLOCK: handles one pattern to compute weights Wp
 this weights are summed up for all patterns to one W matrix
 W += Wp
 
 
 
*/



#define N_N 100

template<typename Tin, typename Tout, cuint TILE_DIM>
__device__ void cuda_upd_weights(const Tin *p, volatile Tout *W, const Tout x) {
	cuint tidx = threadIdx.x;
	cuint col = blockIdx.x * blockDim.x;
	
	
	__shared__ Tin aTile[TILE_DIM];
	
	int colid = col + tidx;
	
	if (tidx < TILE_DIM)
		aTile[tidx] = p[colid];
	
	__syncthreads();
	
	#pragma unroll
	for (int i = 0; i < TILE_DIM; i++) {
		W[i * TILE_DIM + colid] += x * aTile[i] * aTile[colid];
	}
	
}


__global__ void mykernel(cuint nSteps, cuint nP, cuint nN, cuint caseIdx, const int *P,  float *errorB) {
	
	
	cuint stepIdx = blockIdx.x;
	cuint stepSize = blockDim.x * blockIdx.x;
	cuint patternIdx = threadIdx.x;
	
	
	if (patternIdx < nP && stepIdx < nSteps) {
		
		__shared__ volatile float s_W[N_N * N_N]; //weights updated from all patterns
		//extern __shared__ volatile int s_P[]; //each thread has his own pattern
		__shared__ volatile int s_F[N_N];  //we share same feed pattern
		__shared__ volatile float s_S[N_N];
		
		
		float Wij = 0;
		float _N = 1.f / nN;
		
		cuint idx = stepSize + patternIdx * nN;
		//compute W matrix from weights of all patterns
		
		#pragma unroll
		for (int i = 0; i < nN; ++i) {
			#pragma unroll
			for (int j = 0; j < nN; ++j) {
				s_W[i * nN + j] += P[idx + i] * P[idx + j] * _N;
			}
		}
		
		//__device__ void cuda_upd_weights(const Tin *p, Tout *W, const Tout x, cuint P)
		
		int feedIdx = 0; //random index from range [0, nP)
		if (patternIdx == feedIdx) {
			#pragma unroll
			for (int i = 0; i < nN; ++i) {
				s_F[i] = P[idx + i];
			}
		}
		__syncthreads();
		//compute state for given feed index, only once
		if (patternIdx == 0) {
			float Si;
			#pragma unroll
			for (int i = 0; i < nN; ++i) {
				Si = 0;
				#pragma unroll
				for (int j = 0; j < nN; ++j) {
					Si += s_W[i * nN + j] * s_F[i];
				}
				s_S[i] = Si;
			}
			#pragma unroll
			for (int i = 0; i < nN; ++i) {
				errorB[caseIdx] += s_S[i] != s_F[i];
			}
			
		}
		
	}
}

#define thr thrust

int main() {
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	uint deviceIdx = 0;
	cudaDeviceProp prop;
	
	cuint nGpu = init_cuda_gpu(deviceIdx);
	
	cudaProfilerStart();
	
	cuda_get_device_prop(prop, deviceIdx);
	
	
	int nN = N_N;
	int nP = 400;
	
	size_t maxGlobMem = BTOMB(prop.totalGlobalMem);
	size_t maxGlobMem_1p6 = nextpow2<size_t>(maxGlobMem * 1 / 6);
	size_t maxGlobMem_2p6 = 2 * maxGlobMem_1p6;
	size_t maxGlobMem_3p6 = 3 * maxGlobMem_1p6;
	size_t maxGlobMem_4p6 = 4 * maxGlobMem_1p6;
	size_t maxGlobMem_5p6 = 5 * maxGlobMem_1p6;
	
	uint maxThrPerBlc = (uint) prop.maxThreadsPerBlock;
	uint maxShm = (uint) prop.sharedMemPerBlock;
	int maxGridx = prop.maxGridSize[0];
	
	printf("globMem: %zu maxThrPerBlc: %d, shm: %d, gridsize: %d\n",
	       maxGlobMem, maxThrPerBlc, maxShm, maxGridx);
	
	size_t maxGlobFloat = maxGlobMem / sizeof(float);
	size_t maxGlobFloat_4p6 = maxGlobMem_4p6 / sizeof(float);
	size_t maxGlobFloat_3p6 = maxGlobMem_3p6 / sizeof(float);
	size_t maxGlobFloat_2p6 = maxGlobMem_2p6 / sizeof(float);
	size_t maxGlobFloat_1p6 = maxGlobMem_1p6 / sizeof(float);
	size_t maxGlobInt_1p6 = maxGlobMem_1p6 / sizeof(int);
	
	printf("maxGlobFloat_4p6: %zu, maxGlobFloat_3p6: %zu,   maxGlobFloat_2p6: %zu,  maxGlobFloat_1p6: %zu, maxGrid: %d\n",
	       maxGlobFloat_4p6, maxGlobFloat_3p6, maxGlobFloat_2p6, maxGlobFloat_1p6, maxGridx);
	
	size_t nSteps = (size_t) 1e5;
	
	size_t nPatternsPerTime = nSteps * nN;
	
	size_t nTotPatterns = nP * nPatternsPerTime;
	
	size_t nChunksP = BTOMB(nTotPatterns) / (size_t) (maxGlobFloat) + 1;
	size_t nChunksP_4p6 = BTOMB(nTotPatterns) / (size_t) (maxGlobFloat_4p6) + 1;
	size_t nChunksP_3p6 = BTOMB(nTotPatterns) / (size_t) (maxGlobFloat_3p6) + 1;
	size_t nChunksP_2p6 = BTOMB(nTotPatterns) / (size_t) (maxGlobFloat_2p6) + 1;
	size_t nChunksP_1p6 = BTOMB(nTotPatterns) / (size_t) (maxGlobFloat_1p6) + 1;
	
	printf("np2(1e5): %zu, nChunksP: %zu/%zu = %zu\n", nP * nSteps, BTOMB(nTotPatterns), maxGlobFloat, nChunksP);
	printf("np2(1e5): %zu, nChunksP_4p6: %zu/%zu = %zu\n", nP * nSteps, BTOMB(nTotPatterns), maxGlobFloat_4p6, nChunksP_4p6);
	printf("np2(1e5): %zu, nChunksP_3p6: %zu/%zu = %zu\n", nP * nSteps, BTOMB(nTotPatterns), maxGlobFloat_3p6, nChunksP_3p6);
	printf("np2(1e5): %zu, nChunksP_2p6: %zu/%zu = %zu\n", nP * nSteps, BTOMB(nTotPatterns), maxGlobFloat_2p6, nChunksP_2p6);
	printf("np2(1e5): %zu, nChunksP_1p6: %zu/%zu = %zu\n", nP * nSteps, BTOMB(nTotPatterns), maxGlobFloat_1p6, nChunksP_1p6);
	
	size_t nStepsChunk = nSteps / nChunksP_4p6;
	size_t sChunk = nStepsChunk * nP * nN;
	
	printf("nStepsChunk: %zu, sizePatSteps=%zu\n", nStepsChunk, sChunk);
	
	
	float *dev_ErrorB;
	float *hst_ErrorB;
	int *dev_P;
	
	CUDA_MALHST(hst_ErrorB, float, 100);
	CUDA_MALDEV(dev_ErrorB, float, 100);
	CUDA_MALDEV(dev_P, int, sChunk);
	
	CUDA_MEMSET(dev_P, 1, int, sChunk);
	
	dim3 blcPerGrd(nStepsChunk);
	dim3 thrPerBlc(nP);
	
	printf("sizeof(int)=%zu, sizeof(float)=%zu\n", sizeof(int), sizeof(float));
	
	printf("CALL KERNEL: blckPerGrd: %zu, thrPerBlc: %d, SHM: %zu/%zu\n",
	       nSteps, maxThrPerBlc, maxShm / sizeof(int), maxShm / sizeof(float));
	
	fori(0, nChunksP_4p6) {
		CUDA_KERNEL(mykernel,
		            blcPerGrd, thrPerBlc,
		            nStepsChunk, nP, nN, 0, dev_P, dev_ErrorB);
	}
	
	cudaDeviceSynchronize();
	
	CUDA_CP2HST(dev_ErrorB, hst_ErrorB, float, 100);
	
	
	fori(0, 100) printf("[%d] errorB: %f\n",i, hst_ErrorB[i]);
	
	
	cudaProfilerStop();
	
	
	CUDA_FREEDEV(dev_P);
	CUDA_FREEDEV(dev_ErrorB);
	
	return 0;
	
}

