#pragma once


#ifndef MYMPICUDATEST_CUDA_TESTS_H
#define MYMPICUDATEST_CUDA_TESTS_H


#include "cuda_macros.h"
#include "cuda_reduce.cuh"
#include "cuda_kernels.cuh"

int test_parallel_reduce() {
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	int *hst_x, *hst_y;
	int *rand_data;
	
	int *dev_rand_data = NULL; // Random h_x will be allocated here!
	int *dev_x, *dev_xx, *dev_y = NULL;
	
	cuint nRows = 8192;
	cuint nCols = 8192;
	cuint nTot = nCols * nRows;
	
	CUDA_MALHST(rand_data, int, nTot); //2048 x 4096
	CUDA_MALDEV(dev_rand_data, int, nTot);
	CUDA_MALHST(hst_x, int, nCols); //2048
	CUDA_MALDEV(dev_x, int, nCols);
	CUDA_MALHST(hst_y, int, nTot); //4096
	CUDA_MALDEV(dev_y, int, nTot);
	
	CUDA_MALDEV(dev_xx, int, nTot);
	
	// rows, SM         SM          rows
	//( 4096, 2048) * (2048, 1) = (4096)
	
	fori(0, nCols) hst_x[i] = 1;
	
	fori(0, nTot) rand_data[i] = 1;
	
	cuint thrPerBlc = BLOCK_SIZE;
	
	uint blcPerGrd = MAX(8, (nTot + thrPerBlc - 1) / thrPerBlc);
	
	uint nStreams = 1;
	
	cudaEvent_t start, stop;
	
	double sum = 0;
	
	cudaStream_t stream[nStreams];
	
	
	fori(0, nStreams) CUDA_STREAM(stream[i]);
	
	CUDA_ACP2DEV(hst_x, dev_x, int, nCols, stream[0]);
	
	
	
	dim3 dim_grid(nCols/1024);
	dim3 dim_block(1024);
	
	CUDA_START_TIMER(start, stop);
	
	CUDA_ACP2DEV(rand_data, dev_rand_data, int, nTot, stream[nStreams - 1]);
	
	//TODO bad indexing
	CUDA_KERNEL_STR(cuda_populate_mat_rows, dim_grid, dim_block, stream[nStreams - 1], nRows, nCols, dev_x, dev_xx);
	
	cuda_async_reduce7i(MIN(nRows,thrPerBlc), thrPerBlc, nTot, stream[nStreams - 1], dev_rand_data, dev_xx, dev_y);
	
	CUDA_CP2HST(dev_y, hst_y, int, nRows);
	
	
	fori(0, nStreams) cudaStreamSynchronize(stream[i]);
	
	fori(0, nStreams) if (cudaStreamQuery(stream[i])) std::cout << cudaStreamQuery(stream[i]) << std::endl;
	
	
	fori(0, nRows) sum += hst_y[i];
	fori(0, 5) printf("[i:%d] = %d\n", i, hst_y[i]);
	//fori(0, nRows) if (hst_y[i]) printf("[i:%d] = %d\n", i, hst_y[i]);
	fori(nRows - 5, nRows) printf("[i:%d] = %d\n", i, hst_y[i]);
	
	CUDA_END_TIMER_ANS(start, stop, cuda_async_reduce7i, sum);
	
	
	CUDA_FREEHST(hst_x);
	CUDA_FREEHST(hst_y);
	CUDA_FREEHST(rand_data);
	CUDA_FREEDEV(dev_rand_data);
	CUDA_FREEDEV(dev_y);
	CUDA_FREEDEV(dev_x);
	CUDA_FREEDEV(dev_xx);
	
	fori(0, nStreams) cudaStreamDestroy(stream[i]);
	
	return 0;
	
	
}

int test_matvec() {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	int *hst_x, *hst_y;
	int *rand_data;
	
	int *dev_rand_data = NULL; // Random h_x will be allocated here!
	int *dev_x, *dev_y = NULL;
	
	cuint nRows = 8192;
	cuint nCols = 8192;
	cuint nTot = nCols * nRows;
	
	CUDA_MALHST(rand_data, int, nTot); //2048 x 4096
	CUDA_MALDEV(dev_rand_data, int, nTot);
	CUDA_MALHST(hst_x, int, nCols); //2048
	CUDA_MALDEV(dev_x, int, nCols);
	CUDA_MALHST(hst_y, int, nRows); //4096
	CUDA_MALDEV(dev_y, int, nRows);
	
	// rows, SM         SM          rows
	//( 4096, 2048) * (2048, 1) = (4096)
	
	fori(0, nCols) hst_x[i] = 1;
	
	fori(0, nTot) rand_data[i] = 1;
	
	
	cudaEvent_t start, stop;
	
	int nStreams = 1;
	
	cudaStream_t stream[nStreams];
	
	
	fori (0, nStreams) CUDA_STREAM(stream[i]);
	
	CUDA_MEMSET(dev_y, 0, int, nRows);
	
	CUDA_START_TIMER(start, stop);
	
	
	int sStream = nTot / nStreams;
	int sStreamC = nCols;
	int sStreamR = nRows / nStreams;
	
	cuint thrPerBlc = MIN(nCols, BLOCK_SIZE);
	
	uint blcPerGrd = (nCols + thrPerBlc - 1) / thrPerBlc;
	
	//blocksPerGrid, threadsPerBlock,
	dim3 dim_grid(blcPerGrd);
	dim3 dim_block(thrPerBlc);
	
	
	fori (0, nStreams) {
		int offset = i * sStream;
		int offsetR = i * sStreamR;
		if (i==0) CUDA_ACP2DEV(hst_x, dev_x, int, nCols, stream[i]);
		CUDA_ACP2DEV(&rand_data[offset], &dev_rand_data[offset], int, sStream, stream[i]);
		CUDA_KERNEL_DYN_STR(cuda_mul_AV, dim_grid, dim_block, thrPerBlc * sizeof(int), stream[i],
		                    sStreamR, sStreamC, dev_rand_data + offset, dev_x, dev_y + offsetR);
	}
	
	CUDA_CP2HST(dev_y, hst_y, int, nRows);
	
	fori(0, nStreams) cudaStreamSynchronize(stream[i]);
	
	
	fori(0, nStreams) std::cout << cudaStreamQuery(stream[i]) << std::endl;
	
	
	double sum = 0;
	
	fori(0, nRows) sum += hst_y[i];
	
	fori(0, 5) printf("[i:%d] = %d\n", i, hst_y[i]);
	fori(nRows - 5, nRows) printf("[i:%d] = %d\n", i, hst_y[i]);
	
	CUDA_END_TIMER_ANS(start, stop, cuda_mul_AV, sum);
	
	
	CUDA_FREEHST(hst_x);
	CUDA_FREEHST(hst_y);
	CUDA_FREEHST(rand_data);
	CUDA_FREEDEV(dev_rand_data);
	CUDA_FREEDEV(dev_y);
	CUDA_FREEDEV(dev_x);
	
	fori(i, nStreams) cudaStreamDestroy(stream[i]);
	
	
}



#endif //MYMPICUDATEST_CUDA_TESTS_H
