#pragma once


#ifndef MYMPICUDATEST_CUDA_REDUCE_BENCHAMRK_CUH
#define MYMPICUDATEST_CUDA_REDUCE_BENCHAMRK_CUH


#include "cuda_reduce.cuh"


inline int cuda_reduce_tests() {


	const unsigned int THR_PER_BLC = 1024;
	const unsigned int BLC_PER_GRD = 8192;
	const unsigned int GRID_SIZE=  8192 * BLC_PER_GRD;
	
	int *data, *odata;
	int *g_idata;
	int *g_odata;
	
	double t1, t2, t3, t4;
	
	
	data = (int *) malloc(sizeof(int) * GRID_SIZE);
	odata = (int *) malloc(sizeof(int) * BLC_PER_GRD);
	
	cudaMalloc((void **) & g_idata, sizeof(int) * GRID_SIZE);
	cudaMalloc((void **) & g_odata, sizeof(int) * BLC_PER_GRD);
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;

//----------------------------------------------------------------------
	printf("#################################################################\n");
	printf("1\n");
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	reduce1i << < BLC_PER_GRD, THR_PER_BLC,THR_PER_BLC*sizeof(int)  >> > (g_idata, g_odata, GRID_SIZE);
	cudaThreadSynchronize();
	
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	int sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);

//----------------------------------------------------------------------
	printf("#################################################################\n");
	printf("2 No divergence \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	reduce2i << < BLC_PER_GRD, THR_PER_BLC ,THR_PER_BLC*sizeof(int) >> > (g_idata, g_odata, GRID_SIZE);
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);

//------------------------------------------------------------------------------
	printf("#################################################################\n");
	printf("3 Sequential Addressing \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	reduce3 << < BLC_PER_GRD, THR_PER_BLC,THR_PER_BLC*sizeof(int) >> > (g_idata, g_odata, GRID_SIZE);
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);

//------------------------------------------------------------------------------
	printf("#################################################################\n");
	printf("4 First add during load \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	reduce4 << < BLC_PER_GRD, THR_PER_BLC / 2,THR_PER_BLC*sizeof(int)  >> > (g_idata, g_odata, GRID_SIZE);
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);

//------------------------------------------------------------------------------
	printf("#################################################################\n");
	printf("5 Unroll the last warp \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	reduce5 << < BLC_PER_GRD, THR_PER_BLC / 2, THR_PER_BLC * sizeof(int) >> > (g_idata, g_odata, GRID_SIZE);
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);

//------------------------------------------------------------------------------
	printf("#################################################################\n");
	printf("6 Unroll the complete loop \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	cuda_reduce6_int(THR_PER_BLC, BLC_PER_GRD, GRID_SIZE, g_idata, g_odata);
	
	
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms)\n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);
	
	//------------------------------------------------------------------------------
	printf("#################################################################\n");
	
	printf("7 Final \n");
	
	for (int i = 0; i < GRID_SIZE; i++)
		data[i] = 1;
	
	
	cudaMemset(g_odata, 0, sizeof(int) * BLC_PER_GRD);
	
	t4 = wallclock();
	cudaMemcpy(g_idata, data, sizeof(int) * GRID_SIZE, cudaMemcpyHostToDevice);
	
	
	t1 = wallclock();
	
	cuda_reduce7i(THR_PER_BLC, BLC_PER_GRD, GRID_SIZE,  g_idata, g_odata);
	
	cudaThreadSynchronize();
	t3 = wallclock();
	
	cudaMemcpy(odata, g_odata, sizeof(int) * BLC_PER_GRD, cudaMemcpyDeviceToHost);
	
	printf("Kernal elapsed time = %10.3f(ms), band = \n", t3 - t1);
	
	t2 = wallclock();
	printf("Elapsed time = %10.3f(ms)\n", t2 - t4);
	
	sum = 0;
	for (int i = 0; i < BLC_PER_GRD; i++) {
		sum += odata[i];
//   printf("%d %d,,", i, odata[i]);
	}
	printf("Sum = %d, with BLC_PER_GRD %d THR_PER_BLC %d \n", sum, BLC_PER_GRD, THR_PER_BLC);
	
	//------------------------------------------------------------------------------
	printf("#################################################################\n");
	
	sum = 0;
	t1 = wallclock();
	for (int i = 0; i < GRID_SIZE; i++) {
		sum += data[i];
	}
	t2 = wallclock();
	printf("Sum = %d  \n", sum);
	printf("Series elapsed time = %10.3f(ms)\n", t2 - t1);
	
	return 0;
}


#endif