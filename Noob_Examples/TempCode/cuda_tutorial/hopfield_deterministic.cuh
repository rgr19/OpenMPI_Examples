#pragma once

#ifndef MYMPICUDATEST_HOPFIELD_DETERMINISTIC_H
#define MYMPICUDATEST_HOPFIELD_DETERMINISTIC_H

#include "cuda_mul_AV.cuh"

#include <curand.h>
#include <curand_kernel.h>


__device__ volatile cuint idx11() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

/* this GPU kernel function is used to initialize the random states */
//<<<256,1024>>> to avoid if...
__global__ void init_curand_states(cuint size, cuint seed, curandState_t *states) {
	cuint i = idx11();
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
	            i, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
	            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	            &states[i]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(cuint size, cuint maxRand, curandState_t *states, unsigned int *numbers) {
	/* curand works like rand - except that it takes a state as a parameter */
	cuint i = idx11();;
	if (i < size)
		numbers[i] = curand(&states[i]) % maxRand;
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void random_uniform(cuint size, curandState_t *states, float *numbers) {
	/* curand works like rand - except that it takes a state as a parameter */
	cuint i = idx11();;
	if (i < size)
		numbers[i] = curand_uniform(&states[i]);
}

__global__ void random_uniform_pattern(cuint size, curandState_t *states, int *pattern) {
	cuint i = idx11();;
	if (i < size)
		pattern[i] = sign<float>(curand_uniform(&states[i]) * 2 - 1);
	
}


int check_patterns(cuint N, cuint P, int *pattern, int check = 1) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
we will store a random state for every thread  */
	
	if (check) {
		
		int *cpu_nums;
		
		cuint size = N * P;
		
		CUDA_MALHST(cpu_nums, int, size);
		
		CUDA_CP2HST(pattern, cpu_nums, int, size);
		
		/* print them out */
		int pos = 0;
		int neg = 0;
		int zer = 0;
		printf("pattern {");
		
		for (int i = 0; i < size; i++) {
			if (check == 2) {
				if (i % 10 == 0) {
					printf("\n ");
				}
				printf("%d, ", cpu_nums[i]);
			}
			int num = cpu_nums[i];
			if (num > 0) pos++;
			elif (num == 0) zer++;
			else neg++;
		}
		printf("\n} --> ");
		
		printf("pos=%d, zer=%d, neg=%d\n", pos, zer, neg);
		
		CUDA_FREEHST(cpu_nums);
		
	}
	
}

__global__ void cuda_random_choice(curandState_t state, int A, int B, uint *out) {
	float randu_f = curand_uniform(&state);
	randu_f *= (B - A + 0.999999); // You should not use (B-A+1)*
	randu_f += A;
	out[0] = __float2uint_rn(randu_f);
	
}


int hopfield_deterministic() {
	
	#ifdef USE_CUDA_DUMMY
	cudaError_t err;
	#endif
	
	cuint seed = 1234ULL;
	
	cuint N = 200;
	
	cuint nPatterns = 21;
	cuint sPattern[] = {1,
	                    20, 40, 60, 80, 100,
	                    120, 140, 160, 180, 200,
	                    220, 240, 260, 280, 300,
	                    320, 340, 360, 380, 400};
	
	cuint totTime = (int) (1e5);
	cuint nSteps = totTime / N;
	
	float *dev_W, *hst_W;
	int *dev_errorBit, *dev_errorP, *hst_errorP;
	int *hst_errorBit;
	
	CUDA_MALDEV(dev_W, float, N * N);
	CUDA_MALHST(hst_W, float, N * N);
	CUDA_MALDEV(dev_errorP, int, 2 * nPatterns);
	CUDA_MALHST(hst_errorP, int, 2 * nPatterns);
	
	CUDA_MALDEV(dev_errorBit, int, nPatterns);
	CUDA_MALHST(hst_errorBit, int, nPatterns);
	
	CUDA_MEMSET(dev_errorBit, 0, int, nPatterns);
	
	
	curandState *dev_states;
	
	/* allocate space on the GPU for the random states */
	CUDA_MALDEV(dev_states, curandState_t, 256 * 512);
	
	/* invoke the GPU to initialize all of the random ]states */
	
	CUDA_KERNEL(init_curand_states, 256, 512, 256 * 512, seed, dev_states);
	
	int *hst_pattern;
	int *dev_pattern;
	int *dev_feed;
	int *hst_feed;
	int *dev_state;
	float *dev_statef;
	int *hst_state;
	float *hst_statef;
	
	fori(0, nPatterns) {
		
		cuint P = sPattern[i];
		
		cuint size = N * P;
		
		random_choice_t randChoice(seed, 0, P);
		
		
		CUDA_MALHST(hst_feed, int, N);
		CUDA_MALHST(hst_state, int, N);
		CUDA_MALHST(hst_statef, float, N);
		CUDA_MALHST(hst_pattern, int, size);
		CUDA_MALDEV(dev_pattern, int, size);
		CUDA_MALDEV(dev_state, int, N);
		CUDA_MALDEV(dev_statef, float, N);
		//CUDA_MALDEV(dev_feed, int, N);
		
		
		printf("############################### nP: %d\n", P);
		//*
		//fork(0, size) hst_pattern[k] = k;
		
		
		/*fork(0, size) {
			if (k % 15 == 0) printf("\n");
			printf("%4d, ", hst_pattern[k]);
		}
		printf("\n");
		*/
		//CUDA_CP2DEV(hst_pattern, dev_pattern, int, size);
		//*/
		
		
		cudaStream_t stream[P];
		fork (0, P) CUDA_STREAM(stream[k]);
		
		forj(0, 1e3) {
			
			int feedIdx = randChoice.gen();
			//printf("nP: %d, step: %d, feedIdx=%d\n", P, j, feedIdx);
			
			CUDA_MEMSET(dev_W, 0, float, N * N);
			
			/* invoke the kernel to get some random numbers */
			CUDA_KERNEL(random_uniform_pattern, N, P, size, dev_states, dev_pattern);
			
			if (0)
				check_patterns(N, P, dev_pattern, 1);
			
			dev_feed = dev_pattern + feedIdx * N;
			
			
			CUDA_KERNEL_TTT_STR(cuda_upd_weights,
			                    int, float, N, //template args
			                    dim3(1), dim3(N), stream[i], //device args
			                    dev_pattern, dev_W, 1.f / N, P); //kernel args
			
			if (0) {
				CUDA_CP2HST(dev_W, hst_W, float, N * N);
				
				
				forl(0, 0) {
					printf("W[%d] {", l);
					forh(0, 0) {
						printf("%.4f, ", hst_W[l * N + h]);
						if (h > 0 && h % 15 == 0) printf("\n");
					}
					printf("}\n");
				}
				printf("\n\n");
			}
			
			cuint SM = N;//MIN(BLOCK_SIZE, N);
			
			dim3 dimgrid((N + SM - 1) / SM);
			dim3 dimblck(SM);
			
			
			CUDA_KERNEL_DYN(cuda_mul_AVsign_fii,
			                dimgrid, dimblck, dimblck.x * sizeof(int),
			                dimblck.x, N, N, dev_W, dev_feed, dev_state);
			
			
			if (0) {
				CUDA_CP2HST(dev_state, hst_state, int, N);
				
				CUDA_CP2HST(dev_feed, hst_feed, int, N);
				
				printf("feed {");
				fork(0, 0) {
					if (k % 10 == 0) printf("\n");
					//printf("%f, ", hst_statef[k]);
					printf("%2.d., ", hst_feed[k]);
					
				}
				printf("}\n");
				
				printf("S {");
				fork(0, 0) {
					if (k % 10 == 0) printf("\n");
					//printf("%f, ", hst_statef[k]);
					printf("%2.d., ", hst_state[k]);
					
				}
				printf("\n}\n");
			}
			
			CUDA_KERNEL_DYN(cuda_hamming_dist,
			                1, N, N * sizeof(int),
			                N, dev_state, dev_feed, dev_errorBit + i);
			
			
		}
		
		CUDA_CP2HST(dev_errorBit, hst_errorBit, int, nPatterns);
		fork(0, P) cudaStreamDestroy(stream[k]);
		
		printf("error bit {");
		fork(0, nPatterns) {
			if (k % 10 == 0) printf("\n");
			printf("%2.d., ", hst_errorBit[k]);
			
		}
		printf("\n}\n");
		//	curandDestroyGenerator(curandGen);
		
		
		//CUDA_KERNEL()
		
	}
	
	
	#ifdef USE_CUDA
	return 0;
	
	#endif
	
	
	/* free the memory we allocated for the states and numbers */
	cudaFree(dev_states);
	
	
	int *hst_x, *hst_y, *rand_data, *dev_rand_data, *dev_x, *dev_y;
	
	
	cuint nRows = 8192;
	cuint nCols = 8192;
	cuint nTot = nCols * nRows;
	
	CUDA_MALHST(rand_data, int, nTot); //2048 x 4096
	CUDA_MALDEV(dev_rand_data, int, nTot);
	CUDA_MALHST(hst_x, int, nCols); //2048
	CUDA_MALDEV(dev_x, int, nCols);
	CUDA_MALHST(hst_y, int, nRows); //4096
	CUDA_MALDEV(dev_y, int, nRows);
	
	fori(0, nCols) hst_x[i] = 1;
	
	fori(0, nTot) rand_data[i] = 1;
	
	
	cudaEvent_t start, stop;
	
	int nStreams = 10;
	
	cudaStream_t stream[nStreams];
	
	
	fori (0, nStreams) CUDA_STREAM(stream[i]);
	
	CUDA_MEMSET(dev_y, 0, int, nRows);
	
	CUDA_START_TIMER(start, stop);
	
	CUDA_ACP2DEV(hst_x, dev_x, int, nCols, stream[0]);
	
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
		CUDA_ACP2DEV(&rand_data[offset], &dev_rand_data[offset], int, sStream, stream[i]);
		//CUDA_KERNEL_DYN_STR(cuda_mul_AV,
		//                    dim_grid, dim_block, thrPerBlc*sizeof(int), stream[i],
		//                    sStreamR, sStreamC, dev_rand_data + offset, dev_x, dev_y + offsetR);
		
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


#endif //MYMPICUDATEST_HOPFIELD_DETERMINISTIC_H
