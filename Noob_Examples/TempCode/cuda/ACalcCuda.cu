//
// Created by robgrzel on 04.06.17.
//
#include "ACalcCuda.cuh"



template<typename T>  __host__ __device__ void print_arr(T *a, int N) {
	printf("A[%d/%d]=%f\n", 0, N, a[0]);
	for (int i = 1; i < N; i++) {
		printf("A[%d/%d]=%f\n", i, N, a[i]);
		
	}
}


__host__ std::ostream & operator<<(std::ostream & os, const ACalcCudaStep_t & o) {
	os << o.myrank << ", " << o.elemId << ", " << o.ans;
	return os;
	
}

__host__ __device__ void print_id_ans(int id, const ACalcCudaStep_t *prev, const ACalcCudaStep_t *curr) {
	printf("CUDA dev calc: id(%d), ans(prev: %f, curr: %f)\n", id, prev->ans, curr->ans);
}


__host__ __device__  ACalcCudaStep_t & ACalcCudaStep_t::operator=(const ACalcCudaStep_t & a) {
	myrank = a.myrank;
	elemId = a.elemId;
	ans = a.ans;
	return * this;
}

__host__ __device__ ACalcCudaStep_t ACalcCudaStep_t::operator+(const ACalcCudaStep_t & a) const {//DEVICE
	return {myrank, elemId, ans + a.ans}; //TODO PROPER SUM OF WAVES
}

__host__ __device__ ACalcCudaStep_t & ACalcCudaStep_t::operator+=(const ACalcCudaStep_t & a) {
	
	ans += a.ans;
	return * this;
	
}

__host__ __device__ void ACalcCudaStep_t::calc(
		const int i, const ACalcCudaStep_t *arrAnsPrev,   //array
		const ADataConstItem *arrConst,            //array
		const ADataIdemnItem * itemIdem,             //element
		const ADataGen * dataGen                    //element
                      ) {
	//calculate ans and save in struct
	ans = 1 + arrAnsPrev->ans + arrConst->param  + i + dataGen->dt + itemIdem->param;
	
}


__global__ void kernelPerstep(
		const int i,
		const ADataConstItem_t * d_dataConst,
		const ADataIdemnItem_t * d_itemIdem,
		const ADataGen_t * d_dataGen,
		ACalcCudaStep_t *d_ans
                             ) {
	
	const int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	//take one item from h_x const
	const ADataConstItem_t *t_dataConst = &d_dataConst[id];
	
	//from global memory, save current ans as prev ans
	const ACalcCudaStep_t *t_ans = & d_ans[id];
	
	//calculate curr ans based on h_x const, idem and prev ans
	d_ans[id].calc(i, t_ans, t_dataConst, d_itemIdem, d_dataGen);
	
	if (id < 10) {
		print_id_ans(id, t_ans, & d_ans[id]);
	}
}

__global__ void kernelReduce(ACalcCudaStep_t *d_ans, ACalcCudaStep_t *d_ansBlock) {
	
	
	const size_t tidx = threadIdx.x;
	const size_t bdim = blockDim.x;//how many threads in one block
	const size_t idx = blockIdx.x * bdim + tidx;
	
	//DYNAMIC SHARED MEM REQUIRE THIRD PARAMETER OF BITS IN KERNEL CALL <<<A,B,BITS>>>
	
	extern __shared__ ACalcCudaStep_t t_ans[];
	
	t_ans[tidx] = d_ans[idx];
	
	__syncthreads();
	
	for (unsigned int s = 1; s < bdim; s *= 2) {
		if (tidx % (2 * s) == 0) {
			t_ans[tidx] += t_ans[tidx + s];
			
		}
		__syncthreads();
	}
	
	if (tidx == 0) {
		d_ansBlock[blockIdx.x] = t_ans[0];//this works, so it means the problem is overwrite operator+= doesn't
	}
}


void ACalcCuda_t::compute_cuda(
		const AData * data,
		const ADataIdemnItem & itemIdem,
		int myrank) {
	
	
	SDPPR2(myrank, "Copy dataConst.vec to CUDA device d_vecConst\n");
	
	THRUST_COPY(dataConst, d_vecConst);
	
	SDPPR2(myrank, "Copy vecAnsInit to CUDA device d_vecAnsInit\n");
	
	THRUST_COPY(vecAnsInit, d_vecAnsInit);
	
	std::cout <<itemIdem << std::endl;
	std::cout << dataGen << std::endl;
	
	//even if we copy one struct to kernel it has to be after cudamalloc and cudamemcpy
	ADataGen * d_dataGen;
	ADataIdemnItem * d_itemIdem;
	
	CUDA_CPMALDEV(itemIdem,d_itemIdem,ADataIdemnItem,1 );
	
	CUDA_CPMALDEV(dataGen,d_dataGen,ADataGen,1 );
	
	const ADataConstItem *d_dataConst = thrust::raw_pointer_cast(&d_vecConst[0]);
	
	idemn ACalcStep *d_ansBlocks = thrust::raw_pointer_cast(&d_vecAnsBlocks[0]);
	
	idemn ACalcStep *d_ansInit = thrust::raw_pointer_cast(&d_vecAnsInit[0]);
	
	SDPPR2(myrank, "CUDA run kernels (elemId: %d)\n", itemIdem.id);
	
	for (int k = 0; k < nSteps; k++) {
		SDPPR3(myrank, "CUDA run kernels for time step: %d (elemId: %d)\n", k, itemIdem.id);
		
		SDPPR3(myrank, "kernelPerstep launch (blocks:%zu, threads:%zu)\n", blocksPerGrid, threadsPerBlock);
		CUDA_KERNEL(kernelPerstep, blocksPerGrid, threadsPerBlock, k, d_dataConst, d_itemIdem, d_dataGen, d_ansInit);
		
		SDPPR3(myrank, "kernelReduce launch (blocks:%zu, threads:%zu)\n", blocksPerGrid, threadsPerBlock);
		CUDA_KERNEL_DYN(kernelReduce, blocksPerGrid, threadsPerBlock, sizeof(ACalcCudaStep_t) * nConst, d_ansInit, d_ansBlocks);
		
		SDPPR3(myrank, "thrust copy from d_vecAnsBlocks to vecAnsBlocks\n");
		THRUST_COPY(d_vecAnsBlocks, vecAnsBlocks)
		
		/*
		for ( int i = 0; i < blocksPerGrid; i++ ) {
			printf_result_and_id(i,vecAnsBlocks[i]);
			//vecAns[0][currItemId] += vecAnsBlocks[i];
			//vecAns[0][currItemId] += vecAnsBlocks[i]; //TODO : integration over waves : sum_j^N { A_j*dom_j}
		}
*/
	
	}
}


ACalcCuda_t::ACalcCuda_t(int myrank, size_t nConst_, size_t nSteps_, size_t nIdem_, double t0, double t1) :
		myrank(myrank), nIdem(nConst_), nSteps(nSteps_), nConst(nIdem_),
		size2d(nConst_ * nSteps_), t0(t0), t1(t1), dt((t1 - t0) / nSteps) {
	printf("ACalcCuda_t::ACalcCuda_t (rank=%d, %zu, %zu, %zu)\n", myrank, nSteps, nConst, nIdem);
	
	if (nIdem < 1) {
		fprintf(stderr, "ACalcCuda_t::ACalcCuda_t ERROR: nIdem < 1\n");
		exit(-1);
	}
	if (nSteps < 1) {
		fprintf(stderr, "ACalcCuda_t::ACalcCuda_t ERROR: nSteps < 1\n");
		exit(-1);
	}
	if (nConst < 1) {
		fprintf(stderr, "ACalcCuda_t::ACalcCuda_t ERROR: nConst < 1\n");
		exit(-1);
	}
	
	THRUST_HVEC2D(vecAns,ACalcStep_t, nConst, nSteps)
	
	if (myrank) {
		printf("ACalcCuda_t::ACalcCuda_t (rank>0 then allocate: %zu, %zu, %zu)\n", nSteps, nConst, nIdem);
		
		blocksPerGrid = (nIdem + threadsPerBlock - 1) / threadsPerBlock;
		
		if (blocksPerGrid < 1) {
			fprintf(stderr, "ACalcCuda_t::ACalcCuda_t ERROR: blocksPerGrid < 1\n");
			exit(-1);
		}
		
		ACalcCudaStep_t dummyAns = {-1, -1, 0};
		ADataConstItem_t dummyConst = {-1, 0};
		
		THRUST_HVEC(vecAnsInit, ACalcStep, nConst, dummyAns);
		THRUST_HVEC(vecAnsBlocks, ACalcStep, blocksPerGrid, dummyAns);
		
		THRUST_DVEC(d_vecConst, ADataConstItem_t, nConst, dummyConst);
		THRUST_DVEC(d_vecAnsInit, ACalcStep, nConst, dummyAns);
		THRUST_DVEC(d_vecAnsBlocks, ACalcStep, blocksPerGrid, dummyAns);
		
	}
	
}
