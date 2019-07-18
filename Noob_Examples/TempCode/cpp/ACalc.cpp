//
// Created by user on 7/25/2018.
//

#include "ACalc.h"

template<typename T> void print_arr(T *a, int N) {
	printf("A[%d/%d]=%f\n", 0, N, a[0]);
	for (int i = 1; i < N; i++) {
		printf("A[%d/%d]=%f\n", i, N, a[i]);
		
	}
}


std::ostream & operator<<(std::ostream & os, const ACalcStep_t & o) {
	os << o.myrank << ", " << o.elemId << ", " << o.ans;
	return os;
	
}

void print_id_ans(int id, const ACalcStep_t *prev, const ACalcStep_t *curr) {
	printf("CPU host calc: id(%d), ans(prev: %f, curr: %f)\n", id, prev->ans, curr->ans);
}


void ACalcStep_t::calc(
		const int i,
		const ACalcStep_t *arrAnsPrev,   //array
		const ADataConstItem *arrConst,            //array
		const ADataIdemnItem *itemIdem,             //element
		const AData *dataGen                    //element
                      ) {
	//calculate ans and save in struct
	ans = 1 + arrAnsPrev->ans + arrConst->param + i + dataGen->dt + itemIdem->param;
	
	/* TODO: PERFORM COMPUTATION
	*/
	
	//DUMMY COMPUTATION
	for (int j=0; j<1e6; j++) int r = randint(1000,2000);
	
	
}


void kernelPerstep(
		const int id,
		const ADataConstItem_t *d_dataConst,
		const ADataIdemnItem_t *d_itemIdem,
		const AData_t *d_dataGen,
		ACalcStep_t *d_ans) {
	
	//take one item from h_x const
	const ADataConstItem_t *t_dataConst = & d_dataConst[id];
	
	//from global memory, save current ans as prev ans
	const ACalcStep_t *t_ans = & d_ans[id];
	
	//calculate curr ans based on h_x const, idem and prev ans
	d_ans[id].calc(id, t_ans, t_dataConst, d_itemIdem, d_dataGen);
	
	if (id < 10) {
		print_id_ans(id, t_ans, & d_ans[id]);
	}
}


void ACalc_t::compute(
		const AData *dataGen,
		const ADataIdemnItem *itemIdem,
		int myrank) {
	
	std::cout << itemIdem << std::endl;
	return ;
	
	int currItemId = itemIdem->id;
	
	//vecAns[currItemId].myrank = myrank;
	
	//even if we copy one struct to kernel it has to be after cudamalloc and cudamemcpy
	
	const ADataConstItem *dataConst = dataGen->dataC();
	
	idemn ACalcStep *ansInit = vecAnsInit.data();
	
	SDPPR2(myrank, "CPU run kernels (elemId: %d)\n", itemIdem->id);
	
	for (int k = 0; k < nSteps; k++) {
		SDPPR3(myrank, "CPU run compute for time step: %d (elemId: %d)\n", k, itemIdem->id);
		
		kernelPerstep(k, dataConst, itemIdem, dataGen, ansInit);
		
		//vecAns[currItemId] += ansInit;

		/*
		for ( int i = 0; i < blocksPerGrid; i++ ) {
			printf_result_and_id(i,vecAnsBlocks[i]);
			//vecAns[0][currItemId] += vecAnsBlocks[i];
			//vecAns[0][currItemId] += vecAnsBlocks[i]; //TODO : integration over waves : sum_j^N { A_j*dom_j}
		}
*/
		
	}
}

int ACalc_t::add_temp(int id){

};


ACalc_t::ACalc_t(int myrank, size_t nConst_, size_t nSteps_, size_t nIdem_, double t0, double t1) :
		myrank(myrank), nIdem(nConst_), nSteps(nSteps_), nConst(nIdem_),
		size2d(nConst_ * nSteps_), t0(t0), t1(t1), dt((t1 - t0) / nSteps) {
	printf("ACalc_t::ACalc_t (rank=%d, %zu, %zu, %zu)\n", myrank, nSteps, nConst, nIdem);
	
	if (nIdem < 1) {
		fprintf(stderr, "ACalc_t::ACalc_t ERROR: nIdem < 1\n");
		exit(-1);
	}
	if (nSteps < 1) {
		fprintf(stderr, "ACalc_t::ACalc_t ERROR: nSteps < 1\n");
		exit(-1);
	}
	if (nConst < 1) {
		fprintf(stderr, "ACalc_t::ACalc_t ERROR: nConst < 1\n");
		exit(-1);
	}
	
	if (vecAns.size() != nConst * nSteps) {
		fprintf(stderr, "ACalc_t::ACalc_t WARNING: CURR SIZE (vecAns.GRID_SIZE():%zu) DIFFER (nConst*nSteps:%zu)\n",
		        vecAns.size(), nConst * nSteps);
		
		vecAns = std::vector2d<ACalcStep_t>(nConst, nSteps);
		fprintf(stderr, "ACalc_t::ACalc_t WARNING: NEW SIZE (vecAns.GRID_SIZE():%zu) == (nConst*nSteps:%zu)\n",
		        vecAns.size(), nConst * nSteps);
	}
	
	if (myrank) {
		printf("ACalc_t::ACalc_t (rank>0 then allocate: %zu, %zu, %zu)\n", nSteps, nConst, nIdem);
		
		ACalcStep_t dummyAns = {-1, -1, 0};
		
		if (vecAnsInit.size() != nConst)
			std::vector<ACalcStep_t> vecAns(nConst, dummyAns);
		
	}
	
}
