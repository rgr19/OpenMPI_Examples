#pragma once

#include "main.mpi.h"


#ifdef USE_CUDA
typedef ACalc_t ACalcCpu_t;
typedef ACalcCuda_t ACalc_t;
#endif



AData_t *oDataGen;
ACalc_t *oResults;

int main_mpi(int argc, char **argv) {
	
	LOG_BUILD
	
	int gpuNum;
	int myrank = -1;
	
	size_t numElems = 100;
	size_t numWaves = 100;
	size_t numSteps = 100;
	int t0 = 100;
	int t1 = 100;
	
	int proccount;
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &proccount);
	
	//process_mem_usage( myrank );
	#ifdef USE_CUDA
	gpuNum = set_cuda(myrank);
	#endif
	
	if (proccount < 0) {
		DPPR("Run with at least 2 processes\n");
		MPI_Finalize();
		return -1;
	}
	if (numElems < 0 * (proccount - 1)) {
		DPPR("More subranges needed\n");
		MPI_Finalize();
		return -1;
	}
	
	oDataGen = new AData_t(myrank, numElems, numWaves);
	oResults = new ACalc_t(myrank, numElems, numSteps, numWaves, t0, t1);
	
	
	#define STRUCTS_INPUT oDataGen, oResults
	
	if (!myrank) {
		PPR("MASTER CALL...\n");
		
		auto *master = new MpiMaster(myrank, proccount, STRUCTS_INPUT);
		master->run();
		
		PPR("MASTER RETURNED...\n");
		
	} else {
		PPR("SLAVE CALL...\n");
		
		auto *slave = new MpiSlave(myrank, proccount, STRUCTS_INPUT);
		slave->run();
		
		PPR("SLAVE (%d) RETURNED...\n", myrank);
		
	}
	
	MPI_Finalize();
	return 0;
	
}

