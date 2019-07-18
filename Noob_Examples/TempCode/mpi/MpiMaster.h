#pragma once

#ifndef UTILS_MPIMASTER_H
#define UTILS_MPIMASTER_H

#include <mpi.h>
#include <AData.h>
#include <ACalc.h>

#define ID -1
#define DATA 0
#define RESULT 1
#define FINISH 2

typedef int myint;

#ifdef USE_CUDA
typedef ACalcCuda_t ACalc_t;
#include "structs.cuh"
#endif

class MpiMaster {
private:

	MPI_Request *requestSend;
	MPI_Request *requestReceive;
	MPI_Status status;

	int cnt = -1;
	
	int requestId;
	int sentcnt = 0;
	int recvcnt = 0;

	int i;

	int myrank, proccnt;
	
	int slavesCnt[32];
	
	
	AData_t *oDataGen;
	ACalc_t *oResults;
	ADataIdemnItem_t oItemIdem;

public:


	MpiMaster (int myrank, int proccount, AData_t *oDataGen, ACalc_t *oResults);

	~MpiMaster();

	int run();

private:


	int init();

	int send_const();

	int send_idem();

	int isend_idem();

	int irecv_results();

	int run_while_waits();

	int run_while_isend_check_anylast();

	int run_while_icomm();

	int run_while();

	int after_while();

	int kill_slaves();

	int check_results();

	int recv_orphans();

};


#endif //UTILS_MPIMASTER_H
