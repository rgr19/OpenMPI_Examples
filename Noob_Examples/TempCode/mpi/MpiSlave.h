#pragma once


#ifndef UTILS_MPI_SLAVE_H
#define UTILS_MPI_SLAVE_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <AData.h>
#include <ACalc.h>


#ifdef USE_CUDA
typedef ACalcCuda_t ACalc_t;
#include "structs.cuh"
#endif

class MpiSlave {
private:

	MPI_Request *requestSend;
	MPI_Request *requestReceive;
	MPI_Status status;


	int myrank;
	int proccount;

	int flag;
	int NonFinishFlag = 1;

	int prevItemId = 1;
	int currItemId = -1;
	int cnt = -1;


private:

	AData_t * oDataGen;
	ACalc_t * oResults;
	ADataIdemnItem_t oItemIdem[2];

public:


	MpiSlave( int myrank, int proccount,  AData_t *oDataGen, ACalc_t *oResults);

	int run();

private:


	int init();

	int recv_const();

	int recv_idem();

	int irecv_idem();

	int run_while();

	int run_while_calc();

	int run_while_isend();

	int run_while_irecv_tag_data();

	int run_while_irecv_tag_finish();


	bool run_while_irecv();


};


#endif //UTILS_MPI_SLAVE_H
