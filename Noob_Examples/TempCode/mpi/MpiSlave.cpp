#pragma once

#include "MpiSlave.h"

MpiSlave::MpiSlave(int myrank, int proccount, STRUCTS_ARGS) :
		myrank(myrank), proccount(proccount), STRUCTS_INIT {
	//SDPPR(myrank, "SLAVE (%d) INIT @@@ @@@\n", ++cnt);
	
	LOG_BUILD

}

int MpiSlave::run() {
	//SDPPR(myrank, "SLAVE (%d) RUN @@@ @@@\n", ++cnt);

	init();
	
	sleep(1);
	
	
	recv_const();
	
	recv_idem();
	
	irecv_idem();
	
	run_while();
	
	return 0;
	
}

int MpiSlave::init() {

	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) init START\n", ++cnt, currItemId, prevItemId);

	requestSend = (MPI_Request *) malloc(2 * sizeof(MPI_Request));
	requestReceive = (MPI_Request *) malloc(2 * sizeof(MPI_Request));

	for (int i = 0; i < 2; i++) {
		requestSend[i] = MPI_REQUEST_NULL;
	}

	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) init END\n", ++cnt,currItemId,prevItemId);
	return 0;
}


int MpiSlave::recv_const() {
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_const START\n", ++cnt,currItemId,prevItemId);
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_const 1 : MPI_Recv vec const (base, GRID_SIZE: %d) \n", ++cnt,currItemId,prevItemId,oDataGen->size_baseCI());
	MPI_Recv(oDataGen, oDataGen->size_baseCI(), MPI_BYTE, 0, DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_const 2 : MPI_Recv vec const (arr, GRID_SIZE: %d)\n", ++cnt,currItemId,prevItemId,oDataGen->size_itemsC());
	MPI_Recv(oDataGen->vecC.data(), oDataGen->size_itemsC(), MPI_BYTE, 0, DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_const END\n", ++cnt,currItemId,prevItemId);
	
	return 0;

}

int MpiSlave::recv_idem() {
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_idem START\n", ++cnt,currItemId,prevItemId);
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_idem : MPI_Recv vec idem \n", ++cnt, currItemId, prevItemId);

	MPI_Recv(&(oItemIdem[0]), oDataGen->size_itemI(), MPI_BYTE, 0, DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_idem : MPI_Recv vec idem (id: %d)\n", ++cnt,currItemId,prevItemId, oItemIdem[0].id);
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) recv_idem END\n", ++cnt,currItemId,prevItemId);
	
	return 0;

}

int MpiSlave::irecv_idem() {
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) irecv_idem START\n", ++cnt, currItemId, prevItemId);
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) irecv_idem : MPI_Irecv vec idem [0] \n", ++cnt, currItemId, prevItemId);

	MPI_Irecv(&(oItemIdem[1]), oDataGen->size_itemI(), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &requestReceive[0]);
	
	
	
	//SDPPR1(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) irecv_idem : MPI_Irecv vec idem [1] (id: %d)\n", ++cnt, currItemId,prevItemId, oItemIdem[1].id);

	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) irecv_idem END\n", ++cnt, currItemId, prevItemId);

	return 0;


}

int MpiSlave::run_while() {
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while START\n", ++cnt, currItemId, prevItemId);

	bool doBreak;
	
	
	do {
		prevItemId = currItemId;
		currItemId = oItemIdem[0].id;
		
		//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while 1 : \n", ++cnt, currItemId, prevItemId);

		if (currItemId != prevItemId) {
			run_while_calc();
			run_while_isend();
		}

		doBreak = run_while_irecv();
		
		if (doBreak){
			//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while : DO BREAK  \n", ++cnt, currItemId, prevItemId);
			break;
		}

	} while (oItemIdem[0].id != -1 or oItemIdem[1].id != -1);
	
	//SDPPR(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while END\n", ++cnt, currItemId, prevItemId);
	
	return 0;

}


int MpiSlave::run_while_calc() {
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_calc START\n", ++cnt, currItemId, prevItemId);
	
	/* TODO ***********************************************************************************************************
	 * TODO: calculate and put results to array
	 * TODO: ie. resultSlave[i] = SimpleIntegration( myBuffer[i], myBuffer[i] + RANGESIZE );
	 * TODO ***********************************************************************************************************/
	
	//SDPPR4(myrank,"...CUDA Results ::  blocksPerGrid=%zu, nConst=%d, nIdem=%zu, nSteps=%zu \n",oResults->blocksPerGrid, oResults->nConst, oResults->nIdem, oResults->nSteps);
	
	#ifdef USE_CUDA
	oResults->compute_cuda(oDataGen, &oItemIdem[0], myrank);
	#else
	oResults->compute(oDataGen, &oItemIdem[0], myrank);
	#endif
	
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_calc END\n", ++cnt, currItemId, prevItemId);
	
	return 0;

}

int MpiSlave::run_while_isend() {
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_isend START\n", ++cnt, currItemId, prevItemId);
	
	
	//SDPPR4(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_isend 2 : MPI_Waitall (request send START)\n",++cnt, currItemId, prevItemId);
	
	MPI_Waitall(1, (requestSend), MPI_STATUS_IGNORE);
	
	//SDPPR4(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_isend 3 : MPI_Waitall (request send END)\n", ++cnt, currItemId, prevItemId);
	
	
	//SDPPR4(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_isend 4 : MPI_Isend ITEM vecAns(id: %d, ans: %f, size2d: %d) \n", ++cnt, currItemId, prevItemId,oResults->vecAns[0].elemId, oResults->vecAns[0].ans, oResults->size_item() );
	
	MPI_Isend(oResults->vecAns.data(), oResults->size_item(), MPI_BYTE, 0, RESULT, MPI_COMM_WORLD, &(requestSend[0]));
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_isend END\n", ++cnt, currItemId, prevItemId );
	
	return 0;

}

bool MpiSlave::run_while_irecv() {
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv START\n", ++cnt, currItemId, prevItemId);
	
	//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : MPI_Test (reqRecv: %d, FLAG: %d, TAG: %d)\n",++cnt, currItemId, prevItemId, requestReceive[0], flag, status);
	
	MPI_Test(&requestReceive[0], &flag, &status);
	
	//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : MPI_Test (reqRecv: %d, FLAG: %d, TAG: %d)\n",++cnt, currItemId, prevItemId, requestReceive[0], flag, status);
	
	
	//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : MPI_Test : flag == 1 \n",++cnt, currItemId, prevItemId);
	
	if (flag == 1) {//means we have a new vec to receive,NoneFinishFlag means should I stop receive vec
		//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : CHECK FLAG : flag == 1 \n",++cnt, currItemId, prevItemId);
		
		//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : MPI_WAIT (requestReceive[0] START) 2 \n",++cnt, currItemId, prevItemId);
		MPI_Wait(&requestReceive[0], MPI_STATUS_IGNORE);
		//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : MPI_WAIT (requestReceive[0] END) 2 \n",++cnt, currItemId, prevItemId);
		
		//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : IRecved : idem item (id: %d, param: %f)\n",++cnt, currItemId, prevItemId, oItemIdem[1].id, oItemIdem[1].param);
		
		
		//SDPPR4(myrank, "(%d) item id (C: %d, NPATTERNS: %d) run_while_irecv : CHECK STATUS : status.MPI_TAG (%d) \n", ++cnt, currItemId, prevItemId, status.MPI_TAG);
		
		if (status.MPI_TAG == DATA) {
			
			run_while_irecv_tag_data();
			
		} else if (status.MPI_TAG == FINISH) {
			
			run_while_irecv_tag_finish();
			
			//SDPPR2(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv END : DO BREAK \n",++cnt, currItemId, prevItemId);
			
			return true;
		}
	}
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv END : DONT BREAK \n", ++cnt, currItemId, prevItemId);
	
	
	return false;
}

int MpiSlave::run_while_irecv_tag_data() {
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv_tag_data START : status.MPI_TAG == DATA \n",++cnt, currItemId, prevItemId);
	
	oItemIdem[0] = oItemIdem[1];
	
	//SDPPR4(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv_tag_data 2 : MPI_Irecv (size2d item: %d) \n",++cnt, currItemId, prevItemId, oDataGen->size_itemI());
	
	MPI_Irecv(&(oItemIdem[1]), oDataGen->size_itemI(), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &requestReceive[0]);
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv_tag_data END : WHILE item (id[0]: %d, id[1]: %d) \n",++cnt, currItemId, prevItemId, oItemIdem[0].id, oItemIdem[1].id);
	
	

}

int MpiSlave::run_while_irecv_tag_finish() {
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv_tag_finish START : status.MPI_TAG == FINISH \n", ++cnt, currItemId, prevItemId);
	
	//SDPPR3(myrank, "(%d ) item id (C: %d, NPATTERNS: %d) run_while_irecv_tag_finish END : FINALIZE \n", ++cnt, currItemId, prevItemId);
	

	return 0;

}


