#pragma once


#include "MpiMaster.h"


MpiMaster::~MpiMaster() {
	//delete oItemIdem->vec;
	//delete oDataConst->vec;
	//oResults=NULL;
	//oDataGen=NULL;
}

MpiMaster::MpiMaster(int myrank, int proccount, STRUCTS_ARGS) :
		myrank(myrank), proccnt(proccount), STRUCTS_INIT {
	
	LOG_BUILD
	
	memset(slavesCnt, 32, 0);
	
}

int MpiMaster::run() {
	//MDPPR("START ### ###\n");
	
	if (!oResults->vecAns.size()) {
		//MDPPR("Not enough memory");
		MPI_Finalize();
		return -1;
		
	} else {
		
		int status;
		
		status = init();
		
		sleep(1);
		
		status = send_const();
		
		
		status = send_idem();
		
		status = isend_idem();
		
		status = irecv_results();
		
		
		status = run_while();
		
		status = after_while();
		
		//status = check_results();
		
		status = kill_slaves();
		
		
		//MDPPR("FINALIZE... (left to run: check orphans and vecAns)\n");
		
		//status = check_results();
		
		status = recv_orphans();
		
		status = check_results();
		
		//MDPPR("END...\n");
		
		
		return 0;
		
	}
}

int MpiMaster::init() {
	
	//MDPPR("(%d) init 0\n", ++cnt);
	
	
	requestSend = (MPI_Request *) malloc((proccnt - 1) * sizeof(MPI_Request));
	
	requestReceive = (MPI_Request *) malloc((proccnt - 1) * sizeof(MPI_Request));
	
	for (int i = 0; i < (proccnt - 1); i++) {
		requestReceive[i] = MPI_REQUEST_NULL;
	}
	
	//MDPPR1("(%d) init gen vec\n", ++cnt);
	
	/* TODO ***********************************************************************************************************
	 * TODO vec need to be generated to be send
	 * TODO ie. constant vec of wave forces that are the same for all nodes with milions of time steps
	 * TODO ie. idem vec of elements that we split structure into, where:
	 * TODO ... each element require separate computation of forces acting on it from waves
	 * TODO ... then these forces of all elements will be integrated to see forces acting on structure as whole
	 * TODO **********************************************************************************************************/
	
	oDataGen->genC();
	oDataGen->genI();
	
	//MDPPR("(%d) init end\n", ++cnt);
	
	return 0;
	
}


int MpiMaster::send_const() {
	
	//MDPPR("(%d) send_const 0\n", ++cnt);
	
	for (int i = 1; i < proccnt; i++) {
		//MDPPR1("(%d) to SLAVE (%d) send_const 1 : vec const (GRID_SIZE base: %d) \n", ++cnt, i, oDataGen->size_baseCI());
		MPI_Send(oDataGen, oDataGen->size_baseCI(), MPI_BYTE, i, DATA, MPI_COMM_WORLD);
		
		//MDPPR1("(%d) to SLAVE (%d) send_const 2 : vec const GRID_SIZE(items: %zu, bytes: %d)\n", ++cnt, i, oDataGen->vecC.GRID_SIZE(), oDataGen->size_itemsC());
		MPI_Send(oDataGen->vecC.data(), oDataGen->size_itemsC(), MPI_BYTE, i, DATA, MPI_COMM_WORLD);
		//MDPPR1("(%d) to SLAVE (%d) send_const 2 end\n", ++cnt, i);
		
	}
	
	//MDPPR("(%d) send_const end\n", ++cnt);
	
	return 0;
	
}


int MpiMaster::send_idem() {
	//MDPPR("(%d) send_idem 0\n", ++cnt);
	
	for (int i = 1; i < proccnt; i++) {
		
		oItemIdem = (oDataGen->vecI)[i - 1];
		
		//MDPPR1("(%d)  to SLAVE (%d) send_idem : (GRID_SIZE item: %d, item id: %d, param: %f)\n",  ++cnt, i, oDataGen->size_itemI(), oItemIdem.id, oItemIdem.param);
		
		MPI_Send(& (oItemIdem), oDataGen->size_itemI(), MPI_BYTE, i, DATA, MPI_COMM_WORLD);
		
		sentcnt++;
	}
	
	//MDPPR("(%d) send_idem end\n", ++cnt);
	
	return 0;
	
}

int MpiMaster::isend_idem() {
	//MDPPR("(%d) isend_idem 0\n", ++cnt);
	
	for (int i = 1; i < proccnt; i++) {//Process from 1~~proccnt-1
		oItemIdem = oDataGen->vecI[sentcnt];
		
		//MDPPR1("(%d) to SLAVE (%d) isend_idem : (GRID_SIZE item: %d, item id: %d, param: %f)\n",  ++cnt, i, oDataGen->size_itemI(), oItemIdem.id, oItemIdem.param);
		
		MPI_Isend(& (oItemIdem), oDataGen->size_itemI(), MPI_BYTE, i, DATA, MPI_COMM_WORLD, & (requestSend[i - 1]));
		
		sentcnt++;
	}
	
	//MDPPR("(%d) isend_idem end\n", ++cnt);
	
	return 0;
	
}

#define MPI_IRECV_BYTES(ptr, size, id, request) MPI_Irecv(&(ptr), size, MPI_BYTE, id, RESULT, MPI_COMM_WORLD, &(request))

int MpiMaster::irecv_results() {
	//MDPPR("(%d) irecv_results 0\n", ++cnt);
	
	for (int i = 1; i < proccnt; i++) {//Process from 1~~proccnt-1
		
		//MDPPR1("(%d) from SLAVE (%d) irecv_results 1: (GRID_SIZE item: %d)\n", ++cnt, i, oResults->size_item());
		MPI_IRECV_BYTES(oResults->vecAns[i - 1], oResults->size_item(), i, requestReceive[i - 1]);
		//MDPPR1("(%d) from SLAVE (%d) irecv_results end: (GRID_SIZE item: %d, waits in buffer)\n",  ++cnt, i, oResults->size_item());
		
		
		
	}
	
	//MDPPR("(%d) irecv_results end\n", ++cnt);
	
	return 0;
	
}

int MpiMaster::run_while() {
	//MDPPR("(%d) run_while 0\n", ++cnt);
	
	do {
		
		//MDPPR3("(%d) run_while 1: vec idem (id: %d <= nIdem: %zu - 1) )\n",   ++cnt, oItemIdem.id, oDataGen->nIdemn - 1);
		
		run_while_waits();
		run_while_isend_check_anylast();
		run_while_icomm();
		
		//MDPPR3("(%d) run_while 2 \n", ++cnt);
		
	} while (oItemIdem.id != -1);
	
	check_results();
	
	//MDPPR("(%d) run_while end\n", ++cnt);
	
	return 0;
	
	
}

int MpiMaster::run_while_waits() {
	//MDPPR3("(%d) run_while_waits 0\n", ++cnt);
	
	
	MDPPR4("(%d) MPI_Waitany: (request receive from SLAVE: %d)\n", ++cnt, requestId + 1);
	
	MPI_Waitany(proccnt - 1, requestReceive, & requestId, MPI_STATUS_IGNORE);
	
	/* TODO ***********************************************************************************************************
	 * TODO sum up (integrate) vecAns over whole structure
	 * TODO ie. forces in time computed with cuda acting on element
	 * TODO oResults->integrate(oItemIdem,requestId);
	 * TODO **********************************************************************************************************/
	
	MDPPR4("(%d) MPI_Waitany end: (request receive from SLAVE: %d)\n", ++cnt, requestId+1);
	
	slavesCnt[requestId+1]+=1;
	
	
	//MDPPR4("(%d) MPI_Wait 1: (request send (%d) to SLAVE: %d )\n", ++cnt, sentcnt, requestId + 1);
	
	MPI_Wait(& requestSend[requestId], MPI_STATUS_IGNORE);
	
	
	//MDPPR4("(%d) MPI_Wait end: (request send (%d) to SLAVE: %d )\n", ++cnt, sentcnt, requestId + 1);
	
	
	oResults->add_temp(requestId);
	//result += resultTemp[requestId];
	
	
	//MDPPR3("(%d) run_while_waits end\n", ++cnt);
	
	return 0;
	
}

int MpiMaster::run_while_isend_check_anylast() {
	//MDPPR3("WHILE run_while_isend_check_anylast 0: sentcnt: %d, oItemIdem->nIdem: %zu\n", sentcnt, oDataGen->nIdemn);
	
	
	if (sentcnt == oDataGen->nIdemn) {
		//MDPPR4("WHILE run_while_isend_check_anylast 1: sentcnt: %d == oItemIdem->nIdem: %zu\n", sentcnt, oDataGen->nIdemn);
		
		oItemIdem.id = -1;
	} else {
		//MDPPR4("WHILE run_while_isend_check_anylast 1: sentcnt: %d != oItemIdem->nIdem: %zu\n", sentcnt, oDataGen->nIdemn);
		oItemIdem = oDataGen->vecI[sentcnt];
		sentcnt++;
		
	}
	
	
	return 0;
	
}

#define MPI_ISEND_BYTES(ptr, size, id, request)  MPI_Isend(&(ptr), size, MPI_BYTE, id, DATA, MPI_COMM_WORLD, &(request))


int MpiMaster::run_while_icomm() {
	//MDPPR3(" (%d) run_while_icomm 0\n", ++cnt);
	
	
	//MDPPR4(" (%d) MPI_Isend 1: (request send (%d), vec idem (id: %d, param: %f) )\n",   ++cnt, requestId + 1, oItemIdem.id, oItemIdem.param);
	
	MPI_ISEND_BYTES(oItemIdem, oDataGen->size_itemI(), requestId + 1, requestSend[requestId]);
	
	//MDPPR4(" (%d) MPI_Isend end...\n", ++cnt);
	
	//MDPPR4(" (%d) MPI_Irecv 1: (request send (%d), vec idem (id: %d, param: %f) )\n",  ++cnt, requestId + 1, oItemIdem.id, oItemIdem.param);
	
	int resultsIdx = proccnt - 1 + recvcnt;
	int resultsElemIdx = oResults->vecAns[resultsIdx].elemId;
	recvcnt++;
	
	//MDPPR4(" (%d) MPI_Irecv 1: (vecAns (id: %d), vec idem (id: %d) )\n", ++cnt, resultsIdx, resultsElemIdx, oResults->vecAns[resultsIdx].myrank);
	
	MPI_IRECV_BYTES(oResults->vecAns[resultsIdx], oResults->size_item(), requestId + 1, requestReceive[requestId]);
	
	//MDPPR4(" (%d) MPI_Irecv 2 FROM SLAVE (%d): (vecAns (id: %d), vec idem (id: %d) )\n", ++cnt, oResults->vecAns[resultsIdx].myrank, resultsIdx, resultsElemIdx );
	
	
	//MDPPR4(" (%d) MPI_Irecv end...\n", ++cnt);
	
	
	//MDPPR3(" (%d) run_while_icomm end\n", ++cnt);
	
	
	return 0;
	
}


int MpiMaster::after_while() {
	//MDPPR("(%d) after_while 0\n", ++cnt);
	
	MPI_Waitall(1 * (proccnt - 1), requestReceive, MPI_STATUS_IGNORE);
	
	
	for (int i = 0; i < proccnt - 1; i++) {
		recvcnt++;
		/* TODO sum up vecAns
		* TODO: result += resultTemp[i];
		*/
	}
	
	//MDPPR("(%d) after_while end\n", ++cnt);
	
	
	return 0;
	
}

int MpiMaster::kill_slaves() {
	//shut down the slave
	//Maybe I should send special number
	//MDPPR("(%d) kill_slaves 0\n", ++cnt);
	
	oItemIdem.id = -1;
	
	for (i = 1; i < proccnt; i++) {
		MPI_Send(& (oItemIdem), oDataGen->size_itemI(), MPI_BYTE, i, FINISH, MPI_COMM_WORLD);
		MPI_Send(NULL, 0, MPI_DOUBLE, i, FINISH, MPI_COMM_WORLD);
	}
	
	//MDPPR("(%d) kill_slaves end\n", ++cnt);
	
	
	return 0;
	
}

int MpiMaster::check_results() {
	
	//MDPPR("(%d) check_results 0 : cnt (recv: %d, sent: %d, sent-recv: %d)\n", ++cnt, recvcnt, sentcnt, sentcnt - recvcnt);
	
	for (int i = 0; i < sentcnt; i++) {
		
		int slaveId = oResults->vecAns[i].myrank;
		int elemId = oResults->vecAns[i].elemId;
		double ans = oResults->vecAns[i].ans;
		
		
		//MDPPR3("(%d) check_results 1 MPI_Irecv vecAns: (id: %d, slave: %d, itemId: %d, ans=%f)\n", ++cnt, i, slaveId, elemId, ans);
		
	}
	
	sleep(1);
	
	
	for (int i = 0; i < 16; i++) {
		MDPPR2("(%d) check_results 2 MPI_Irecv vecAns: (slave id: %d, results no: %d)\n", ++cnt, i, slavesCnt[i]);
	}
	
	
	//MDPPR("(%d) check_results end\n", ++cnt);
	
	return 0;
	
}

#define MPI_RECV_BYTES(ptr, size, status) MPI_Recv(&(ptr), size, MPI_BYTE, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &(status))

int MpiMaster::recv_orphans() {
	
	//MDPPR("(%d) recv_orphans 0\n", ++cnt);
	
	while (recvcnt < sentcnt) {
		MPI_RECV_BYTES(oResults->vecAns[recvcnt], oResults->size_item(), status);
		recvcnt++;
	}
	
	//MDPPR("(%d) recv_orphans end\n", ++cnt);
	
	return 0;
	
}

