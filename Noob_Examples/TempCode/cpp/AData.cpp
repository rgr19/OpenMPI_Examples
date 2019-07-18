//
// Created by robgrzel on 04.06.17.
//
#include <vector>
#include "AData.h"


std::ostream & operator<<(std::ostream & os, const ADataConstItem_t & o) {
	os << o.id << ", " << o.param;
	return os;
	
}


std::ostream & operator<<(std::ostream & os, const ADataIdemnItem_t & o) {
	os << o.id << ", " << o.param;
	return os;
	
}

std::ostream & operator<<(std::ostream & os, const std::vector<ADataConstItem_t> & vec) {
	
	for (auto & i: vec) os << i << '\n';
	return os;
	
}

std::ostream & operator<<(std::ostream & os, const std::vector<ADataIdemnItem_t> & vec) {
	
	for (auto & i: vec) os << i << '\n';
	return os;
	
}

std::ostream & operator<<(std::ostream & os, const AData_t data) {
	os << "DATA GEN: myrank " << data.myrank
	   << ", n(C: " << data.nConst
	   << ", I: " << data.nIdemn
	   << "), GRID_SIZE(self: " << data.sizeSelf
	   << ", base_C: " << data.sizeBase_C
	   << ", base_CI: " << data.sizeBase_CI
	   << ", itemI: " << data.sizeItemI
	   << ", itemC: " << data.sizeItemC
	   << "), dt: " << data.dt
	   << '\n' << '\n';
	
	os << "VEC_CONST: [" << '\n';
	for (auto & i: data.vecC) os << i << '\n';
	os << " ] " << '\n' << '\n';
	os << "VEC_IDEMN: [" << '\n';
	for (auto & i: data.vecI) os << i << '\n';
	os << " ] " << '\n';
	
	return os;
};

std::ostream & operator<<(std::ostream & os, const AData_t *data) {
	os << "DATA GEN: myrank " << data->myrank
	   << ", n(C: " << data->nConst
	   << ", I: " << data->nIdemn
	   << "), GRID_SIZE(self: " << data->sizeSelf
	   << ", base_C: " << data->sizeBase_C
	   << ", base_CI: " << data->sizeBase_CI
	   << ", itemI: " << data->sizeItemI
	   << ", itemC: " << data->sizeItemC
	   << "), dt: " << data->dt
	   << '\n' << '\n';
	
	os << "VEC_CONST: [" << '\n';
	for (auto & i: data->vecC) os << i << '\n';
	os << " ] " << '\n' << '\n';
	os << "VEC_IDEMN: [" << '\n';
	for (auto & i: data->vecI) os << i << '\n';
	os << " ] " << '\n';
	
	return os;
};


void AData_t::genC() {
	PPR("AData_t.genC : nConst=%zu\n", nConst);
	
	
	int j = 0;
	for (auto & i: vecC) {
		i.param = j * 0.1;
		i.id = j;
		j++;
		if (j < 10) std::cout << i << std::endl;
	}
	
	emptyC = false;
	sizeItemsC = sizeItemC * nConst;
	
	PPR("AData_t.genC : sizeItemsC=%zu\n", sizeItemsC);
	
	
}

void AData_t::genI() {
	PPR("AData_t.genI : nIdem=%zu\n", nIdemn);
	
	
	int j = 0;
	for (auto & i: vecI) {
		i.param = j * 0.1;
		i.id = j;
		j++;
		if (j < 10) std::cout << i << std::endl;
	}
	
	emptyI = false;
	sizeItemsI = sizeItemI * nIdemn;
	
	PPR("AData_t.genI : sizeItemsI=%zu\n", sizeItemsI);
	
	
}


void AData_t::saveI() {
	PPR("AData_t saveI >> myrank: %d, nIdemn: %zu, nConst: %zu \n",
	    myrank, nIdemn, nConst);
	ostream2file("dataidem.txt", vecC);
};

void AData_t::saveC() {
	
	PPR("AData_t saveC >> myrank: %d, nIdemn: %zu, nConst: %zu \n",
	    myrank, nIdemn, nConst);
	ostream2file("dataconst.txt", vecI);
	
};

AData_t::AData_t(int myrank, size_t nIdemn, size_t nConst) :
		myrank(myrank), nIdemn(nIdemn), nConst(nConst), dt(0.1),
		emptyI(true), emptyC(true), sizeItemsC(0), sizeItemsI(0) {
	
	PPR("AData_t >> myrank: %d, nIdemn: %zu, nConst: %zu \n",
	    myrank, nIdemn, nConst);
	
	LOG_BUILD
	
	vecI.resize(nIdemn);
	vecC.resize(nConst);
	
}

AData_t::AData_t(int myrank, size_t nIdemn, size_t nConst, double dt) :
		myrank(myrank), nIdemn(nIdemn), nConst(nConst), dt(dt),
		emptyI(true), emptyC(true), sizeItemsC(0), sizeItemsI(0) {
	PPR("AData_t >> myrank: %d, nIdemn: %zu, nConst: %zu, dt: %f \n",
	    myrank, nIdemn, nConst, dt);
	
	LOG_BUILD
	vecI.resize(nIdemn);
	vecC.resize(nConst);
	
	emptyC = true;
	emptyI = true;
	sizeItemsC = sizeItemC * nConst;
	sizeItemsI = sizeItemsI * nIdemn;
	
}


AData_t::~AData_t() {
	emptyI = true;
	emptyC = true;
	sizeItemsI = 0;
	sizeItemsC = 0;
	
	std::vector<ADataConstItem_t>().swap(vecC);
	std::vector<ADataIdemnItem_t>().swap(vecI);
	
}

