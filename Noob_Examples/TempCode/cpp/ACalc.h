//
// Created by user on 7/25/2018.
//

#ifndef ACALC_H
#define ACALC_H


#include <cstring>

#include "AData.h"

#include "CppUtils/containers/std_vector_2d.h"


typedef struct ACalcStep_t {
	
	int myrank = -1;
	int elemId = -1;
	double ans = 0;
	
	inline ACalcStep_t & operator=(const int x) {
		ans = x;
		return * this;
	}
	
	inline ACalcStep_t & operator=(const float x) {
		ans = x;
		return * this;
	}
	
	inline ACalcStep_t & operator=(const double x) {
		ans = x;
		return * this;
	}
	
	inline ACalcStep_t & operator=(const ACalcStep_t & a) {
		myrank = a.myrank;
		elemId = a.elemId;
		ans = a.ans;
		return * this;
	}
	
	
	inline ACalcStep_t & operator+=(const ACalcStep_t & a) {
		ans += a.ans;
		return * this;
	}
	
	inline ACalcStep_t & operator+=(const ACalcStep_t *a) {
		ans += a->ans;
		return * this;
	}
	
	inline ACalcStep_t & operator+=(const ADataIdemnItem_t & a) {
		ans += a.param;
		return * this;
	}
	
	inline ACalcStep_t & operator+=(const ADataIdemnItem_t *a) {
		ans += a->param;
		return * this;
	}
	
	inline ACalcStep_t operator+(const ACalcStep_t & a) const { return {myrank, elemId, ans + a.ans}; }
	
	inline ACalcStep_t operator*(const ACalcStep_t *a) { return {myrank, elemId, ans + a->ans}; }
	
	inline ACalcStep_t operator*(const ACalcStep_t & a) { return {myrank, elemId, ans * a.ans}; }
	
	
	void calc(int i,
	          const ACalcStep_t *,
	          const ADataConstItem *,
	          const ADataIdemnItem *,
	          const AData *
	         );
	
} ACalcStep;


std::ostream & operator<<(std::ostream & os, const ACalcStep_t & o);


typedef struct ACalc_t {
	int myrank;
	const size_t sizeself = sizeof(ACalc_t);
	const size_t sizeitem = sizeof(ACalcStep_t);
	const size_t sizebase = sizeself
	                        - 3 * sizeof(vecSum)
	                        - sizeof(vecAns);
	
	bool empty = true;
	
	size_t nIdem;
	size_t nConst;
	size_t nSteps;
	size_t size2d; //nConst * nSteps
	
	double t0, t1, dt;
	
	std::vector2d<ACalcStep> vecAns;
	std::vector<ACalcStep> vecSum;
	std::vector<ACalcStep> vecAnsInit;
	
	int add_temp(int id);
	
	inline int size_self() const { return (int) sizeself; };
	
	inline int size_item() const { return (int) sizeitem; };
	
	inline int size_base() const { return (int) sizebase; };
	
	inline int size_items() const { return static_cast<int>(sizeitem * nConst); };
	
	inline int size_items2d() const { return static_cast<int>(sizeitem * nConst * nSteps); };
	
	~ACalc_t();
	
	ACalc_t(int myrank, size_t nConst, size_t nSteps, size_t nIdem, double t0, double t1);
	
	
	void compute(const AData_t *, const ADataIdemnItem_t *, int myrank);
	
	
} AResultSystem;


void printElemDevice(int id, ACalcStep_t & res);

void printElemHost(ACalcStep_t & res);

void kernelPerstep(int,
                   const ADataConstItem *,
                   const ADataIdemnItem *,
                   const AData_t *,
                   ACalcStep_t *);


#define STRUCTS_ARGS                                    \
    AData_t *oDataGen,                             \
    ACalc_t *oResults

#define STRUCTS_INIT                                    \
    oDataGen(oDataGen),                                   \
    oResults(oResults)


#endif //MYMPICUDATEST_ACALC_H
