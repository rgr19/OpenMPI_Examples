#pragma once

#ifndef UTILS_STRUCTS_H
#define UTILS_STRUCTS_H

#define idemn


#include "CppUtils/template_utils.h"
#include "CppUtils/debug_printing.h"


typedef struct ADataConstItem_t {
	
	int id = -1;
	double param = 0;
	
} ADataConstItem;


typedef struct ADataIdemnItem_t {
	
	int id = -1; //id of elem
	double param = 0; //position init\

	
	inline ADataIdemnItem_t operator*(const ADataIdemnItem_t * a) {
		return {id, param  * a->param};
	}
	
	inline const ADataIdemnItem_t  operator*(const ADataIdemnItem_t & a) {
		return {id, param  * a.param};
	}
	
	friend inline const ADataIdemnItem_t  operator*(const ADataIdemnItem_t & L, const ADataIdemnItem_t & R) {
		return {L.id, L.param  * R.param};
	}
	
} ADataIdemnItem;




typedef struct AData_t {
	const int myrank;
	
	const size_t sizeSelf = sizeof(AData_t);
	const size_t sizeItemC = sizeof(ADataConstItem_t);
	const size_t sizeItemI = sizeof(ADataIdemnItem_t);
	const size_t sizeBase_C = sizeSelf - sizeof(vecC);
	const size_t sizeBase_CI = sizeSelf - sizeof(vecC) - sizeof(vecI);
	
	idemn size_t sizeItemsI;
	idemn size_t sizeItemsC;
	
	idemn bool emptyI, emptyC;
	
	//shared between elements
	const size_t nIdemn;
	const size_t nConst;
	
	double dt = 0.1;
	
	std::vector<ADataConstItem> vecC;
	
	std::vector<ADataIdemnItem> vecI;
	
	
	void genI();
	
	void genC();
	
	void saveI();
	
	void saveC();
	
	inline int size_self() const {return (int) sizeSelf; };
	inline int size_itemC() const {return (int) sizeItemC; };
	inline int size_itemI() const {return (int) sizeItemI; };
	inline int size_baseC() const {return (int) sizeBase_C; };
	inline int size_baseCI() const {return (int) sizeBase_CI; };
	
	inline int size_itemsC() const {return static_cast<int>(sizeItemsC); };
	inline int size_itemsI() const {return static_cast<int>(sizeItemsI); };
	
	inline auto beginC() const { return vecC.begin(); };
	inline auto endC() const { return vecC.end(); };
	inline auto dataC() const { return vecC.data(); };
	
	inline auto beginI() const { return vecI.begin(); };
	inline auto endI() const { return vecI.end(); };
	inline auto dataI() const { return vecI.data(); };
	
	
	AData_t(int myrank, size_t nIdemn, size_t nConst);
	
	AData_t(int myrank, size_t nIdemn, size_t nConst, double dt);
	
	~AData_t();
	
	
} AData;

std::ostream& operator << (std::ostream& os, const ADataConstItem_t & o);


std::ostream& operator << (std::ostream& os, const ADataIdemnItem_t & o) ;

std::ostream& operator << (std::ostream& os, const std::vector<ADataConstItem_t> & vec);

std::ostream& operator << (std::ostream& os, const std::vector<ADataIdemnItem_t> & vec);

std::ostream& operator << (std::ostream& os, const AData_t * data);

std::ostream& operator << (std::ostream& os, AData_t data);



#endif //UTILS_STRUCTS_H
