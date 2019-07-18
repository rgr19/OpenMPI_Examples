//
// Created by user on 7/22/2018.
//

#include "AData.h"
#include "ACalc.h"
#include <iostream>
#include <chrono>


#define TIMEIT(kernel, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    kernel(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout << duration << std::endl;                                                                 \
}

int mat_mul_std(size_t N, size_t P, size_t M) {
	// mat A(N rows, NPATTERNS columns) x
	// mat B(NPATTERNS rows, M columns) =
	// mac C(N rows, M columns)
	std::vector2d<ADataIdemnItem_t> a(N, P);
	std::vector2d<ADataIdemnItem_t> b(P, M);
	std::vector2d<ACalcStep_t> c(N, M);
	
	memset(a.data(),1,sizeof(ADataIdemnItem_t)*N*P);
	memset(b.data(),1,sizeof(ADataIdemnItem_t)*P*M);
	
	for (int _ = 0; _ < 100; _++) {
		
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
				c[c.idx(i,j)].ans = 0;
				for (int k = 0; k < P; k++) {
					c[c.idx(i,j)].ans += a.get(i, k).param * b.get(k, j).param;
				}
			}
		}
	}
	
	
	printf("a[10][10]=%f\n", a[a.idx(10,10)]);
	printf("b[10][10]=%f\n", b[b.idx(10,10)]);
	printf("c[10][10]=%f\n", c[c.idx(10,10)]);
	
}

int mat_mul_c(size_t N, size_t P, size_t M) {
	// mat A(N rows, NPATTERNS columns) x
	// mat B(NPATTERNS rows, M columns) =
	// mac C(N rows, M columns)
	double a[N][P];
	double b[P][M];
	double c[N][M];
	
	printf("sizeof (a: %zu, b: %zu, c: %zu)\n", sizeof(a), sizeof(b), sizeof(c));
	
	
	for (int j = 0; j < P; j++) {
		for (int i = 0; i < N; i++) {
			a[i][j] = 1;
		}
		for (int i = 0; i < M; i++) {
			b[j][i] = 1;
		}
	}
	
	for (int _ = 0; _ < 100; _++) {
		

		
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
				c[i][j] = 0;
				for (int k = 0; k < P; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		
	}
	
	printf("a[10][10]=%f\n", a[10][10]);
	printf("b[10][10]=%f\n", b[10][10]);
	printf("c[10][10]=%f\n", c[10][10]);
	
}

int main(int argc, char **argv) {
	
	AData_t *oDataGen;
	ACalc_t *oResults;
	
	ADataIdemnItem_t oItemIdem[2];
	
	int gpuNum;
	int myrank = 0;
	
	size_t numElems = 200;
	size_t numWaves = 1000;
	size_t numSteps = 200;
	int t0 = 100;
	int t1 = 100;
	
	TIMEIT(mat_mul_std, numElems, numSteps, numWaves);
	TIMEIT(mat_mul_c, numElems, numSteps, numWaves);
	
	return 0;
	oDataGen = new AData_t(myrank, numElems, numWaves);
	oResults = new ACalc_t(myrank, numElems, numSteps, numWaves, t0, t1);
	
	oItemIdem[0] = (oDataGen->vecI)[0];
	
	oDataGen->genI();
	oDataGen->genC();
	
	oDataGen->saveI();
	oDataGen->saveC();
	
	oResults->compute(oDataGen, oItemIdem, 1);
	
}