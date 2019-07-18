//
// Created by user on 7/26/2018.
//

#include <cstdlib>

typedef struct {
// Number of neurons
	int size;
// State of neurons
	int *S;
// Matrix of the weights of the connections ;
	int **W;
// Table of the external inputs of the neurons;
	int *X;
} NNetwork;

int threshold(int x){
	return x/(1+abs(x));
}

void Update(NNetwork *pN) {
	int activation = 0;
// may introduce parallelism
	int idx = rand() % (pN->size);
	
	activation = pN->X[idx];
	for (int j = 0; j < pN->size; j++) {
		int Wi = pN->W[idx][j];
		activation += Wi * pN->S[j];
	}
	
	pN->S[idx] = threshold(activation);
// end
}