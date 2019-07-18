//
// Created by user on 7/23/2018.
//

#include "containers/thrust_vector_2d.cuh"
#include "AData.h"

#include "cuda_dummy.h"

int main(int argc, char ** argv){
	
	ADataConst_t *oDataConst;
	
	int gpuNum;
	int myrank = 1;
	
	int numElems = 100;
	int numWaves = 100;
	int numSteps = 100;
	int t0 = 100;
	int t1 = 100;
	
	int proccount;
	
	oDataConst = new ADataConst_t(myrank, numWaves);
	
	thrust::host_vector2d<ADataConstItem> vec;
	
	THRUST_HVEC2D(vec, ADataConstItem, 5, 3);
	
	
	
	
	vec.print();
	
	vec.ppr();
	
}