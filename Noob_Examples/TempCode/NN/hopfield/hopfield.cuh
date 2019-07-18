#ifndef hopfiel_h
#define hopfiel_h


#include "../cuda_dummy.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>

#define sizeGrid 65535
#define sizeBlock 1024
#define sizeWarp 32


static int verbose_mode = 0;

__global__ void training(int dimP, int nP, int *ps, float *ws);
__global__ void hopActivation(int dimP, float *ws, int *pt, int *at);
float * lState (int nPatterns, int dimPattern, int *patterns);
int * actFunc(int dP, int *pattern, float *weight);


#endif