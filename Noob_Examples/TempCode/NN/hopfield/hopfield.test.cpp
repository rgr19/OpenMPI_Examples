#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <regex.h>
#include <iostream>

#include "hopfield.cuh"

#define NPATTERNS 4
#define DIM_Y 11
#define DIM_X 11

char patternsChar[NPATTERNS][DIM_Y][DIM_X] = {
		{
				"          ",
				"    OO    ",
				"    OO    ",
				"   OOOO   ",
				"   O  O   ",
				"  OO  OO  ",
				"  O    O  ",
				" OOOOOOOO ",
				" OOOOOOOO ",
				"OO      OO",
				"OO      OO"
		},
		{
				"          ",
				"OOOOOO    ",
				"OOOOOOO   ",
				"OO   OO   ",
				"OOOOOOO   ",
				"OOOOOOO   ",
				"OO   OOO  ",
				"OO    OO  ",
				"OO   OOO  ",
				"OOOOOOO   ",
				"OOOOOO    "
		},
		{
				"          ",
				"OOOOOOOOOO",
				"OOOOOOOOOO",
				"OO      OO",
				"OO        ",
				"OO        ",
				"OO        ",
				"OO        ",
				"OO      OO",
				"OOOOOOOOOO",
				"OOOOOOOOOO"
		},
		{
				"          ",
				"OO      OO",
				"OO      OO",
				"OO      OO",
				"OO      OO",
				"OOOOOOOOOO",
				"OOOOOOOOOO",
				"OO      OO",
				"OO      OO",
				"OO      OO",
				"OO      OO"
		}
};

/* Input data for recognition */
char inputChar[NPATTERNS][DIM_Y][DIM_X] =
		{
				{
						"    OO    ",
						"  O OO  O ",
						"  O OO  O ",
						" O O OO O ",
						"   O  OO  ",
						"  OO      ",
						"  O    OO ",
						" OOOO  OO ",
						" OOO OOOO ",
						"OO   O  OO",
						"OO  O   OO"
				},
				
				{
						"OO OOOO   ",
						"OOOOOO  O ",
						"OO   OOOO ",
						"OO   OOOO ",
						"OOO OO OO ",
						"OOO OOO O ",
						"OO   OOO  ",
						"OO O  OO  ",
						"OO O OOO  ",
						"OO  OOO  O",
						"O OOOO  O "
				},
				
				{
						"OOOOOOOOOO",
						"OOOOOOOOOO",
						"OO      OO",
						"OO      OO",
						"OO        ",
						"OOOOOO    ",
						"OO    OOO ",
						"OO        ",
						"OO      OO",
						"OOOOOOOOOO",
						"OOOOOOOOOO"
				},
				
				{
						"OO      OO",
						"OO      OO",
						"OO OOOO OO",
						"OO      OO",
						"OOOOOOOOOO",
						"OOOOOOOOOO",
						"OOOOOOOOOO",
						"OO      OO",
						"OO OOOO OO",
						"OO      OO",
						"OO      OO"
				}
		};


#define CICLI 32

#define LOG_BUILD {                                                                                           \
    const char *buildString = "This build " __FILE__ " was compiled at " __DATE__ ", " __TIME__ ".\n";          \
    printf("#### BUILD INFO: %s", buildString);                                                                 \
}

/* Convert points */
#define ZERO_OR_ONE(x) ((x)==-1 ? 0 : 1)
#define BINARY(x) ((x)== 0 ? -1 : 1)


bool verbose = false;


void print_weights(float *weights, int dimP) {
	int i, j;
	printf("Weights:\n");
	for (i = 0; i < dimP; i++) {
		printf("[ ");
		for (j = 0; j < dimP; j++) {
			printf("%.3f ", weights[i * dimP + j]);
		}
		printf("]\n");
	}
}


int checkVal(float *weights, int *epat, int dimPattern, int nPatterns) {
	int i = 0;
	for (i = 0; i < CICLI; i++)
		epat = actFunc(dimPattern, epat, weights);
	if (epat == NULL) {
		printf("Error on Activarion\n");
		return 1;
	}
	
	printf("Activation:\t[ ");
	for (i = 0; i < dimPattern; i++)
		printf("%i ", epat[i]);
	printf("]\n");
	
	return 0;
}


/*
 * void arange( ) {
			for ( size_t i = 0; i < N; i++ ) {
				for ( size_t j = 0; j < M; j++ ) {
					for ( size_t k = 0; k < NPATTERNS; k++ ) {
						vec[k + NPATTERNS * j + NPATTERNS * M * i] = k + NPATTERNS * j + NPATTERNS * M * i;
					}
				}
			}
		}
 */

/* Convert points of 'O' to the binary -1 or +1 */
void pointstoBinary(int *patters, int *input) {
	int k, i, j, idx;
	
	for (i = 0; i < DIM_X; i++) {
		for (j = 0; j < DIM_Y; j++) {
			for (k = 0; k < NPATTERNS; k++) {
				
				/* Make points binary and convert 3d matrix to 2d */
				//k + NPATTERNS * j + NPATTERNS * DIM_Y * i
				idx = k + NPATTERNS * j + NPATTERNS * DIM_Y * i;
				patters[idx] = BINARY(patternsChar[k][i][j] == 'O');
				input[idx] = BINARY(inputChar[k][i][j] == 'O');
			}
		}
	}
}

float *patternsFromExample(size_t nPatterns, size_t dimPattern, int *input) {
	int *patterns;
	int i, j;
	time_t t;
	size_t size = nPatterns * dimPattern * dimPattern;
	
	
	float *weights = lState(nPatterns, dimPattern, patterns);
	if (weights == NULL) {
		printf("Error on Learning\n");
		return NULL;
	}
	
	if (verbose_mode) {
		print_weights(weights, dimPattern);
	}
	free(patterns);
	return weights;
}

float *randomValue(size_t nPatterns, size_t dimPattern) {
	int *patterns;
	int i, j;
	time_t t;
	size_t size = nPatterns * dimPattern * dimPattern;
	
	
	printf("generating patternsChar: size (NM: %zu, N: %d, M:%d)\n", size, nPatterns, dimPattern);
	
	patterns = (int *) malloc(size * sizeof(int));
	if (patterns == nullptr) {
		std::cerr << "PATTERNS NOT GENERATED! ERROR..." << std::endl;
		exit(-10);
	}
	
	srand((unsigned) time(& t));
	for (i = 0; i < size; i++) {
		int p = rand() % 2;
		
		patterns[i] = p;
		
	}
	
	
	printf("Patterns Generated: \n");
	if (verbose) {
		for (j = 0; j < nPatterns; j++) {
			printf("\t\t[ ");
			for (i = 0; i < dimPattern; i++) {
				printf("%d ", patterns[j * dimPattern + i]);
			}
			printf("]\n");
		}
	}
	
	float *weights = lState(nPatterns, dimPattern, patterns);
	if (weights == NULL) {
		printf("Error on Learning\n");
		return NULL;
	}
	
	if (verbose_mode) {
		print_weights(weights, dimPattern);
	}
	free(patterns);
	return weights;
}


int parserFile(char *path, int *pts) {
	FILE *fp;
	char buffer[255];
	const char *p;
	regex_t re;
	regmatch_t match;
	
	fp = fopen(path, "r");
	if (fp == NULL) return 1;
	if (regcomp(& re, "[0-1]", REG_EXTENDED) != 0) return 1;
	
	int i = 0;
	while (fgets(buffer, 255, (FILE *) fp)) {
		p = buffer;
		while (regexec(& re, p, 1, & match, 0) == 0) {
			pts[i] = p[match.rm_so] - '0';
			p += match.rm_eo;
			i++;
		}
	}
	regfree(& re);
	return 0;
}


int main(int argc, char *argv[]) {
	
	LOG_BUILD
	
	
	int type = 0;
	int *recognize;
	char *pathPf = NULL, *pathRf = NULL;
	
	size_t dimxy = DIM_X * DIM_Y;
	
	int *input = (int *) malloc(dimxy * sizeof(int));
	
	printf("generating input: size (dimxy : %zu)\n", dimxy);
	
	if (input == nullptr) {
		std::cerr << "malloc for input ERROR..." << std::endl;
		exit(-10);
	}
	
	int *patterns = (int *) malloc(dimxy * sizeof(int));
	
	printf("generating patterns: size (dimxy : %zu)\n", dimxy);
	
	if (patterns == nullptr) {
		std::cerr << "malloc for patterns ERROR..." << std::endl;
		exit(-10);
	}
	
	pointstoBinary(patterns, input);
	
	
	printf("Patterns Generated: \n");
	if (verbose) {
		for (int j = 0; j < NPATTERNS; j++) {
			printf("\t\t[ ");
			for (i = 0; i < dimPattern; i++) {
				printf("%d ", patterns[j * dimPattern + i]);
			}
			printf("]\n");
		}
	}
	
	float *weights = patternsFromExample(NPATTERNS, DIM_X * DIM_Y, input);
	
	if (weights == NULL) {
		printf("Error on Learning");
		exit(1);
	}
	
	printf(" ]\n");
	checkVal(weights, input, dimP, nP);
	free(weights);
	free(recognize);
	
	return 0;
}