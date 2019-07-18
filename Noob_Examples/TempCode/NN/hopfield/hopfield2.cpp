/* Author: Alex Bod
 * Website: https://www.alexbod.com
 * Details: https://www.alexbod.com/hopfield-network/
 * License: The GNU General Public License, version 2
 * hopfield.c: Hopfield network in C
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


/* Convert points */
#define ZERO_OR_ONE(x) ((x)==-1 ? 0 : 1)
#define BINARY(x) ((x)== 0 ? -1 : 1)

#define NUMBER_OF_VECTORS 4
#define X 11
#define Y 10
#define AREA (X * Y)

/* Network struct */
typedef struct {
	int points;
	int *output;
	int *threshold;
	int **weight;
} net;

/* Input data for learning */
char x[NUMBER_OF_VECTORS][Y][X] = {
		{
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
char y[NUMBER_OF_VECTORS][Y][X] =
		{
				{
						"    OO    ",
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
						"OO      OO",
						"OO OOOO OO",
						"OO      OO",
						"OO      OO"
				}
		};

/* Input data for learning */
int input[NUMBER_OF_VECTORS][AREA];

/* Input data for recognition */
int notcorrect[NUMBER_OF_VECTORS][AREA];

/* Print the result */
void printNet(net *network) {
	int i, j;
	
	for (i = 0; i < Y; i++) {
		for (j = 0; j < X; j++) {
			printf("%c", ZERO_OR_ONE(network->output[i * X + j]) ? 'O' : ' ');
		}
		printf("\n");
	}
	printf("\n");
}

/* Create the net */
void createNet(net *network) {
	int i;
	
	network->points = AREA; /* Number of points = net area */
	network->output = (int *) calloc(AREA, sizeof(int));
	network->threshold = (int *) calloc(AREA, sizeof(int));
	network->weight = (int **) calloc(AREA, sizeof(int *));
	
	memset(network->threshold, 0, AREA * sizeof(int ));
	
	/* Fill thresholds with zeros and allocating memory for weight matrix */
	for (i = 0; i < AREA; i++) {
		network->weight[i] = (int *) calloc(AREA, sizeof(int));
	}
}

/* Convert points of 'O' to the binary -1 or +1 */
void pointstoBinary(net *network) {
	int n, i, j;
	
	for (n = 0; n < NUMBER_OF_VECTORS; n++) {
		for (i = 0; i < Y; i++) {
			for (j = 0; j < X; j++) {
				
				/* Make points binary and convert 3d matrix to 2d */
				input[n][i * X + j] = BINARY(x[n][i][j] == 'O');
				notcorrect[n][i * X + j] = BINARY(y[n][i][j] == 'O');
			}
		}
	}
}

/* Calculate the weight matrix = learning */
void calculateWeights(net *network) {
	int i, j, n;
	int weight;
	
	for (i = 0; i < network->points; i++) {
		for (j = 0; j < network->points; j++) {
			weight = 0;
			for (n = 0; n < NUMBER_OF_VECTORS; n++) {
				
				/* Main formula for calculating weight matrix */
				weight += input[n][i] * input[n][j];
			}
			
			/* Fill the weight matrix */
			network->weight[i][j] = weight;
		}
	}
	
	for (i = 0; i < network->points; i++) {
		network->weight[i][i] = 0;
	}
	
}

/* Set the input vector to the Net->output */
void setInput(net *network, int *input) {
	int i;
	
	for (i = 0; i < network->points; i++) {
		network->output[i] = input[i];
	}
	printNet(network);
}

/* Set the Net->output to the output vector */
void getOutput(net *network, int *output) {
	int i;
	
	for (i = 0; i < network->points; i++) {
		output[i] = network->output[i];
	}
	printNet(network);
	printf("----------\n\na");
}

/* Next iteration to find the local minimum = recognized pattern */
int nextIteration(net *network, int i) {
	int j;
	int sum, out = 0;
	int changed;
	
	changed = 0;
	sum = 0;
	//output
	for (j = 0; j < network->points; j++) {
		sum += network->weight[i][j] * network->output[j];
	}
	
	//activation
	if (sum != network->threshold[i]) {
		if (sum > network->threshold[i]) out = 1;
		else out = -1;
		if (out != network->output[i]) {
			network->output[i] = out;
			changed = 1;
		}
	}
	return changed;
}

/* Asynchronous correction */
void asynCor(net *network) {
	int i = 0;
	int updLast =0;
	bool doWhile;
	int isUpd = false;
	int neuronId;
	int updDelay;
	
	do {
		i++;
		/* Every time take random element for the correction */
		neuronId = rand() % (network->points);
		isUpd = nextIteration(network, neuronId);
		if (isUpd) updLast = i;
		
		updDelay = i - updLast ;
		doWhile = updDelay < network->points*10;
		
	} while (doWhile);
}

/* Find the local minimum = recognizing the pattern */
void findLocalMinimum(net *network, int *input) {
	int output[AREA];
	
	/* Print not correct input for recognizing */
	setInput(network, input);
	
	/* Asynchronous correction */
	asynCor(network);
	
	/* Print recognized output */
	getOutput(network, output);
}

int main() {
	net network;
	int n;
	
	/* Allocate memory and create the net */
	createNet(& network);
	
	/* Make the points matrix binary */
	pointstoBinary(& network);
	
	/* Calculate the weight matrix */
	calculateWeights(& network);
	
	/* Find the local minimum = recognizing the pattern */
	for (n = 0; n < NUMBER_OF_VECTORS; n++) {
		findLocalMinimum(& network, notcorrect[n]);
	}
	
	return 0;
}