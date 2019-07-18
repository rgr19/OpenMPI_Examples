//
// Created by User on 2/23/2019.
//

#include <mpi.h>
//#include "main.mpi.h"

int main(int argc, char **argv){

    int myrank;
    int proccount;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

}