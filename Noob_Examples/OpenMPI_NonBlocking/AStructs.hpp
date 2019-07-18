//
// Created by a on 3/19/19.
//

#ifndef OPENMPI_NONBLOCKING_ASTRUCTS_HPP
#define OPENMPI_NONBLOCKING_ASTRUCTS_HPP

#include<stdio.h>

#include <iostream>
#include <random>

#define size_t size_t

#define LOG_BUILD printf("LOG BUILD: date[%s], time[%s], file[%s]\n",__DATE__, __TIME__, __FILE__);
#define PPR(fmt, ...) printf(fmt, ##__VA_ARGS__);

#ifdef DEBUG
    #define DPPR(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__);
    #define MDPPR(fmt, ...) fprintf(stderr, "..."); fprintf(stderr, fmt, ##__VA_ARGS__);
    #define MDPPR1(fmt, ...) fprintf(stderr, "......"); fprintf(stderr, fmt, ##__VA_ARGS__);
    #define MDPPR2(fmt, ...) fprintf(stderr, "........."); fprintf(stderr, fmt, ##__VA_ARGS__);
    #define MDPPR3(fmt, ...) fprintf(stderr, "............"); fprintf(stderr, fmt, ##__VA_ARGS__);
    #define MDPPR4(fmt, ...) fprintf(stderr, "..............."); fprintf(stderr, fmt, ##__VA_ARGS__);
#else
    #define DPPR(fmt, ...)  ((void)0)
    #define MDPPR(fmt,...)  ((void)0)
    #define MDPPR1(fmt,...) ((void)0)
    #define MDPPR2(fmt,...) ((void)0)
    #define MDPPR3(fmt,...) ((void)0)
    #define MDPPR4(fmt,...) ((void)0)
#endif


#include <vector>
#include <unistd.h>


typedef enum MPI_FLAGS_t {
    DATA,
    RESULT,
    FINISH,
} MPI_FLAGS;


typedef struct Ans_t {
    int myrank;
    int elemId;
    double ans;
} Ans;

typedef struct ADataIdemnItem_t {

    int id;


} ADataIdemnItem;


typedef struct AData_t {
    int myrank;
    size_t nIdemn;

    std::vector<Ans> vecAns;
    std::vector<float> vecC;
    std::vector<ADataIdemnItem> vecI;

    AData_t(int myrank, size_t nIdemn, size_t numWaves) : myrank(myrank), nIdemn(nIdemn) {

    }

    int size_baseCI(){
        return sizeof(myrank) + sizeof(nIdemn);
    }

    int size_itemsC(){
        return 1;
    }

    int size_itemI(){
        return 1;
    }

    int genC(){

    }

    int genI(){

    }

} AData;



typedef struct ACalc_t{

    std::vector<Ans_t> vecAns;


    ACalc_t(int myrank, size_t numElems, size_t numSteps, size_t numWaves, int t0, int t1){

    }

    int empty(){
        return vecAns.empty();
    }

    int size_item(){

    }

    int add_temp(int requestId){

    }

    int compute(AData_t * oDataGen, ADataIdemnItem * oItemIdem, int myrank){
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<> distr(1, 10); // define the range

        sleep(static_cast<unsigned int>(distr(eng)));
        return 0;

    }


} ACalc;







#endif //OPENMPI_NONBLOCKING_ASTRUCTS_HPP
