#pragma once

#ifndef STD_UTILS_H
#define STD_UTILS_H


#include <iostream>
#include <chrono>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>

#ifndef idemn
#define idemn
#endif

#ifndef elif
#define elif else if
#endif


#include <iostream>
#include <random>

/*
int array[width * height];

int SetElement(int row, int col, int value)
{
	array[width * row + col] = value;
}
*/

#define IDX2(W, r, c) W*r+c


#define IDXCOL idx + (e + BLOCK_SIZE * m) * nRows


#define fori(s, n) for(int i=s; i<n; ++i)
#define forj(s, n) for(int j=s; j<n; ++j)
#define fork(s, n) for(int k=s; k<n; ++k)
#define forl(s, n) for(int l=s; l<n; ++l)
#define forh(s, n) for(int h=s; h<n; ++h)
#define forauto(vec) for(auto & i : vec)

#define LOG_BUILD {                                                                                           \
    const char *buildString = "This build " __FILE__ " was compiled at " __DATE__ ", " __TIME__ ".\n";          \
    printf("#### BUILD INFO: %s", buildString);                                                                 \
}

#define randint(min, max) (min + (rand() % static_cast<int>(max - min + 1)))

typedef const unsigned int cuint;
typedef unsigned int uint;

#define cuint cuint
#define uint uint


#define DATA 0
#define RESULT 1
#define FINISH 2

#define ID -1

#define BTOMB(x) (x)/1024/1024
#define MBTOB(x) (x)*1024*1024

template<typename T>
T nextpow2(T x)
{
	T n = x;
	--n;
	
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	
	return n + 1;
}


#define TIMEIT(function, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    function(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout<< "TIMEIT : " #function << "[microseconds]" << duration << std::endl;                                    \
}


#define TIMEIT_10(function, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    fori(0,10) function(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout<< "TIMEIT_10 : " #function << "[microseconds]" << duration << std::endl;                                    \
}


#define TIMEIT_100(function, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    fori(0,100) function(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout<< "TIMEIT_1000 : " #function << "[microseconds]" << duration << std::endl;                                    \
}


#define TIMEIT_1000(function, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    fori(0,1000) function(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout<< "TIMEIT_1000 : " #function << "[microseconds]" << duration << std::endl;                                    \
}

typedef struct random_choice_t {
	std::mt19937 eng; // seed the generator
	std::uniform_int_distribution<> distr; // define the range
	
	random_choice_t(cuint seed, cuint a, cuint b) : eng(seed), distr(a, b) {
	
	}
	
	inline void set_range(cuint a, cuint b) {
		distr = std::uniform_int_distribution<>(a, b);
	}
	
	inline int gen() { return distr(eng); };
	
} rand_choice;


#define RANDOM_CHOICE(distr, eng) distr(eng)

int random_choice(int a, int b, int seed = 0) {
	std::mt19937 eng(seed); // seed the generator
	std::uniform_int_distribution<> distr(a, b); // define the range
	return distr(eng);
}

double wallclock(void) {
	struct timeval tv;
	struct timezone tz;
	double t;
	
	gettimeofday(&tv, &tz);
	
	t = (double) tv.tv_sec * 1000;
	t += ((double) tv.tv_usec) / 1000.0;
	
	return t;
}// millisecond

inline bool is_power_of_2(unsigned int x) {
	return x > 0 && !(x & (x - 1));
}

extern "C"
bool isPow2(unsigned int x) {
	return ((x & (x - 1)) == 0);
}

unsigned int nextPow2(unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

#endif //MYMPICUDATEST_STD_UTILS_H
