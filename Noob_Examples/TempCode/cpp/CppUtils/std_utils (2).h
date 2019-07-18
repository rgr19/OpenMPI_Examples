//
// Created by user on 7/27/2018.
//

#ifndef STD_UTILS_H
#define STD_UTILS_H


#include <iostream>
#include <chrono>
#include <unistd.h>

#ifndef idemn
#define idemn
#endif

#ifndef elif
#define elif else if
#endif

#define forint(s, n) for(int i=s; i<n; i++)
#define forauto(vec) for(auto & i : vec)

#define LOG_BUILD {                                                                                           \
	const char *buildString = "This build " __FILE__ " was compiled at " __DATE__ ", " __TIME__ ".\n";          \
	printf("#### BUILD INFO: %s", buildString);                                                                 \
}

#define randint(min, max) (min + (rand() % static_cast<int>(max - min + 1)))




#define DATA 0
#define RESULT 1
#define FINISH 2

#define ID -1

#define TIME_MEASURE(function, ...){                                                                           \
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();      \
    function(__VA_ARGS__);                                                                                           \
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();                        \
    std::cout << duration << std::endl;                                                                 \
}


#endif //MYMPICUDATEST_STD_UTILS_H
