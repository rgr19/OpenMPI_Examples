
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(MyHopfieldTest LANGUAGES CXX CUDA)
message(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(USE_CUDA 1)


set(CPP_SOURCE_FILES
        cuda_dummy.h
        hopfield/hopfield.cu
        hopfield/hopfield.cuh
        hopfield/hopfield.test.cpp
        )


if (USE_CUDA)
    message(USE_CUDA)

    #########################################################
    # FIND CUDA
    #########################################################

    INCLUDE(FindCUDA)

    find_package(CUDA REQUIRED)

    message(CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS})

    include_directories(${CUDA_INCLUDE_DIRS})

    IF (${CUDA_LIBRARY_DIRS})
        message(CUDA_LIBRARY_DIRS0:${CUDA_LIBRARY_DIRS})
    ELSE ()
        set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
    ENDIF (${CUDA_LIBRARY_DIRS})

    link_directories(${CUDA_LIBRARY_DIRS})

    set(CUDA_ARCH_60 "-gencode arch=compute_60,code=sm_61")
    set(CUDA_ARGS "-pg -g -lineinfo --std=c++14 -O3 -use_fast_math")
    #    set(CUDA_ARGS " --std=c++14 -O0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_60} ${CUDA_ARGS}")
    #crash when --device-debug -G is on
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_60}")

    message(CUDA_LIBRARY_DIRS:${CUDA_LIBRARY_DIRS})
    message(CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS})


else (USE_CUDA)


endif (USE_CUDA)


cuda_add_executable(${PROJECT_NAME}  ${CPP_SOURCE_FILES})


if (USE_CUDA)
    ###########################
    # LINK CUDA
    ###########################
    message(LINK CUDA)

    #target_link_libraries(${PROJECT_NAME} cuda cudart curand cudnn)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA USE_CUDNN USE_THRUST) #equal to macro #define USE_CUDA...


endif (USE_CUDA)
