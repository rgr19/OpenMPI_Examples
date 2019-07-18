cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(CudaUtils LANGUAGES CXX CUDA)
message(${PROJECT_NAME})


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(CUDA_SOURCE_FILES
        containers/thrust_vector_2d.cuh
        containers/thrust_vector_2d.cu
        #cuda_dummy.h
        )

cuda_add_library(${PROJECT_NAME} STATIC ${CUDA_SOURCE_FILES})

message(DEBUG_LEVELS: ${DEBUG_LEVELS})

target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA ${DEBUG_LEVELS})

#########################################################
# LINK
#########################################################

target_link_libraries(${PROJECT_NAME} MyCpp)

target_link_libraries(${PROJECT_NAME} cuda cudart)

##################################################################################################################
# TEST
##################################################################################################################

project(CudaUtilsTest LANGUAGES CXX CUDA)

cuda_add_executable(${PROJECT_NAME} test.cudautils.cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} CudaUtils)

target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA) #defines macro variable in code

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

foreach (dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach ()

