cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MyCuda LANGUAGES CXX CUDA)
message(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)


message(EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH})
message(LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH})
message(CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR})



message(EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH})
message(LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH})
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})
message(CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD})
message(CUDA_LIBRARY_DIRS:${CUDA_LIBRARY_DIRS})
message(CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS})


#########################################################
# DIRS
#########################################################


add_subdirectory(CudaUtils)


#########################################################
# ADD EXECUTABLE
#########################################################

set(CUDA_SOURCE_FILES
        ACalcCuda.cuh
        ACalcCuda.cu
        )

cuda_add_library(${PROJECT_NAME} ${CUDA_SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA)


#########################################################
# LINK
#########################################################

target_link_libraries(${PROJECT_NAME} MyCpp CudaUtils)
target_link_libraries(${PROJECT_NAME} cuda cudart)# curand cudnn)

##################################################################################################################
# TEST
##################################################################################################################

project(MyCudaTest LANGUAGES CXX CUDA)

cuda_add_executable(${PROJECT_NAME} test.cuda.cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} MyCuda)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA) #defines macro variable in code

