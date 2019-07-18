cmake_minimum_required(VERSION 3.11)
project(MyCpp LANGUAGES CXX)
message(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(CppUtils)

set(CPP_SOURCE_FILES
        AData.h
        AData.cpp
        ACalc.h
        ACalc.cpp
        )

add_library(${PROJECT_NAME} STATIC ${CPP_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} CppUtils)

project(MyCppTest)

add_executable(${PROJECT_NAME} test.cpp.cpp)

target_link_libraries(${PROJECT_NAME} MyCpp CppUtils)