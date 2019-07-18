cmake_minimum_required(VERSION 3.11)
project(CppUtils)
message(${PROJECT_NAME})


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)


set(CPP_SOURCE_FILES
        utils.h
        template_utils.cpp
        template_utils.h
        Containers/std_utils.h
        Containers/std_vector_3d.h
        Containers/std_vector_3d.cpp
        Containers/std_vector_2d.cpp
        Containers/std_vector_2d.h
        debug_printing.h
        )

add_library(${PROJECT_NAME} STATIC ${CPP_SOURCE_FILES})

message(DEBUG_LEVELS: ${DEBUG_LEVELS})

target_compile_definitions(${PROJECT_NAME} PUBLIC ${DEBUG_LEVELS})
