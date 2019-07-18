
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(MyDeepLearningTest LANGUAGES CXX CUDA)
message(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)


set(CPP_SOURCE_FILES
        cifar10_reader.hpp
        test.deeplearning.cpp
        )


add_executable(${PROJECT_NAME}  ${CPP_SOURCE_FILES})

if (USE_CV2)

    ###########################
    # LINK CV2
    ###########################
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

endif (USE_CV2)

