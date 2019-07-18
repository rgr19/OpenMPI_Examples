
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(MyCudaTutorial LANGUAGES CXX CUDA)
message(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)


set(CPP_SOURCE_FILES
        std_utils.h
        cuda_reduce.cuh
        cuda_kernels.cuh
        cuda_dummy.h
        cuda_macros.h
        cuda_reduce_benchmark.cuh
        test.cuda_tutorial.cu
        test.cuda_tutorial.cpp
        hopfield_deterministic.cuh
        cuda_tests.cuh
        cuda_mul_AV.cuh
        )


if (USE_CUDA)
    message(USE_CUDA)
    cuda_add_executable(${PROJECT_NAME} ${CPP_SOURCE_FILES})

    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

    set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA_DUMMY)

else (USE_CUDA)
    add_executable(${PROJECT_NAME} ${CPP_SOURCE_FILES})
endif (USE_CUDA)


if (USE_CV2)

    ###########################
    # LINK CV2
    ###########################
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

endif (USE_CV2)


if (USE_CUDA)
    ###########################
    # LINK CUDA
    ###########################
    message(LINK CUDA)

    target_link_libraries(${PROJECT_NAME} cuda cudart curand cublas)


endif (USE_CUDA)

###########################
# LINK MPI/PTHREADS
###########################

if (USE_MPI)
    message(LINK MPI/PTHREADS)

    if (USE_MPI.OPENMPI)
        message("LINK OpenMPI...")
        target_link_libraries(${PROJECT_NAME} /usr/lib/openmpi/lib/libmpi.so)
    elseif (USE_MPI.MPICH)
        message("LINK MPICH...")
        target_link_libraries(${PROJECT_NAME} /usr/lib/mpich/lib/libmpich.so)
    endif (USE_MPI.OPENMPI)

endif (USE_MPI)

if (USE_OMP)
    target_link_libraries(${PROJECT_NAME} -fopenmp)
endif (USE_OMP)


if (CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif ()
