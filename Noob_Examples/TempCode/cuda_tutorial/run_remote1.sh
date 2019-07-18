#!/bin/bash
set -x
set -e


export PROJECT_ROOT="/home/a/W/Cuda/cuda_clion_tutorial"
export PROJECT_NAME="MyCudaTutorial"
export PROJECT_BUILDDIR="cbuild"
export PROJECT_PATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}
export SUBPROJECT_PATH=${PROJECT_ROOT}/"cuda_tutorial"
export PROJECT_BUILDPATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}/${PROJECT_NAME}

export DISPLAY=L0:0.0

cd ${SUBPROJECT_PATH}

#make && chmod 755 ${PROJECT_NAME} && ./${PROJECT_NAME}

#-x=cu \

time nvcc \
  -std=c++11 \
  --expt-extended-lambda \
  -gencode arch=compute_61,code=compute_61 \
  -DUSE_CUDA=0 \
  -I ./ \
  -I /usr/local/cuda/samples/common/inc \
  -lcuda -lcurand -lcublas\
  -o hello \
  test.cuda_tutorial.cu



chmod 755 hello
./hello

sleep 1

nvprof ./hello --system-profiling on --print-gpu-trace on --profile-from-start off



cd ${PROJECT_PATH} && chmod 755 ${PROJECT_NAME}
#${PROJECT_BUILDPATH}

sleep 1

# ./${PROJECT_NAME}
#nvprof ./${PROJECT_NAME}
#LONG OUTPUT
#nvprof -o prof.nvvp --print-gpu-trace ./${PROJECT_NAME}
#nvvp prof.nvvp #crash on xming

sleep 20

