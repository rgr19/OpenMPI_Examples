#!/bin/bash
set -x

export DISPLAY=L0:0.0

export PROJECT_ROOT="/home/a/W/Cuda/cuda_clion_tutorial"
export PROJECT_NAME="CudaUtilsTest"
export PROJECT_BUILDDIR="cbuild"
export PROJECT_PATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}
export PROJECT_BUILDPATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}/${PROJECT_NAME}

cd ${PROJECT_PATH}



chmod 755 ${PROJECT_NAME}
#${PROJECT_BUILDPATH}

sleep 2

./${PROJECT_NAME}

sleep 10