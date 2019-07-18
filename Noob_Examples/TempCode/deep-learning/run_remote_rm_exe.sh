#!/bin/bash
set -x

export PROJECT_ROOT="/home/a/W/Cuda/cuda_clion_tutorial"
export PROJECT_NAME="MyDeepLearningTest"
export PROJECT_BUILDDIR="cbuild"
export PROJECT_PATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}
export PROJECT_BUILDPATH=${PROJECT_ROOT}/${PROJECT_BUILDDIR}/${PROJECT_NAME}

cd ${PROJECT_PATH}

[ ! -e ${PROJECT_NAME} ] || rm ${PROJECT_NAME}
