#!/usr/bin/env bash



dos2unix run_remote.sh

scp ../../cbuild/CudaUtilsTest a@M:/home/a/W/Cuda/cuda_clion_tutorial/cbuild/CudaUtilsTest

export DISPLAY=L0:0.0

ssh -4XYCA a@10.10.1.1 < run_remote.sh

sleep 10