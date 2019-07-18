#!/usr/bin/env bash



dos2unix run_remote.sh

scp ../cbuild/MyCudaTest a@M:/home/a/W/Cuda/cuda_clion_tutorial/cbuild/MyCudaTest

export DISPLAY=L0:0.0

ssh -4YCA a@10.10.1.1 < run_remote.sh
