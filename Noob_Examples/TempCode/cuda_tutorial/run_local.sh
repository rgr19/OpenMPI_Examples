#!/usr/bin/env bash
set -x

ls


dos2unix run_remote.sh

scp * a@M:/home/a/W/Cuda/cuda_clion_tutorial/cuda_tutorial

scp ../cbuild/MyCudaTutorial a@M:/home/a/W/Cuda/cuda_clion_tutorial/cbuild/MyCudaTutorial

export DISPLAY=L0:0.0

ssh -4YCA a@10.10.1.1 < run_remote1.sh
