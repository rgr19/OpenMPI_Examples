CUDA_INCLUDEPATH=/usr/local/cuda/include
NVCC_OPTS=-O3 -arch=sm_61 -Xcompiler -Wall -Xcompiler -Wextra -m64

MyCudaTutorial: test.cuda_tutorial.cpp reduce.o Makefile
	nvcc -o MyCudaTutorial test.cuda_tutorial.cpp reduce.o $(NVCC_OPTS)

reduce.o: reduce.cu
	nvcc -c reduce.cu $(NVCC_OPTS)

clean:
	rm -f *.o MyCudaTutorial