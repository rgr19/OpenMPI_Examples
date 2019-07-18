//
// Created by user on 7/29/2018.
//

#ifndef MYMPICUDATEST_CUDA_THRUST_TEST_CUH_H
#define MYMPICUDATEST_CUDA_THRUST_TEST_CUH_H

#ifdef USE_THRUST

int test_thrust() {
	
	
	test_matvec();
	
	return 0;
	
	size_t n = 256 * 1024;
	const float alpha = 0.5f;
	int nerror;
	
	thr::host_vector<float> hVecX;
	thr::host_vector<float> hVecY;
	thr::host_vector<float> hVecYtmp;
	thr::host_vector<float> hVecY1tmp;
	thr::device_vector<float> dVecX;
	thr::device_vector<float> dVecX1;
	thr::device_vector<float> dVecY;
	thr::device_vector<float> dVecY1;
	thr::device_vector<float> dVecAlpha;
	
	THRUST_HVEC(hVecX, float, n, 0);
	THRUST_HVEC(hVecY, float, n, 0);
	THRUST_HVEC(hVecYtmp, float, n, 0);
	THRUST_HVEC(hVecY1tmp, float, n, 0);
	THRUST_DVEC(dVecX, float, n, 0);
	THRUST_DVEC(dVecY, float, n, 0);
	THRUST_DVEC(dVecX1, float, n, 0);
	THRUST_DVEC(dVecY1, float, n, 0);
	THRUST_DVEC(dVecAlpha, float, n, alpha);
	
	int j = 0;
	for (auto &i : hVecX) i = j++;
	
	j = 0;
	for (auto &i : hVecY) i = j++;
	
	THRUST_COPY(hVecX, dVecX);
	THRUST_COPY(hVecX, dVecX1);
	THRUST_COPY(hVecY, dVecY);
	THRUST_COPY(hVecY, dVecY1);
	
	size_t blockSize = 512;
	size_t nBlocks = n / blockSize + (n % blockSize > 0);
	
	std::cout << "blockSize: " << blockSize << ", nBlocks: " << nBlocks << std::endl;
	
	const float *dX = thrust::raw_pointer_cast(&dVecX[0]);
	idemn float *dY = thrust::raw_pointer_cast(&dVecY[0]);
	
	
	const float *dX1 = thrust::raw_pointer_cast(&dVecX1[0]);
	idemn float *dY1 = thrust::raw_pointer_cast(&dVecY1[0]);
	
	
	const float *hX = thrust::raw_pointer_cast(&hVecX[0]);
	idemn float *hY = thrust::raw_pointer_cast(&hVecY[0]);
	
	const float *dAlpha = thrust::raw_pointer_cast(&dVecAlpha[0]);
	
	
	TIMEIT_1000_CUDA(CUDA_KERNEL, saxpy_gpu, nBlocks, blockSize, n, alpha, dX, dY);
	
	
	TIMEIT_1000(saxpy_cpu, n, alpha, hX, hY);
	
	
	//cublasInit();
	
	cublasHandle_t cublasHand;
	cublasCreate(&cublasHand);
	
	TIMEIT_1000_CUDA(cublasSaxpy, cublasHand, n, &alpha, dX1, 1, dY1, 1);
	cublasDestroy(cublasHand);
	
	THRUST_COPY(dVecY, hVecYtmp);
	
	THRUST_COPY(dVecY1, hVecY1tmp);
	
	
	fori(0, n) {
		if (!int(hVecY[i]) ^ int(hVecYtmp[i]) ^ int(hVecY1tmp[i])) {
			printf("RESULTS DIFFER: ERROR! [i:%d] h:%f != d:%f != c:%f\n",
			       i, hVecY[i], hVecYtmp[i], hVecY1tmp[i]);
		}
	}
	
	
	
	//for (auto & i : dataset.test_labels) std::cout << i <<std::endl;
	
	
	return 0;
}


#endif


#endif //MYMPICUDATEST_CUDA_THRUST_TEST_CUH_H
