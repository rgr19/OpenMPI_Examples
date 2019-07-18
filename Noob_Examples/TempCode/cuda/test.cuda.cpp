//
// Created by user on 7/22/2018.
//

#include <iostream>

#include "ACalcCuda.cuh"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>

int main(int argc, char ** argv){
	const char *buildString = "This build " __FILE__ " was compiled at " __DATE__ ", " __TIME__ ".\n";
	printf("#### BUILD INFO: %s", buildString);
	
	ADataGen_t      *pDataGen;
	ACalcCuda_t         *pResults;
	ADataConst_t    *pDataConst;
	ADataIdemn_t     *pDataIdem;
	
	ADataIdemnItem_t oItemIdem[2];
	
	int gpuNum;
	int myrank = 1;
	
	size_t numElems = 1000;
	size_t numWaves = 1000;
	size_t numSteps = 1000;
	int t0 = 100;
	int t1 = 100;
	
	int proccount;
	
	pDataGen = new ADataGen_t(myrank, numElems, numWaves);
	pDataConst = new ADataConst_t(myrank, numWaves);
	pDataIdem = new ADataIdemn_t(myrank, numElems);
	gpuNum = set_cuda(myrank);
	
	
	pDataGen->gen(pDataConst);
	pDataGen->gen(pDataIdem);
	
	
	oItemIdem[0] = pDataIdem->vec[10];
	
	
	pResults = new ACalcCuda_t(myrank, numElems, numSteps, numWaves, t0, t1);
	
	pResults->compute_cuda(*pDataConst, *pDataGen, oItemIdem[0], myrank);
	
	
	cv::Mat src_host;
	
	src_host = cv::imread("tensorflow.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	cv::namedWindow("my image");
	
	cv::imshow("my image", src_host);
	
	cv::waitKey(5000);
	
	return 0;
	
}