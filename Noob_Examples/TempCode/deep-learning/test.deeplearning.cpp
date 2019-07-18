//
// Created by user on 7/26/2018.
//
#include "cifar10_reader.hpp"

int main(){
	
	cifar::datasetPath = "../h_x/cifar/cifar-10-batches-bin";
	
	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	
	
	//for (auto & i : dataset.test_labels) std::cout << i <<std::endl;
	
	
}
