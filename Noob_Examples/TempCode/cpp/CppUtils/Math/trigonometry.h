//
// Created by robgrzel on 03.07.17.
//

#ifndef EXAM_TRIGONOMETRY_H
#define EXAM_TRIGONOMETRY_H

enum normal_types {
	nxType = 0,
	nyType = 1,
	nzType = 2
};

const double normal[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

void normal_vector(normal_types normalType, double * normal_) {
	
	normal_[0]=normal[normalType][0];
	normal_[1]=normal[normalType][1];
	normal_[2]=normal[normalType][2];
	
}


#endif //EXAM_TRIGONOMETRY_H
