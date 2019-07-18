//
// Created by robgrzel on 03.07.17.
//

#ifndef EXAM_ENV_H
#define EXAM_ENV_H

#include "const.h"
#include "math.h"


enum wave_types {
	regularWaves = 0,
	irregularWaves = 1,
};

enum env_cond_types {
	waves = 0,
	current = 1,
	wind = 2,
	wavesCurrent = 3,
	wavesWind = 4,
	currentWind = 5,
	currentWavesWind = 6,
};

enum struct_case_types {
	verticalMonopile = 1,
	verticalBuoy = 2,
};

enum constrain_types {
	constFull = -3,
	constTXYZ = -2,
	constRXYZ = -1,
	constNone = 0,
	constX = 1,
	constY = 2,
	constZ = 3,
	constRX = 4,
	constRY = 5,
	constRZ = 6
	
};

enum wave_forces_solution_formulas {
	potentialTheory = 0,
	morisonEquation = 1,
	macCamyFuchsDifr = 2,
};

enum surface_smoothness {
	surfaceSmooth = 0,
	surfaceRought = 1,
};


enum dof_types {
	DOFX = 0,
	DOFY = 1,
	DOFZ = 2,
	DOFRX = 3,
	DOFRY = 4,
	DOFRZ = 5
};


enum wave_load_types {
	I_allInertia = 0,
	II_diffractionRegion = 1,
	III_largeInertiaSmallDrag = 2,
	IV_diffractionRegion = 3,
	V_inertiaAndDrag = 4,
	VI_largeDrag = 5,
	VII_breakingWave = 6,
	X_errorType = -1,
};

enum structure_volume_types {
	smallVolume = 0,
	largeVolume = 1,
	errorVolume = -1,
};

enum cylinder_sect_types{
	sectionCircle = 0,
	sectionElipsisV = 1,
	sectionElipsisH = 2,
	sectionLineH = 3,
	sectionCircularCylinder2FinsV = 4,
	sectionCircularCylinder2FinsH = 5,
	sectionCross = 6,
	sectionSquareV = 7,
	sectionRectangleV = 8,
	sectionRombV = 9,
	sectionCircleOnWall = 10,
	sectionCircleHalfInWall=11,
	sectionHH = 12,
	sectionCirclePairAMovingBFixed = 13,
	sectionCylinderInPipe = 14,
	sectionSquareRoundCorners = 15,
	sectionSquareInShallowWater = 16,
	
	
};

enum cylinder_body_types{
	//V - vertical, A - axial, L - lateral
	bodyCircularDiscV = 0,
	bodyEllipticalDiscV = 1,
	bodyRectangularPlatesV = 2,
	bodyTriangularPlatesV = 3,
	bodySpheres = 4,
	bodySpheroidsLorA = 5,
	bodyEllipsoidA = 6,
	bodySquarePrismsV = 7,
	bodyRightCircularCylinderV = 8,
};

void wave_loads_regimes(double D, double waveHeight, double waveLenght, wave_load_types & waveLoadType);

void structure_volume_type(const wave_load_types & waveLoadType, structure_volume_types & volumeType);


double added_mass_coef(cylinder_body_types & btype, double a, double b = 0);
double added_mass_coef(cylinder_sect_types & stype, double a, double b = 0);

double drag_coef();


#endif //EXAM_ENV_H
