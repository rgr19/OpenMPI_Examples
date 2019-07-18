//
// Created by robgrzel on 05.07.17.
//

#include "env.h"


double added_mass_coef( cylinder_body_types & btype, double a, double b = 0) {
	/*
	 The added mass coefficient, Ca, is a function of the
	size and shape of the body, the direction in which it is 0
	moved through the fluid with respect to an axis in the body,
	and the density and viscosity of the fluid. It can be
	calculated using inviscid incompressible irrotational flow
	theory by equating the rate of change of the total kinetic
	energy of the fluid and body to the rate at which the
	force driving the body does work. The body experiences a
	resistance to its motion of amount MU. The added mass is
	the same whether the fluid is accelerating past the body
	or the body is accelerating in the fluid. The difference
	between the two situations is the appearance of the dynamic
	buoyancy force pVtJ mentioned above in the case of the
	fluid moving past the body. In some papers, where added
	mass is defined to be 1 + Ca, then, it is indeed different
	for the two cases. Values of Ca for various shapes have
	 been calculated: e.g. for a sphere k = 0.5 (Cm = 1.5) and
	for a right circular cylinder k = 1.0 (Cm = 2.0). In the
	case of flat plates a nominal volume, such as a volume of
	revolution, is often used in performing the calculation.
	 */
	
	
	double r, Ca;
	
	switch (btype) {
		case (bodyCircularDiscV): {
			Ca = 2 / PI;
			return Ca;
			
		}
		case (bodyEllipticalDiscV): {
			r = b / a;
			
			if (r <= 1.) {
				Ca = 0.637;
			} elif (r > 14.3) {
				Ca = 1.;
			} else {
				
				double A[7] = {
						2.29083781e-01,
						5.73963846e-01,
						-1.95547137e-01,
						3.52349275e-02,
						-3.43039031e-03,
						1.69822269e-04,
						-3.34414153e-06
				};
				
				Ca = calc_poly(, r, A);
			}
			
			return Ca;
		}
		case (bodyRectangularPlatesV): {
			r = b / a;
			
			if (r <= 1.) {
				Ca = 0.579;
			} elif (r > 10) {
				Ca = 1;
			} else {
				
				
				double A[5] = {
						3.12945683e-01,
						3.50492145e-01,
						-8.12158738e-02,
						8.48660276e-03,
						-3.23399970e-04
				};
				
				
				Ca = calc_poly(5, r, A);
			}
			return Ca;
			
		}
		case (bodySquarePrismsV) : {
			r = b / a;
			
			if (r <= 1.) {
				Ca = 0.68;
			} elif (r > 10) {
				Ca = 0.08;
			} else {
				double A[5] = {1.01401612e+00,
				               -4.60956692e-01,
				               9.03907524e-02,
				               -7.87199509e-03,
				               2.50754148e-04};
				
				
				Ca = calc_poly(5, r, A);
			}
			return Ca;
			
		}
		case (bodyRightCircularCylinderV): {
			//b = L, a = R = D/2
			r = b / 2 / a;
			
			if (r <= 1.2) {
				Ca = 0.62;
			} elif (r > 9) {
				Ca = 1;
			} else {
				double A[6] = {5.37627289e-01,
				               1.36019614e-01,
				               -1.80773191e-02,
				               1.27679354e-03,
				               -4.68882635e-05,
				               7.03336076e-07};
				
				Ca = calc_poly(6, r, A);
				
			}
			return Ca;
			
		}
		
	}
};

double added_mass_coef(cylinder_sect_types & stype, double a, double b = 0){
	double r, Ca = 1;
	switch(stype){
		case(sectionCircle):{
			return Ca;
		} case (sectionElipsisV):{
			return Ca;
		} case (sectionElipsisH):{
			return Ca;
		} case (sectionLineH): {
			return Ca;
		} case (sectionCircularCylinder2FinsV):{
			return Ca;
		} case (sectionCircularCylinder2FinsH):{
			r = a/b;
			Ca = 1 - pow(r,2) + pow(r,4);
			return Ca;
		}
	}
	
};


double drag_coef_steady_flow(double Re) {
	
	double A[131] = {
			1.05107479e+000, -7.49778999e-001, 1.34663514e-001, 7.39525346e-001,
			-9.61761624e-001, -1.49445981e+000, 3.15660086e+000, 6.83095926e-002,
			-3.35542094e+000, 2.11763856e+000, 5.05726993e-001, -1.18336715e+000,
			6.01547507e-001, -1.37885590e-001, 9.40816826e-003, 1.56627187e-003,
			-1.31155240e-004, -3.39405058e-005, -2.72824421e-007, 5.90942865e-007,
			7.65711758e-008, -4.88572308e-010, -1.51044132e-009, -2.37080804e-010,
			-1.24507338e-011, 2.24507360e-012, 6.64717256e-013, 8.76726271e-014,
			5.01513471e-015, -6.44716068e-016, -2.34227117e-016, -3.84158198e-017,
			-3.96332952e-018, -1.45558153e-019, 4.37816902e-020, 1.29456177e-020,
			2.17156549e-021, 2.59936506e-022, 1.95782806e-023, -3.72193811e-025,
			-4.73231917e-025, -1.08205222e-025, -1.73058658e-026, -2.15818367e-027,
			-1.97067780e-028, -6.88220876e-030, 2.13907704e-030, 6.75191828e-031,
			1.28973874e-031, 1.95402850e-032, 2.44905922e-033, 2.42665681e-034,
			1.41659144e-035, -1.13616585e-036, -5.73673748e-037, -1.26296881e-037,
			-2.15228738e-038, -3.10993683e-039, -3.87556583e-040, -3.99058853e-041,
			-2.80677314e-042, 5.20076362e-044, 6.93677462e-044, 1.72632596e-044,
			3.18564122e-045, 5.01050139e-046, 6.95622183e-047, 8.52742724e-048,
			8.85329713e-049, 6.61794349e-050, 3.79092713e-053, -1.30652984e-051,
			-3.49302612e-052, -6.74607685e-053, -1.11033669e-053, -1.63032606e-054,
			-2.15872332e-055, -2.54324219e-056, -2.52975873e-057, -1.71953143e-058,
			4.24271613e-060, 4.57774889e-060, 1.14990530e-060, 2.18285952e-061,
			3.59046842e-062, 5.32291357e-063, 7.18728964e-064, 8.74097058e-065,
			9.22439667e-066, 7.41826382e-067, 1.51912711e-068, -1.07760793e-068,
			-3.23237100e-069, -6.58586021e-070, -1.13429440e-070, -1.74838775e-071,
			-2.45575872e-072, -3.12501998e-073, -3.52202757e-074, -3.23013463e-075,
			-1.63025153e-076, 2.12747163e-077, 9.09533828e-078, 2.03957019e-078,
			3.68695801e-079, 5.85961835e-080, 8.38174318e-081, 1.07804878e-081,
			1.21002467e-082, 1.06683170e-083, 4.07647257e-085, -1.09706994e-085,
			-3.82696798e-086, -8.09988873e-087, -1.39748872e-087, -2.09942764e-088,
			-2.74952515e-089, -3.00278554e-090, -2.27232660e-091, 2.53069345e-093,
			5.61429476e-093, 1.44535123e-093, 2.64955012e-094, 3.87162630e-095,
			4.31719096e-096, 2.49317247e-097, -3.65519746e-098, -1.56411750e-098,
			-2.96912627e-099, -2.47471592e-100, 6.61927178e-101
	};
	
	double Ca = pow(10, calc_poly(131, log10(Re), A));
	
	
	return Ca;
	
	
}

double inertia_coef_steady_flow(double D, double wavelen) {
	double A[11] = {
			2.2176837,
			1.21096525,
			7.85620815,
			-168.69882255,
			547.38250749,
			-656.47855961,
			-24.75443785,
			850.58892757,
			-848.40828878,
			330.35458748,
			-40.94670778
	};
	
	double r = D / wavelen;
	
	double Cm = calc_poly(11, r, A);
	
	return Cm;
	
}


void wave_loads_regimes(double D, double waveHeight, double waveLenght, wave_load_types & waveLoadType) {
	/*
	
	 Recommended Practice DNV-RP-H103, April 2011
	 Page 16 – 2. General Methods of Analysis
	 Figure 2-3
	Different wave force regimes (Chakrabarti, 1987).
	D = characteristic dimension, H = wave height, λ = wave length.
	 */
	
	double y = log10(waveHeight / D);
	double x = log10(PI * D / waveLenght);
	
	double xr[7] = {-2.002927,
	                -1.660488,
	                -0.539512,
	                -0.314146,
	                0.244878,
	                1.000004,
	                1.000004};
	double yr[7] = {1.650991,
	                1.309848,
	                0.177268,
	                -0.035615,
	                -0.603267,
	                -1.351057,
	                -2.004365};
	
	double x0 = -2;
	double x1 = 1;
	double y0 = 1.651;
	double y1 = -1.3511;
	
	int s = -line_side(x0, y0, x1, y1, x, y);
	
	wave_load_types regimes0[3] = {VI_largeDrag,
	                               V_inertiaAndDrag,
	                               III_largeInertiaSmallDrag};
	wave_load_types regimes1[2][2] = {
			{III_largeInertiaSmallDrag, IV_diffractionRegion},
			{I_allInertia,              II_diffractionRegion}
	};
	
	waveLoadType = X_errorType;
	
	if (s == 1)
		waveLoadType = VII_breakingWave;
	else {
		for (int i = 0; i < 3; i++) {
			if (yr[i] > y and y >= yr[i + 1]) {
				s = line_side(xr[i], yr[i], xr[i + 1], yr[i + 1], x, y);
				if (s == 1)
					waveLoadType = regimes0[i];
				else
					waveLoadType = VII_breakingWave;
				return;
			}
		}
		for (int i = 3; i < 5; i++) {
			if (yr[i] > y and y >= yr[i + 1]) {
				if (x < xr[i])
					waveLoadType = regimes1[i - 3][0];
				else {
					s = line_side(xr[i], yr[i], xr[i + 1], yr[i + 1], x, y);
					if (s == 1)
						waveLoadType = regimes1[i - 3][1];
					else
						waveLoadType = VII_breakingWave;
				}
				return;
			}
		}
		int i = 5;
		if (yr[i] > y and y >= yr[i + 1]) {
			if (x < xr[3])
				waveLoadType = I_allInertia;
			else if (x < xr[5])
				waveLoadType = II_diffractionRegion;
			else
				waveLoadType = VII_breakingWave;
			return;
		}
	}
	
	return;
}


void structure_volume_type(const wave_load_types waveLoadType, structure_volume_types & volumeType) {
	/*
	 Recommended Practice DNV-RP-H103, April 2011
	 Page 16 – 2. General Methods of Analysis
		2.3 Wave loads on large volume structures
		2.3.1 Introduction
		2.3.1.1 Offshore structures are normally characterized as either large volume structures or small volume
		structures. For a large volume structure the structure’s ability to create waves is important in the force
		calculations while for small volume structures this is negligible.
		2.3.1.2 Small volume structures may be divided into structures dominated by drag force and structures
		dominated by inertia (mass) force. Figure 2-3 shows the different regimes where area I, III, V and VI covers
		small volume structures.
		2.3.1.3 The term large volume structure is used for structures with dimensions D on the same order of
		magnitude as typical wave lengths λ of ocean waves exciting the structure, usually D > λ/ 6. This corresponds
		to the diffraction wave force regimes II and IV shown in Figure 2-3 below where this boundary is equivalently
		defined as π D / λ > 0.5.
	 
		H - wave height,  = 2*Amp
		
	 */
	switch (waveLoadType) {
		case (I_allInertia) : {
			volumeType = smallVolume;
			return;
		}
		case (II_diffractionRegion) : {
			volumeType = largeVolume;
			return;
		}
		case (III_largeInertiaSmallDrag) : {
			volumeType = smallVolume;
			return;
		}
		case (IV_diffractionRegion) : {
			volumeType = largeVolume;
			return;
		}
		case (V_inertiaAndDrag) : {
			volumeType = smallVolume;
			return;
		}
		case (VI_largeDrag) : {
			volumeType = smallVolume;
			return;
		}
		default:
			volumeType = errorVolume;
			return;
	}
	
}


void wave_forces_select_by_regimes(const wave_load_types & waveLoadType, const structure_volume_types & volumeType){

	switch (volumeType){
		case (smallVolume) : {
			switch (waveLoadType) {
				case (I_allInertia):{
				
				} case (III_largeInertiaSmallDrag):{
				
				}
			}
		} case (largeVolume):{
			switch (waveLoadType){
				case (II_diffractionRegion) :{
				
				} case (IV_diffractionRegion):{
				
				}
			}
		}
	}
	
;}

void
wave_forces_select(double D, double wavelen, double KC, double Re, surface_smoothness & ss, double & Cm, double & Cd) {
	/*
	 * Geir Moe IEAWIND
	 Part II – Chapter 6 – Section 6 - Linear wave theory 257
		6.6.6 Which Formula to Use to Find the Wave Forces
			Fig. 6.40 shows how to decide whether the flow is separated or whether potential theory can be
			applied. For the latter case the mass coefficient Cm is 2 if the diameter is less than 0.2 times the
			wavelength. Otherwise Cm may be determined from Fig. 6.41
			
	 * WAVE FORCES ON CYLINDERS - ANTHONY GRAHAM DIXON
	An implicit assumption inherent in equation (1.3) is
	that the motion of the fluid can be characterised by a
	single acceleration, U, that is constant over the region
	of the body. Therefore, in the application of the added
	mass concept to the continuously varying accelerations
	(in time and space) of waves, we are restricted to
	situations where the body is small compared to the
	wavelength, so that we can consider the flow field to have
	an approximately constant.accele.ration over the region of
	the body. For larger bodies than this the incident wave
	will be considerably scattered and we must turn to
	diffraction theory for the evaluation of the forces. This
	will be discussed in a later section.
	 */
	
	if (KC > 2) {
		//MORISON FORMULA
		switch (ss) {
			case (surfaceRought) : {
				//fig 6.38, 6.39
				return;
			}
			case (surfaceSmooth) : {
				//fig 6.36
				//fig 6.37
				return;
			}
		}
	} else {
		//POTENTIAL THEORY
		if (D < 0.2 * wavelen) {
			Cm = 2;
			return;
		} else {
			//fig 6.41
			Cm = inertia_coef_steady_flow(D, wavelen);
			//Cd = drag_coef_steady_flow(Re);
			return;
		}
	}
	
}