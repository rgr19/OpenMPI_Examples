//#include "stdafx.h"
#include "math.h"

// Very fast sqrt

double calc_poly(int n, double x, double * a){
	double y = 0;
	
	for (int i = 0; i < n; i++) {
		y += a[i] * pow(x, i);
	}
	
	return y;
}


int polyfit(
		const double* const X, //x
		const double* const Y, //y
		const unsigned int n,
		const unsigned int order,
		double* coefficients
) {
	// Declarations...
	// ----------------------------------
	enum {
		maxOrder = 11
	};

	double B[maxOrder + 1]                        = {0.0f};
	double P[((maxOrder + 1) * 2) + 1]            = {0.0f};
	double A[(maxOrder + 1) * 2 * (maxOrder + 1)] = {0.0f};

	double x, y, powx;

	unsigned int ii, jj, kk;

	// Verify initial conditions....
	// ----------------------------------

	// This method requires that the n >
	// (order+1)
	if (n <= order) {
		return -1;
	}

	// This method has imposed an arbitrary bound of
	// order <= maxOrder.  Increase maxOrder if necessary.
	if (order > maxOrder) {
		return -1;
	}

	// Begin Code...
	// ----------------------------------

	// Identify the column vector
	for ( ii = 0; ii < n; ii++ ) {
		x    = X[ii];
		y    = Y[ii];
		powx = 1;

		for ( jj = 0; jj < (order + 1); jj++ ) {
			B[jj] = B[jj] + (y * powx);
			powx = powx * x;
		}
	}

	// Initialize the PowX array
	P[0] = n;

	// Compute the sum of the Powers of x
	for ( ii = 0; ii < n; ii++ ) {
		x    = X[ii];
		powx = X[ii];

		for ( jj = 1; jj < ((2 * (order + 1)) + 1); jj++ ) {
			P[jj] = P[jj] + powx;
			powx = powx * x;
		}
	}

	// Initialize the reduction matrix
	//
	for ( ii = 0; ii < (order + 1); ii++ ) {
		for ( jj = 0; jj < (order + 1); jj++ ) {
			A[(ii * (2 * (order + 1))) + jj] = P[ii + jj];
		}

		A[(ii * (2 * (order + 1))) + (ii + (order + 1))] = 1;
	}

	// Move the Identity matrix portion of the redux matrix
	// to the left side (find the inverse of the left side
	// of the redux matrix
	for ( ii = 0; ii < (order + 1); ii++ ) {
		x = A[(ii * (2 * (order + 1))) + ii];
		if (x != 0) {
			for ( kk = 0; kk < (2 * (order + 1)); kk++ ) {
				A[(ii * (2 * (order + 1))) + kk] =
						A[(ii * (2 * (order + 1))) + kk] / x;
			}

			for ( jj = 0; jj < (order + 1); jj++ ) {
				if ((jj - ii) != 0) {
					y        = A[(jj * (2 * (order + 1))) + ii];
					for ( kk = 0; kk < (2 * (order + 1)); kk++ ) {
						A[(jj * (2 * (order + 1))) + kk] =
								A[(jj * (2 * (order + 1))) + kk] -
								y * A[(ii * (2 * (order + 1))) + kk];
					}
				}
			}
		} else {
			// Cannot work with singular matrices
			return -1;
		}
	}

	// Calculate and Identify the coefficients
	for ( ii = 0; ii < (order + 1); ii++ ) {
		for ( jj = 0; jj < (order + 1); jj++ ) {
			x        = 0;
			for ( kk = 0; kk < (order + 1); kk++ ) {
				x = x + (A[(ii * (2 * (order + 1))) + (kk + (order + 1))] *
				         B[kk]);
			}
			coefficients[ii] = x;
		}
	}

	return 0;
}

double dround(double v, int n){
	return round( v * pow(10,n) ) / pow(10,n);
}

int iround( double r ) {
	return (r > 0.0) ? (r + 0.5) : (r - 0.5);
}

long double simpshy( const double& h, const std::vector<double>& y ) {

	const long  n   = int( y.size() );
	const bool  odd = bool( n % 2 );
	long double sum = 0;

	if (n == 1) {
		return .5 * h * (y.front() + y.back());

	} else {

		long m = n;

		if (odd) {
			sum = 3.0 / 8.0 * h * (y[n - 3] + 3.0 * (y[n - 2] + y[n - 1]) + y[n]);
			m   = n - 3;
		}

		if (m > 1) {

			double sub = y.front();

			for ( int i = 1; i < m - 2; i += 2 ) {
				sub += 4.0 * y[i] + 2.0 * y[i + 1];
			}
			sub += 4.0 * y[m - 1] + y[m];
			sum += h * sub / 3.0;
		}

		return sum;
	}
}

long double simpsxy( const std::vector<double>& X, const std::vector<double>& Y ) {

	size_t n = X.size();

	double x1, x2, x3, y1, y2, y3;
	double dx21, dx32, dx31;
	double dy21, dy32;
	double dy_dx_21;
	double dy_dx_32;
	double a, b, c, s;

	long double S = 0;
	printf("######################## : n=%zu\n",n);

	if (n < 5) {
		for (int i=1; i<n; i++){
			S += (X[i] - X[i-1]) * (Y[i] + Y[i-1]) * 0.5;
		}
		printf("S=%Lf\n",S);

	} else {

		for ( int i = 2; i < n; i += 2 ) {
			x1   = X[i - 2];
			x2   = X[i - 1];
			x3   = X[i];
			y1   = Y[i - 2];
			y2   = Y[i - 1];
			y3   = Y[i];
			dx21 = x2 - x1;
			dx32 = x3 - x2;
			dx31 = x3 - x1;
			dy21 = y2 - y1;
			dy32 = y3 - y2;

			dy_dx_21 = dy21 / dx21;
			dy_dx_32 = dy32 / dx32;

			a = (dy_dx_32 - dy_dx_21) / dx31;
			b = dy_dx_32 - a * (x2 + x3);
			c = y1 - a * pow( x1, 2 ) - b * x1;

			s = a / 3 * (pow( x3, 3 ) - pow( x1, 3 )) + b / 2 * (pow( x3, 2 ) - pow( x1, 2 )) + c * (x3 - x1);

			printf("x=[%f,%f,%f], y=[%f,%f,%f], dxc=[%f,%f,%f], dy=[%f,%f], dy_dx=[%f,%f], c=[%f,%f,%f], s=%f/%Lf\n",
			      x1,x2,x3,y1,y2,y3,dx21,dx32,dx31,dy21,dy32,dy_dx_21,dy_dx_32,a,b,c,s,S
			      );
			S += s;
		}
	}

	return S;

}

double mind( std::vector<double> x ) {
	double min = *std::min_element( x.begin(), x.end() );
	return min;
}

double maxd( std::vector<double> x ) {
	double max = *std::max_element( x.begin(), x.end() );
	return max;
}


long arg_mind( std::vector<double> x ) {
	long argMin = std::distance( x.begin(), std::min_element( x.begin(), x.end() ) );
	return argMin;
}


long arg_maxd( std::vector<double> x ) {
	long argMax = std::distance( x.begin(), std::max_element( x.begin(), x.end() ) );
	return argMax;
}


void randn2pi( const size_t N, double* A ) {

	std::mt19937                     gen( 0 );
	//std::default_random_engine       de( 0 ); //seed
	std::normal_distribution<double> nd( 0, 1 ); //mean followed by stdiv

	for ( int i = 0; i < N; ++i ) {
		A[i] = 2 * PI * nd( gen ); //Generate numbers;

	}
}


void randn( const size_t N, double* A ) {

	std::mt19937                     gen( 0 );
	//std::default_random_engine       de( 0 ); //seed
	std::normal_distribution<double> nd( 0, 1 ); //mean followed by stdiv

	for ( int i = 0; i < N; ++i ) {
		A[i] = 2 * nd( gen ); //Generate numbers;

	}
}

float SQRTF( float number ) {
	long        i;
	float       x, y;
	const float threehalfs = 1.5f;

	x = number * 0.5f;
	y = number;
	i = *( long* ) &y;
	i = 0x5F3759DF - (i >> 1);
	y = *( float* ) &i;
	y = y * (threehalfs - (x * y * y));
	y = y * (threehalfs - (x * y * y));
	return number * y;
}

double SQRT( double number ) {
	long         i;
	double       x, y;
	const double threehalfs = 1.5;

	x = number * 0.5;
	y = number;
	i = *( long* ) &y;
	i = 0x5F3759DF - (i >> 1);
	y = *( double* ) &i;
	y = y * (threehalfs - (x * y * y));
	y = y * (threehalfs - (x * y * y));
	return number * y;
}

//Very fast 1/sqrt
float RSQRTF( float number ) {
	long        i;
	float       x, y;
	const float threehalfs = 1.5f;

	x = number * 0.5f;
	y = number;
	i = *( long* ) &y;                    // evil floating point bit level hacking
	i = 0x5F3759DF - (i >> 1);            // what the fuck?
	y = *( float* ) &i;
	y = y * (threehalfs - (x * y * y)); // 1st iteration
	y = y * (threehalfs - (x * y * y)); // 2nd iteration, this can be removed
	return y;
}

double RSQRT( double number ) {
	long         i;
	double       x, y;
	const double threehalfs = 1.5;

	x = number * 0.5;
	y = number;
	i = *( long* ) &y;                    // evil doubleing point bit level hacking
	i = 0x5F3759DF - (i >> 1);            // what the fuck?
	y = *( double* ) &i;
	y = y * (threehalfs - (x * y * y)); // 1st iteration
	y = y * (threehalfs - (x * y * y)); // 2nd iteration, this can be removed
	return y;
}
