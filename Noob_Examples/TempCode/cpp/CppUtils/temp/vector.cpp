

#include "vector.h"

Vector::Vector( ) {
}


Vector::Vector( unsigned int count ) {
	_count = count;
	_data.resize( _count );
}

Vector::~Vector( ) {
	_data.clear();
}

void Vector::randInit( ) {
	for ( unsigned int i = 0; i < _count; i++ ) {
		_data[i] = ( double ) rand() / (3000 * RAND_MAX);
	}
}

void Vector::zeros( ) {
	for ( unsigned int i = 0; i < _count; i++ ) {
		_data[i] = 0;
	}
}

unsigned int Vector::GetCount( ) const {
	return _count;
}

std::vector<double> Vector::GetData( ) const {
	return _data;
}

double Vector::GetItem( unsigned int i ) {
	return _data[i];
}

double Vector::GetMax( ) {
	double maxItem  = *std::max_element( _data.begin(), _data.end() );
	double maxIndex = std::distance( _data.begin(), std::max_element( _data.begin(), _data.end() ) );
	_data[maxIndex] = 0;
	double maxItem2 = *std::max_element( _data.begin(), _data.end() );
	_data[maxIndex] = maxItem;
	if (maxItem - maxItem2 < 0.2) {
		return 1000;
	} else {
		return maxIndex;
	}
}

void Vector::SetData( std::vector<double> data ) {
	_data = data;
}

void Vector::SetData( Vector v ) {
	_data  = v.GetData();
	_count = v.GetCount();
}

void Vector::SetItem( unsigned int i, double value ) {
	_data[i] = value;
}

void Vector::printData( ) {
	for ( unsigned int i = 0; i < _count; i++ ) {
		std::cout << _data[i] << std::endl;
	}
}

void Vector::writeData( std::string name ) {
	std::ofstream ofs;
	ofs.open( name.c_str(), std::ofstream::out | std::ofstream::app );

	ofs << _count << std::endl;
	for ( unsigned int i = 0; i < _count; i++ ) {
		ofs << _data[i] << std::endl;
	}

	ofs.close();
}

Vector Vector::add( Vector v ) {
	if (_count != v.GetCount()) {
		std::cout << "Az osszeadando vektorok nem azonos meretuek." << std::endl;
	}

	Vector             retV( v.GetCount() );
	for ( unsigned int i = 0; i < _count; i++ ) {
		retV.SetItem( i, _data[i] + v.GetItem( i ) );
	}
	return retV;

}

Vector Vector::sub( Vector v ) {
	if (_count != v.GetCount()) {
		std::cout << "A kivonando vektorok nem azonos meretuek." << std::endl;
	}

	Vector             retV( v.GetCount() );
	for ( unsigned int i = 0; i < _count; i++ ) {
		retV.SetItem( i, _data[i] - v.GetItem( i ) );
	}
	return retV;
}

Vector Vector::multiplyByScalar( double s ) {
	Vector             retV( _count );
	for ( unsigned int i = 0; i < _count; i++ ) {
		retV.SetItem( i, _data[i] * s );
	}
	return retV;
}

Vector Vector::HadamardProd( Vector v ) {
	if (_count != v.GetCount()) {
		std::cout << "A szorzando(hadamard) vektorok nem azonos meretuek." << std::endl;
	}

	Vector             retV( v.GetCount() );
	for ( unsigned int i = 0; i < _count; i++ ) {
		retV.SetItem( i, _data[i] * v.GetItem( i ) );
	}
	return retV;
}
