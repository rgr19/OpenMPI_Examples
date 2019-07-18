//
// Created by robgrzel on 17.05.17.
//

#ifndef STABILITY_AND_DYNAMICS_OF_SHIP_AND_OFFSHORE_STRUCTURES_STD_VECTOR_3D_H
#define STABILITY_AND_DYNAMICS_OF_SHIP_AND_OFFSHORE_STRUCTURES_STD_VECTOR_3D_H

#include <vector>
#include <cstdio>
#include <iostream>

template <typename T> class std_vector_3d {

	private:
		size_t         tmp;
		size_t         M, N, P;
		std::vector<T> vec;

	public:

		std::vector<T> get_vec( ) const {
			return vec;
		}

		inline T& set_elem( size_t i, size_t j, size_t k ) {
			// you could check indexes here e.g.: assert in debug mode
			return vec[i * N * P + j * P + k];
		}

		//x + y * WIDTH + z * WIDTH * DEPTH
		inline const T& get_elem( size_t i, size_t j, size_t k ) {
			// you could check indexes here e.g.: assert in debug mode
			return vec[i * N * P + j * P + k];
		}
		inline T& operator ()( size_t i, size_t j, size_t k ) {// For writing
			return vec[i * N * P + j * P + k];
		}

		inline const T operator ()( size_t i, size_t j, size_t k ) const {// For reading
			return vec[i * N * P + j * P + k];
		}

		std_vector_3d & operator =( std_vector_3d<T>& other ) {
			vec.swap( other.vec );
			return *this;
		}

		std_vector_3d & operator =( const std_vector_3d<T>& other ) {
			vec.swap( std::vector<T>{other.vec} ); // create temporary copy and swap
			return *this;
		}
		std_vector_3d & operator =( std::initializer_list<T> ilist ) {
			vec = std::vector<T>{ilist};
			return *this;
		}

		void set_size( size_t M_ = 0, size_t N_ = 0, size_t P_ = 0 ) {
			M = M_;
			N = N_;
			P = P_;
		}


		void move( size_t id, size_t jd, size_t kd, size_t is, size_t js, size_t ks ) {
			vec[id * N * P + jd * P + kd] = vec[is * N * P + js * P + ks];
		}

		void resize( size_t M_ = 0, size_t N_ = 0, size_t P_ = 0 ) {
			set_size( M_, N_, P_ );
			vec.resize( M_ * N_ * P_ );
		}

		void reset( size_t M_ = 0, size_t N_ = 0, size_t P_ = 0 ) {
			set_size( M_, N_, P_ );
			vec = std::vector<T>( M_ * N_ * P_, 0 );

		}


		void arange( ) {
			for ( size_t i = 0; i < N; i++ ) {
				for ( size_t j = 0; j < M; j++ ) {
					for ( size_t k = 0; k < P; k++ ) {
						vec[k + P * j + P * M * i] = k + P * j + P * M * i;
					}
				}
			}
		}

		void mones(){
			fill(-1);
		}

		void ones(){
			fill(1);
		}

		void zeros(){
			fill(0);
		}

		void fill( T s) {
			std::fill(vec.begin(),vec.end(),s);

		}

		void print( ) {
			std::cout << "Matrix=\n";
			for ( size_t i = 0; i < N; i++ ) {
				std::cout << "[";
				for ( size_t j = 0; j < M; j++ ) {
					std::cout << "[";
					for ( size_t k = 0; k < P; k++ ) {
						std::cout << vec[k + P * j + P * M * i] << ", ";
					}
					std::cout << "], ";
				}
				std::cout << "]\n";
			};
		}

		void append_row( ) {

			N++;

			std::vector<float> row = std::vector<float>( M * P, 0 );
			vec.insert( vec.end(), row.begin(), row.end() );
		}

		void remove_row( ) {


			vec.erase( vec.end() - M * P, vec.end() );
			N--;

		}

		void append_col( ) {

			M++;

			std::vector<float> col = std::vector<float>( P, -1 );


			for ( size_t i = 0, id; i < N; i++ ) {
				id = P * M * i + P * (M - 1);
				vec.insert( vec.begin() + id, col.begin(), col.end() );
			}
		}

		void remove_col( ) {


			for ( size_t i = N, id; i > 0; i-- ) {
				id = P * M * (i - 1) + P * (M - 1);
				vec.erase( vec.begin() + id, vec.begin() + id + P );
			}

			M--;

		}
		void append_plane( ) {

			P++;

			for ( size_t i = 0; i < N * M; i++ ) {
				vec.insert( vec.begin() + P * i + (P - 1), -2 );
			}
		}

		void remove_plane( ) {

			for ( size_t i = N * M; i > 0; i-- ) {
				vec.erase( vec.begin() + P * (i - 1) + (P - 1) );
			}

			P--;
		}

		std_vector_3d( size_t M=1, size_t N=1, size_t P=1) {
			set_size(M,N,P);
			vec = std::vector<T>(M*N*P,0);
		}

};


#endif //STABILITY_AND_DYNAMICS_OF_SHIP_AND_OFFSHORE_STRUCTURES_STD_VECTOR_3D_H
