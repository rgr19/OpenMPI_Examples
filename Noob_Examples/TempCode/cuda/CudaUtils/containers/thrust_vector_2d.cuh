//
// Created by user on 7/23/2018.
//

#ifndef MYMPICUDATEST_THRUST_VECTOR_2D_CUH
#define MYMPICUDATEST_THRUST_VECTOR_2D_CUH


#include "template_utils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>


namespace thrust {
	
	template<class T>
	class host_vector2d : public thrust::host_vector<T> {
	private:
		
		unsigned long W;
		unsigned long H;
		
		unsigned long idx(const int x, const int y) const {
			return y * W + x;
		}
	
	public:
		
		using thrust::host_vector<T>::begin;
		using thrust::host_vector<T>::end;
		using thrust::host_vector<T>::size;
		using thrust::host_vector<T>::resize;
		using thrust::host_vector<T>::data;
		
		using underlying_t = thrust::host_vector<T> ;
		
		unsigned long width() const {
			return W;
		};
		
		unsigned long height() const {
			return H;
		}
		

		T & get(const int x, const int y) const {
			return data()[idx(x,y)];
		}
		
		T const & get(const int x, const int y){
			return data()[idx(x,y)];
		}

		
		unsigned long sizeWH() const {
			return W * H;
		}
		
		unsigned long size_items() const {
			return size() * sizeof(T);
		}
		
		T *buffer() {
			return &this[0];
		}
		
		T const *buffer() const {
			return &this[0];
		}
		
		void print(){
			print("", true);
		}
		
		void ppr(){
			print("THRUST::HOST_VECTOR_2D",false);
		}
		
		void print(const char *name, bool printRaw) {
			
			if (printRaw) {
				for (int i = 0; i < H; i++) {
					for (int j = 0; j < W; j++) {
						std::cout << get(j, i) << '\n';
					}
					printf("\n");
				}
			} else {
				printf("%s = [\n", name);
				for (int i = 0; i < H; i++) {
					printf("[%d] : [ ", i);
					for (int j = 0; j < W; j++) {
						std::cout << "{" << get(j, i) << "}, ";
					}
					printf("]\n");
				}
				printf("]\n");
			}
		}
		
		
		auto slice(int n, int begin = 0, int finish = -1, char axis = 0) {
			
			thrust::host_vector<T> vec;
			
			if (axis) {
				
				if (finish == -1) {
					finish = W;
				}
				
				
				vec.resize(finish - begin);
				for (int i = begin, j = 0; i < finish; i++, j++) {
					vec[j] = get(i, n);
				}
				
			} else {
				
				if (finish == -1) {
					finish = H;
				}
				
				vec.resize(finish - begin);
				for (int i = begin, j = 0; i < finish; i++, j++) {
					vec[j] = get(n, i);
				}
				
			}
			
			return vec;
		}
		
		thrust::host_vector<T> slicex(int n, int finish = -1, int begin = 0) {
			return slice(n, begin, finish, 1);
		}
		
		
		thrust::host_vector<T> slicey(int n, int finish = -1, int begin = 0) {
			return slice(n, begin, finish, 0);
		}
		
		void fromfile(std::string pathFile, char delim = ' ') {
			
			std::fstream fs(pathFile, std::ios::in);
			
			if (!fs) {
				std::cout << "Error opening file\n";
				
				return;
			}
			
			size_t size[2];
			
			size[0] = file_line_count(pathFile.c_str(), delim);
			size[1] = file_lines_count(pathFile.c_str());
			
			resize2d(size[0], size[1]);
			
			T a;
			
			int i = 0;
			
			bool doBreak = false;
			
			while (true) {
				for (int j = 0; j < size[0]; j++) {
					fs >> a;
					if (fs.eof()) {
						doBreak = true;
						break;
					} else {
						get(j, i) = a;
					}
				}
				if (doBreak) {
					break;
				} else {
					i++;
				}
			}
			
			fs.close();
			
			
		}
		
		
		void resize2d(size_t w, size_t h) {
			W = w;
			H = h;
			resize(w * h);
		}
		
		host_vector2d(unsigned long  w = 0, unsigned long  h = 0) : W(w), H(h) {
			
			resize(W*H);
		}
	};
	
	
}

#endif //MYMPICUDATEST_THRUST_VECTOR_2D_H
