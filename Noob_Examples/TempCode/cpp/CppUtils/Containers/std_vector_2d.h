//
// Created by user on 7/23/2018.
//

#ifndef MYMPICUDATEST_THRUST_VECTOR_2D_CUH
#define MYMPICUDATEST_THRUST_VECTOR_2D_CUH

#include <vector>
#include "template_utils.h"


namespace std {
	
	template<class T>
	class vector2d : public std::vector<T> {
	private:
		
		unsigned long W;
		unsigned long H;
		

	
	public:
		
		using std::vector<T>::begin;
		using std::vector<T>::end;
		using std::vector<T>::size;
		using std::vector<T>::resize;
		using std::vector<T>::data;
		using std::vector<T>::at;
		using std::vector<T>::operator[];
		
		using underlying_t = std::vector<T> ;
		
		inline unsigned long idx(const int x, const int y) const {
			return y * W + x;
		}
		
		inline unsigned long width() const {
			return W;
		};
		
		inline unsigned long height() const {
			return H;
		}
		
		
		inline T & get(const int x, const int y) const {
			return operator[](idx(x,y));
		}
		
		inline T & get(const int x, const int y){
			return operator[](idx(x,y));
		}
		
		inline void set(const int x, const int y, int val){
			data()[idx(x,y)] = val;
		}
		
		inline void set(const int x, const int y, T val){
			data()[idx(x,y)] = val;
		}
		
		
		inline unsigned long sizeWH() const {
			return W * H;
		}
		
		inline unsigned long size_items() const {
			return size() * sizeof(T);
		}
		
		inline T *buffer() {
			return &this[0];
		}
		
		inline T const *buffer() const {
			return &this[0];
		}
		
		inline void print(){
			print("", true);
		}
		
		inline void ppr(){
			print("THRUST::HOST_VECTOR_2D",false);
		}
		
		inline void print(const char *name, bool printRaw) {
			
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
		
		
		inline auto slice(int n, int begin = 0, int finish = -1, char axis = 0) {
			
			std::vector<T> vec;
			
			if (axis) {
				
				if (finish == -1) {
					finish = static_cast<int>(W);
				}
				
				
				vec.resize(static_cast<unsigned long>(finish - begin));
				for (int i = begin, j = 0; i < finish; i++, j++) {
					vec[j] = get(i, n);
				}
				
			} else {
				
				if (finish == -1) {
					finish = static_cast<int>(H);
				}
				
				vec.resize(static_cast<unsigned long>(finish - begin));
				for (int i = begin, j = 0; i < finish; i++, j++) {
					vec[j] = get(n, i);
				}
				
			}
			
			return vec;
		}
		
		inline	std::vector<T> slicex(int n, int finish = -1, int begin = 0) {
			return slice(n, begin, finish, 1);
		}
		
		
		inline std::vector<T> slicey(int n, int finish = -1, int begin = 0) {
			return slice(n, begin, finish, 0);
		}
		
		inline void fromfile(std::string pathFile, char delim = ' ') {
			
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
		
		
		inline void resize2d(size_t w, size_t h) {
			W = w;
			H = h;
			resize(w * h);
		}
		
		inline vector2d(unsigned long  w = 0, unsigned long  h = 0) : W(w), H(h) {
			
			resize(W*H);
		}
		

		

		
	};
	
	
}

#endif //MYMPICUDATEST_THRUST_VECTOR_2D_H
