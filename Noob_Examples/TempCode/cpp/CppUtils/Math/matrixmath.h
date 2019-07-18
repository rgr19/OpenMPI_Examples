//
// Created by robgrzel on 11.06.17.
//

#ifndef UTILS_MATRIXMATH_H
#define UTILS_MATRIXMATH_H

#include <vector>
#include <valarray>

#include "../utils.h"
#include "../Containers/std_utils.h"

namespace UTILS {


    template<class T>
    class FARRAY {
    public:
        FARRAY() {
            p = NULL;
            array(0);
            owner = false;
        }

        FARRAY(T *t) {
            p = t;
            owner = false;
        }

        FARRAY(T &t) {
            p = &t;
            array(0);
            owner = false;
        }

        FARRAY(std::vector<T> &t) {
            p = &t[0];
            array(0);
            owner = false;
        }

        FARRAY(std::valarray<T> &t) {
            p = &t[0];
            array(0);
            owner = false;
        }

        FARRAY(const FARRAY &a) {
            p = (T *) a.p;
            int i;
            for (i = 0; i < 8; ++i) step[i] = a.step[i];
#ifdef _DEBUG
            for (i=0; i<8; ++i) dims[i] = a.dims[i];
#endif
            owner = false;
        }

        const FARRAY &operator=(const FARRAY &a) {
            p = (T *) a.p;
            int i;
            for (i = 0; i < 8; ++i) step[i] = a.step[i];
#ifdef _DEBUG
            for (i=0; i<8; ++i) dims[i] = a.dims[i];
#endif
            owner = false;
            return *this;
        }

        //FARRAY(const FARRAY& a) {p = (T*)(a.p); step1 = a.step1; owner=false;}
        FARRAY(int i, int j) {
            p = NULL;
            allocate(i, j);
        }

        FARRAY(int i, int j, int k) {
            p = NULL;
            allocate(i, j, k);
        }

        FARRAY(int i, int j, int k, int l) {
            p = NULL;
            allocate(i, j, k, l);
        }

        FARRAY(int i, int j, int k, int l, int I) {
            p = NULL;
            allocate(i, j, k, l, I);
        }

        FARRAY(int i, int j, int k, int l, int I, int J) {
            p = NULL;
            allocate(i, j, k, l, I, J);
        }

        FARRAY(int i, int j, int k, int l, int I, int J, int K) {
            p = NULL;
            allocate(i, j, k, l, I, J, K);
        }

        FARRAY(int i, int j, int k, int l, int I, int J, int K, int L) {
            p = NULL;
            allocate(i, j, k, l, I, J, K, L);
        }

        ~FARRAY() {
            deallocate();
        }

        T &operator()(int i) {
#ifdef _DEBUG
            //checkbounds(i);
#endif
            return p[i - 1];
        }

        T &operator()(int i, int j) {
#ifdef _DEBUG
            checkbounds(i,j);
#endif
            return p[(i - 1) + step[0] * (j - 1)];
        }

        T &operator()(int i, int j, int k) {
#ifdef _DEBUG
            checkbounds(i,j,k);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1)];
        }

        T &operator()(int i, int j, int k, int l) {
#ifdef _DEBUG
            checkbounds(i,j,k,l);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1) + step[2] * (l - 1)];
        }

        T &operator()(int i, int j, int k, int l, int I) {
#ifdef _DEBUG
            checkbounds(i,j,k,l,I);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1) + step[2] * (l - 1) + step[3] * (I - 1)];
        }

        T &operator()(int i, int j, int k, int l, int I, int J) {
#ifdef _DEBUG
            checkbounds(i,j,k,k,I,J);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1) + step[2] * (l - 1) + step[3] * (I - 1) +
                     step[4] * (J - 1)];
        }

        T &operator()(int i, int j, int k, int l, int I, int J, int K) {
#ifdef _DEBUG
            checkbounds(i,j,k,l,I,J,K);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1) + step[2] * (l - 1) + step[3] * (I - 1) +
                     step[4] * (J - 1) + step[5] * (K - 1)];
        }

        T &operator()(int i, int j, int k, int l, int I, int J, int K, int L) {
#ifdef _DEBUG
            checkbounds(i,j,k,l,I,J,K,L);
#endif
            return p[(i - 1) + step[0] * (j - 1) + step[1] * (k - 1) + step[2] * (l - 1) + step[3] * (I - 1) +
                     step[4] * (J - 1) + step[5] * (K - 1) + step[6] * (L - 1)];
        }

        T &operator[](int i) {
            return p[i];
        }

#ifdef _DEBUG
        void checkbounds(int i,int j=1,int k=1,int l=1,int I=1,int J=1,int K=1,int L=1) {
            if (i>dims[0] || j>dims[1] || k>dims[2] || l>dims[3] || I>dims[4] || J>dims[5] || K>dims[6] || L>dims[7])
                throw SCATMECH_exception("Array out of bounds in FARRAY");
        }
#endif

#ifdef _DEBUG
        void set_dims(int i,int j=1,int k=1,int l=1,int I=1,int J=1,int K=1,int L=1) {
            dims[0]=i;
            dims[1]=j;
            dims[2]=k;
            dims[3]=l;
            dims[4]=I;
            dims[5]=J;
            dims[6]=K;
            dims[7]=L;
        }
#endif

        void array(int i) {
            step[0] = i;
#ifdef _DEBUG
            set_dims(i);
#endif
        }

        void array(int i, int j) {
            step[1] = j * (step[0] = i);
#ifdef _DEBUG
            set_dims(i,j);
#endif
        }

        void array(int i, int j, int k) {
            step[2] = k * (step[1] = j * (step[0] = i));
#ifdef _DEBUG
            set_dims(i,j,k);
#endif
        }

        void array(int i, int j, int k, int l) {
            step[3] = l * (step[2] = k * (step[1] = j * (step[0] = i)));
#ifdef _DEBUG
            set_dims(i,j,k,l);
#endif
        }

        void array(int i, int j, int k, int l, int I) {
            step[4] = I * (step[3] = l * (step[2] = k * (step[1] = j * (step[0] = i))));
#ifdef _DEBUG
            set_dims(i,j,k,l,I);
#endif
        }

        void array(int i, int j, int k, int l, int I, int J) {
            step[5] = J * (step[4] = I * (step[3] = l * (step[2] = k * (step[1] = j * (step[0] = i)))));
#ifdef _DEBUG
            set_dims(i,j,k,l,I,J);
#endif
        }

        void array(int i, int j, int k, int l, int I, int J, int K) {
            step[6] = K * (step[5] = J * (step[4] = I * (step[3] = l * (step[2] = k * (step[1] = j * (step[0] = i))))));
#ifdef _DEBUG
            set_dims(i,j,k,l,I,J,K);
#endif
        }

        void array(int i, int j, int k, int l, int I, int J, int K, int L) {
            step[7] = L * (step[6] = K * (step[5] = J * (step[4] = I * (step[3] = l * (step[2] = k * (step[1] = j *
                                                                                                                (step[0] = i)))))));
#ifdef _DEBUG
            set_dims(i,j,k,l,I,J,K,L);
#endif
        }

        void allocate(int i) {
            deallocate();
            array(i);
            p = new T[i];
            owner = true;
        }

        void allocate(int i, int j) {
            deallocate();
            unsigned size = i * j;
            array(i, j);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k) {
            deallocate();
            unsigned size = i * j * k;
            array(i, j, k);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k, int l) {
            deallocate();
            unsigned size = i * j * k * l;
            array(i, j, k, l);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k, int l, int I) {
            deallocate();
            unsigned size = i * j * k * l * I;
            array(i, j, k, l, I);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k, int l, int I, int J) {
            deallocate();
            unsigned size = i * j * k * l * I * J;
            array(i, j, k, l, I, J);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k, int l, int I, int J, int K) {
            deallocate();
            unsigned size = i * j * k * l * I * J * K;
            array(i, j, k, l, I, J, K);
            p = new T[size];
            owner = true;
        }

        void allocate(int i, int j, int k, int l, int I, int J, int K, int L) {
            deallocate();
            unsigned size = i * j * k * l * I * J * K * L;
            p = new T[size];
            owner = true;
        }

        void deallocate() {
            if (owner && p != NULL) delete[] p;
            p = 0;
#ifdef _DEBUG
            for (int i=0; i<8; ++i) dims[i] = 0;
#endif
        }

        std::string show() {
            std::ostringstream o;
            o << p << ' ';
            for (int i = 0; i < 8; ++i) o << step[i] << ' ';
            return o.str();
        }

        T &operator=(const T &v) {
            *p = v;
            return *p;
        }

        operator T() const {
            return *p;
        }

    protected:
        bool owner;
        int step[8];
#ifdef _DEBUG
        int dims[8];
#endif
        T *p;
    };

    typedef FARRAY<std::complex<double>> CFARRAY;
    typedef FARRAY<double> DFARRAY;
    typedef FARRAY<int> IFARRAY;


}


#endif //UTILS_MATRIXMATH_H
