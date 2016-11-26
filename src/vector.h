#ifndef VECTOR_HEADER_INCLUDED
#define VECTOR_HEADER_INCLUDED
#include "matrix.h"
#include <vector>
class Vector: public Matrix{
 private:

 public:
	Vector(int size): Matrix(size, 1){}
	Vector(const Matrix& m): Matrix (m){}
	Vector& operator=(const Matrix&);
	Vector(const std::vector<double> init);
	Vector concatenate(Vector& o);
	Vector subset(int start, int end);
	std::vector<double> to_std_vector();
	
};
#endif
