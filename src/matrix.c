#include <vector>
#include <assert.h>
#include "matrix.h"

Matrix::Matrix(int width, int height) : height(height), width(width){
	size = height * width;

	matrix.reserve(size);
}
/*
Matrix::Matrix(Matrix* o){
	this->height = o->height;
	this->width = o->width;
	this->size = o->size;
	this->matrix = o->matrix;
}

inline Matrix Matrix::operator=(Matrix &o){
	return new Matrix(o);
}
*/
inline Matrix Matrix::operator+(Matrix &o){
	Matrix a(*this);
	for (int i = 0; i < size; i++)
		matrix[i] += o.matrix[i];
	return a;
}

inline void Matrix::operator+=(Matrix &o){
	for (int i = 0; i < size; i++)
		matrix[i] += o.matrix[i];
}


inline Matrix Matrix::operator-(Matrix &o){
	Matrix a(*this);
	for (int i = 0; i < size; i++)
		matrix[i] -= o.matrix[i];
	return a;
}

inline void Matrix::operator-=(Matrix &o){
	for (int i = 0; i < size; i++)
		matrix[i] -= o.matrix[i];
}

Matrix Matrix::dot(Matrix &o){
	assert(this->width == o.height);

	Matrix out(o.width, this->height);
	
	for (int row = 0; row < this->height; row++){
		
		for (int col = 0; col < o.width; col++){

			double dot = 0;
			for (int element = 0; element < this->width; element++){

				dot += this->get_value(element, row) * o.get_value(col, element);
				
			}
			printf("Dot: %f\t Col: %d\t Row: %d\n", dot, col, row);
			out.set_value(col, row, dot);
			
		}
			
	}
	return out;
}

