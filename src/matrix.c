#include <vector>
#include <assert.h>
#include "matrix.h"
Matrix::Matrix() : height(0), width(0){
	matrix.resize(height * width);
}
Matrix::Matrix(int width, int height) : height(height), width(width){
	matrix.resize(height * width);
}

Matrix::Matrix(const Matrix& o){
	this->height = o.height;
	this->width = o.width;
	this->matrix = o.matrix;
}

Matrix& Matrix::operator=(const Matrix &o){
	
	this->height = o.height;
	this->width = o.width;
	this->matrix = o.matrix;
	
	return *this;
}

Matrix Matrix::operator+(const Matrix &o){
	Matrix out(width, height);
	for (int i = 0; i < get_size(); i++){
		out.matrix[i] = matrix[i] + o.matrix[i];
	}
	return out;
}

void Matrix::operator+=(const Matrix &o){
	for (int i = 0; i < get_size(); i++)
		matrix[i] += o.matrix[i];
}


Matrix Matrix::operator-(const Matrix &o){
	Matrix out(width, height);
	for (int i = 0; i < get_size(); i++)
		out.matrix[i] = matrix[i] - o.matrix[i];
	return out;
}

void Matrix::operator-=(const Matrix &o){
	for (int i = 0; i < get_size(); i++)
		matrix[i] -= o.matrix[i];
}

Matrix Matrix::dot(Matrix& o){
	assert(this->width == o.height);

	Matrix out(o.width, this->height);
	
	for (int row = 0; row < this->height; row++){
		
		for (int col = 0; col < o.width; col++){

			double dot = 0;
			for (int element = 0; element < this->width; element++){

				dot += this->get_value(element, row) * o.get_value(col, element);
				
			}
			out.set_value(col, row, dot);
			
		}
			
	}
	return out;
}

