#include <vector>
#include <assert.h>
#include "matrix.h"

Matrix::Matrix() : height(0), width(0){
	matrix.resize(height * width);
}

Matrix::Matrix(int height, int width) : height(height), width(width){
	matrix.resize(height * width);
}

Matrix::Matrix(const Matrix& o){
	this->height = o.height;
	this->width = o.width;
	this->matrix = o.matrix;
}

Matrix Matrix::concatenate(Matrix&o){
	Matrix n = *this;
	n.matrix.insert(n.matrix.end(), o.matrix.begin(), o.matrix.end());
	n.height += o.height;
	return n;
}

Matrix Matrix::subset_matrix(int start, int end) {
	Matrix m(end - start, 1);
	for (int i = start; i < end; i++){
		m.set_direct_value(i, get_direct_value(i));
	}
	return m;
}

Matrix& Matrix::operator=(const Matrix &o){
	
	this->height = o.height;
	this->width = o.width;
	this->matrix = o.matrix;
	
	return *this;
}

Matrix Matrix::operator+(const Matrix &o){
	assert(this->get_size() == o.matrix.size());
	Matrix out(height, width);
	for (int i = 0; i < get_size(); i++){
		out.matrix[i] = matrix[i] + o.matrix[i];
	}
	return out;
}

Matrix Matrix::operator*(const Matrix &o){
	assert(this->get_size() == o.matrix.size());
	Matrix out(height, width);
	for (int i = 0; i < get_size(); i++){
		out.matrix[i] = matrix[i] * o.matrix[i];
	}
	return out;
}

Matrix Matrix::operator+(double scalar){
	Matrix out(height, width);
	for (int i = 0; i < get_size(); i++){
		out.matrix[i] = matrix[i] + scalar;
	}
	return out;
}

Matrix Matrix::operator*(double scalar){
	Matrix out(height, width);
	for (int i = 0; i < get_size(); i++){
		out.matrix[i] = matrix[i] * scalar; 
	}
	return out;
}

void Matrix::operator+=(const Matrix &o){
	assert(this->get_size() == o.matrix.size());
	for (int i = 0; i < get_size(); i++)
		matrix[i] += o.matrix[i];
}


Matrix Matrix::operator-(const Matrix &o){
	assert(this->get_size() == o.matrix.size());
	Matrix out(width, height);
	for (int i = 0; i < get_size(); i++)
		out.matrix[i] = matrix[i] - o.matrix[i];
	return out;
}

void Matrix::operator-=(const Matrix &o){
	assert(this->get_size() == o.matrix.size());
	for (int i = 0; i < get_size(); i++)
		matrix[i] -= o.matrix[i];
}

Matrix Matrix::dot(Matrix o){
	assert(this->width == o.height);

	Matrix out(this->height, o.width);
	
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


Matrix Matrix::transpose(){
	Matrix out(width, height);
	for (int row = 0; row < get_height(); row++){
		for (int col = 0; col < get_width(); col++){
			double val = get_value(col, row);
			out.set_value(row, col, val);
		}
	}
	return out;
}


Matrix& Matrix::Mtanh(){
	for (int i = 0; i < get_size(); i++){
		double val = get_direct_value(i);
		val = tanh(val);
		set_direct_value(i,val);
	}
	return *this;
}


Matrix& Matrix::sigmoid(){
	for (int i = 0; i < get_size(); i++){
		double val = get_direct_value(i);
		// Fast sigmoid function, if having trouble chance to realistic
		val = 1 / (1 + exp(-val));
		set_direct_value(i,val);
	}
	return *this;
}


void Matrix::fill_gaussian(std::default_random_engine generator, double mean, double stddev){
	std::normal_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < get_size(); i++){
		set_direct_value(i, dist(generator));
	}
}

void Matrix::clear_matrix(){
	for (int i = 0; i < get_size(); i++){
		set_direct_value(i, 0);
	}
}

void Matrix::write_to_json(std::ostream& os){
	os << "{" << std::endl;
	os << "\"height\" : " << height << std::endl;
	os << "\"width\" : " << width << std::endl;
	os << "\"matrix\" : {";
	for (unsigned int i = 0; i < matrix.size() - 1; i++){
		os << matrix[i] << ", ";
	}
	os << matrix[matrix.size() - 1] << "}" << std::endl;
	os << "}";
}

std::ostream& operator<<(std::ostream& os, Matrix& matrix){
	matrix.write_to_json(os);
	return os;
}
