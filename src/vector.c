#include "vector.h"
#include <vector>

Vector::Vector(const std::vector<double> init): Matrix(init.size(), 1){
	for (int i = 0; i < init.size(); i++){
		set_direct_value(i, init[i]);
	}
}

Vector& Vector::operator=(const Matrix& m){
	Matrix::operator=(m);
	return *this;
}

Vector Vector::concatenate(Vector& o){
	return Matrix::concatenate(o);
}


Vector Vector::subset(int start, int end){
	return  Matrix::subset_matrix(start, end);
}

std::vector<double> Vector::to_std_vector(){
	std::vector<double> vec;
	vec.resize(get_size());
	for(int i = 0; i < get_size(); i++){
		vec[i] = get_direct_value(i);
	}
	return vec;
}
