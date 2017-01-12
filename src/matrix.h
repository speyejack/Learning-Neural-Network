#ifndef MATRIX_HEADER_INCLUDED
#define MATRIX_HEADER_INCLUDED
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <random>
#include <ostream>
class Matrix {
 private:
	std::vector<double> matrix;
	
	int height;
	int width;

 protected:
	inline void set_direct_value(int i, double value) {matrix[i] = value;}
	Matrix concatenate(Matrix& o);
	Matrix subset_matrix(int start, int end);
	
 public:
	
	Matrix();
	Matrix(int height, int width);
	Matrix(const Matrix&);
	inline int get_width() {return width;}
	inline int get_height() {return height;}
	inline int get_size() {return matrix.size();}
	inline double get_direct_value(int i) {return matrix[i];}
	inline void set_value(int x, int y, double value) {assert (y < height && x < width); set_direct_value(width * y + x, value);} 
	inline double get_value(int x, int y) {assert (y < height && x < width); return get_direct_value(width * y + x);}
	void operator+=(const Matrix &o);
	void operator-=(const Matrix &o);
	Matrix operator+(const Matrix &o);
	Matrix operator-(const Matrix &o);
	Matrix operator*(const Matrix &o);
	Matrix operator+(double scalar);
	Matrix operator*(double scalar);
    Matrix& operator=(const Matrix &o);
	Matrix dot(Matrix o);
	Matrix transpose();
	Matrix sigmoid();
	Matrix sigDeriv();
	Matrix Mtanh();
	Matrix MtanhDeriv();
	void fill_gaussian(std::default_random_engine& generator, double mean, double stddev);
	void clear_matrix();
	void write_to_json(std::ostream&);
};

std::ostream& operator<<(std::ostream&, Matrix&);
#endif
