#ifndef MATRIX_HEADER_INCLUDED
#define MATRIX_HEADER_INCLUDED
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <random>
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
	Matrix& Mtanh();
	Matrix& sigmoid();
	void fill_gaussian(std::default_random_engine generator, double mean, double stddev);
	void clear_matrix();
};

static inline void printMatrix(Matrix &o){
	printf("{{");
	for (int i = 0; i < o.get_size(); i++){
		printf("%.1f", o.get_direct_value(i));
		if (i == (o.get_size() - 1))
			printf("}");
	   
		else if ((i + 1) % o.get_width() == 0)
			printf("},{");
		else
			printf(",");
			
	}
	printf("}\n");
	// printf("W: %d H: %d S: %d\n", o.get_width(), o.get_height(), o.get_size());
}
#endif
