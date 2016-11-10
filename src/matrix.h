#ifndef MATRIX_HEADER_INCLUDED
#define MATRIX_HEADER_INCLUDED
#include <vector>
#include <stdio.h>

class Matrix {
 private:
	//	explicit Matrix(Matrix*);

	std::vector<double> matrix;
	
	int size;
	int height;
	int width;
	inline void set_direct_value(int i, double value) {matrix[i] = value;}
 public:
	explicit Matrix(int height, int width);

	inline int get_width() {return width;}
	inline int get_height() {return height;}
	inline int get_size() {return size;}
	inline double get_direct_value(int i) {return matrix[i];}
	inline void set_value(int x, int y, double value) {set_direct_value(width * y + x, value);} 
	inline double get_value(int x, int y) {return get_direct_value(width * y + x);}
	inline void operator+=(Matrix &o);
	inline void operator-=(Matrix &o);
	inline Matrix operator+(Matrix &o);
	inline Matrix operator-(Matrix &o);
	//inline Matrix operator=(Matrix &o);
	Matrix dot(Matrix &o);
};

static inline void printMatrix(Matrix &o){
	printf("W: %d H: %d\n", o.get_width(), o.get_height());
	printf("{{");
	for (int i = 0; i < o.get_size(); i++){
		printf("%f", o.get_direct_value(i));
		if (i == (o.get_size() - 1))
			printf("}");
		else if ((i + 1) % o.get_width() == 0)
			printf("},{");
		else
			printf(",");
			
	}
	printf("}\n");
		
}
#endif
