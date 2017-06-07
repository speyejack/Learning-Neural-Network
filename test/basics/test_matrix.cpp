#include <iostream>
#include "../../src/matrix.h"
#include "../../test/common/asserts.h"
#include "test_matrix.h"

int main(){
	runAllMatrixTests();
}

void runAllMatrixTests(){
	testCompareMatrixEqual();
	testMatrixGetHeight();
	testMatrixGetWidth();
	testMatrixGetValue();
	testMatrixSetValue();
	testMatrixGeneration();
	printf("PASSED: Matrix Tests\n");
}

void testCompareMatrixEqual(){
	Matrix a = generateAMatrix();
	Matrix b = generateBMatrix();
	assertTrue("compareMatrixEqual found same matrix not equal", compareMatrixEqual(a,a));
	assertFalse("compareMatrixEqual found different matrices equal", compareMatrixEqual(a,b));
}

Matrix generateAMatrix(){
	Matrix a(3,2);
	int counter = 0;
	for (int i = 0; i < a.get_height(); i++){
		for (int j = 0; j < a.get_width(); j++){
			a.set_value(j, i, counter++);
		}
	}
	return a;
}

void testMatrixGeneration(){
	/*Matrix a = generateMatrix([[1,2],
							   [3,4],
							   [5,6]],
							  3, 2);
	Matrix b = generateAMatrix();
	assertTrue("Generated matrices not the same", compareMatrixEqual(a,b));
	*/
}

Matrix generateMatrix(int** arrayMatrix, int height, int width){
	Matrix matrix = Matrix(height, width);
	setMatrix(matrix, arrayMatrix, height, width);
	return matrix;
}

void setMatrix(Matrix& matrix, int** arrayMatrix, int height, int width){
	for (int row = 0; row < height; row++){
		setRow(matrix, arrayMatrix[row], row, width);
	}
}

void setRow(Matrix& matrix, int* arrayRow, int row, int width){
	for (int col = 0; col < width; col++){
		matrix.set_value(col, row, arrayRow[col]);
	}
}

Matrix generateBMatrix(){
	Matrix b(1,3);
	int counter = 0;
	for (int i = 0; i < b.get_height(); i++){
		for (int j = 0; j < b.get_width(); j++){
			b.set_value(j, i, counter++);
		}
	}
	return b;
}

bool compareMatrixEqual(Matrix& a, Matrix& b){
	return compareMatrixSizesEqual(a, b) && compareMatrixValuesEqual(a, b);
}

bool compareMatrixSizesEqual(Matrix& a, Matrix& b){
	return a.get_height() == b.get_height() &&  a.get_width() == b.get_width();
}

bool compareMatrixValuesEqual(Matrix& a, Matrix& b){
	for (int y = 0; y < a.get_height(); y++){
		for (int x = 0; x < a.get_width(); x++){
			if (a.get_value(x,y) != b.get_value(x,y)){
				return false;
			}
		}
	}
	return true;
}

void testMatrixGetHeight(){
	Matrix a = Matrix(5,2);
	assertTrue("Matrix Get Height didn't return expect value of 5", a.get_height() == 5);
}

void testMatrixGetWidth(){
	Matrix a = Matrix(5,2);
	assertTrue("Matrix Get Height didn't return expect value of 2", a.get_width() == 2);
}

void testMatrixGetValue(){
	Matrix a = Matrix(1,1);
	assertTrue("Matrix Get Value didn't get value of 0 back", a.get_value(0,0) == 0);
}

void testMatrixSetValue(){
	Matrix a = Matrix(1,1);
	int new_value = 7;
	a.set_value(0,0,new_value);
	assertTrue("Matrix Set Value didn't get value of 7 back", a.get_value(0,0) == new_value);
}

void testMatrixTranspose(){
	
}
