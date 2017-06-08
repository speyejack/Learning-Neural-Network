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
	testMatrixTranspose();
	testMatrixClear();
	testMatrixDot();
	printf("PASSED: Matrix Tests\n");
}

void testCompareMatrixEqual(){
	int a[] = {0,1,
			   2,3,
			   4,5};
	int b[] = {5,8,
			   2,2,
			   3,9};
	Matrix aM = generateMatrix(a, 3, 2);
	Matrix bM = generateMatrix(b, 3, 2);
	assertTrue("compareMatrixEqual found same matrix not equal", compareMatrixEqual(aM,aM));
	assertFalse("compareMatrixEqual found different matrices equal", compareMatrixEqual(aM,bM));
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
	int mat[] = {0,1,
				 2,3,
				 4,5};
	Matrix a = generateMatrix(&mat[0], 3, 2);
	Matrix b = generateAMatrix();
	assertTrue("Generated matrices not the same", compareMatrixEqual(a,b));
}

Matrix generateMatrix(int* arrayMatrix, int height, int width){
	Matrix matrix = Matrix(height, width);
	setMatrix(matrix, arrayMatrix, height, width);
	return matrix;
}

void setMatrix(Matrix& matrix, int* arrayMatrix, int height, int width){
	for (int row = 0; row < height; row++){
		setRow(matrix, &arrayMatrix[width * row], row, width);
	}
}

void setRow(Matrix& matrix, int* arrayRow, int row, int width){
	for (int col = 0; col < width; col++){
		matrix.set_value(col, row, arrayRow[col]);
	}
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
	int mat[] = {3, 5, 3,
				 3, 5, 3,
				 3, 5, 3,
				 3, 5, 3};
	
	int trans[] = {3, 3, 3, 3,
				   5, 5, 5, 5,
				   3, 3, 3, 3};

	Matrix a = generateMatrix(mat, 4, 3);
	Matrix b = generateMatrix(trans, 3, 4);
	Matrix transMat = a.transpose();
	assertTrue("Matrix Transposing failed", compareMatrixEqual(transMat,b));
}

void testMatrixClear(){
	int mat[] = {4, 0,
				 9, 2,
				 2, 5};
	
	int clear[] = {0, 0,
				   0, 0,
				   0, 0};

	Matrix matM = generateMatrix(mat, 3, 2);
	Matrix clearM = generateMatrix(mat, 3, 2);
	assertTrue("Matrix clear failed", compareMatrixEqual(matM, clearM));
}

void testMatrixDot(){
	int a[] = {5, 9, 2,
			   1, 4, 2};
	
	int b[] = {8, 5, 1,
			   2, 9, 2,
			   3, 4, 7};

	int result[] = {64, 114, 37,
			   22, 49, 23};
	Matrix aM = generateMatrix(a, 2, 3);
	Matrix bM = generateMatrix(b, 3, 3);
	Matrix resultM = generateMatrix(result, 2, 3);
	Matrix dot = aM.dot(bM);
	assertTrue("Matrix dot product failed", compareMatrixEqual(dot, resultM));
}
