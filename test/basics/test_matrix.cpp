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
	testMatrixScalarAddition();
	testMatrixMatrixAddition();
	testMatrixMatrixSubtraction();
	testMatrixScalarMultiplication();
	testMatrixMatrixMultiplication();
	testMatrixMatrixInPlaceAddition();
	testMatrixMatrixInPlaceSubtraction();
	testMatrixAssignment();
	printf("PASSED: Matrix Tests\n");
}

void testCompareMatrixEqual(){
	int a[] = {0,1,
			   2,3,
			   4,5};
	int b[] = {5,8,
			   2,2,
			   3,9};
	Matrix aMatrix = generateMatrix(a, 3, 2);
	Matrix bMatrix = generateMatrix(b, 3, 2);
	assertTrue("compareMatrixEqual found same matrix not equal", compareMatrixEqual(aMatrix, aMatrix));
	assertFalse("compareMatrixEqual found different matrices equal", compareMatrixEqual(aMatrix, bMatrix));
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
	int a[] = {4, 0,
				 9, 2,
				 2, 5};
	
	int result[] = {0, 0,
					0, 0,
					0, 0};

	Matrix aMatrix = generateMatrix(a, 3, 2);
	Matrix resultMatrix = generateMatrix(result, 3, 2);

	aMatrix.clear_matrix();
	assertTrue("Matrix clear failed", compareMatrixEqual(aMatrix, resultMatrix));
}

void testMatrixDot(){
	int a[] = {5, 9, 2,
			   1, 4, 2};
	
	int b[] = {8, 5, 1,
			   2, 9, 2,
			   3, 4, 7};

	int result[] = {64, 114, 37,
			   22, 49, 23};
	Matrix aMatrix = generateMatrix(a, 2, 3);
	Matrix bMatrix = generateMatrix(b, 3, 3);
	Matrix resultMatrix = generateMatrix(result, 2, 3);
	Matrix dot = aMatrix.dot(bMatrix);
	assertTrue("Matrix dot product failed", compareMatrixEqual(dot, resultMatrix));
}

void testMatrixScalarAddition(){
	int a[] = {3, 8, 1,
			   9, 5, 0};
	int result[] = {5, 10, 3,
					11, 7, 2};
	Matrix aMatrix = generateMatrix(a, 2, 3);
	Matrix resultMatrix = generateMatrix(result, 2, 3);

	Matrix bMatrix = aMatrix + 2;
	assertTrue("Matrix scalar addition failed", compareMatrixEqual(bMatrix, resultMatrix));
}

void testMatrixScalarMultiplication(){
	int a[] = {3, 8, 1,
			   9, 5, 0};

	int result[] = {6, 16, 2,
					18, 10, 0};
	Matrix aMatrix = generateMatrix(a, 2, 3);
	Matrix resultMatrix = generateMatrix(result, 2, 3);

	Matrix bMatrix = aMatrix * 2;
	assertTrue("Matrix scalar multiplication failed", compareMatrixEqual(bMatrix, resultMatrix));
}
	
void testMatrixMatrixAddition(){
	int a[] = {8, 4,
			   9, 2};
	
	int b[] = {2, 9,
			   9, 1};

	int result[] = {10, 13,
					18, 3};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix bMatrix = generateMatrix(b, 2, 2);
	Matrix resultMatrix = generateMatrix(result, 2, 2);
	
	Matrix cMatrix = aMatrix + bMatrix;
	assertTrue("Matrix matrix addition failed", compareMatrixEqual(cMatrix, resultMatrix));
}
	
void testMatrixMatrixSubtraction(){
	int a[] = {8, 4,
			   9, 2};
	
	int b[] = {2, 9,
			   9, 1};

	int result[] = {6, -5,
					0, 1};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix bMatrix = generateMatrix(b, 2, 2);
	Matrix resultMatrix = generateMatrix(result, 2, 2);
	
	Matrix cMatrix = aMatrix - bMatrix;
	assertTrue("Matrix matrix subtraction failed", compareMatrixEqual(cMatrix, resultMatrix));
}

void testMatrixMatrixMultiplication(){
	int a[] = {8, 4,
			   9, 2};
	
	int b[] = {2, 9,
			   9, 1};

	int result[] = {16, 36,
					81, 2};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix bMatrix = generateMatrix(b, 2, 2);
	Matrix resultMatrix = generateMatrix(result, 2, 2);
	
	Matrix cMatrix = aMatrix * bMatrix;
	assertTrue("Matrix matrix multplication failed", compareMatrixEqual(cMatrix, resultMatrix));
}

void testMatrixMatrixInPlaceAddition(){
	int a[] = {8, 4,
			   9, 2};
	
	int b[] = {2, 9,
			   9, 1};

	int result[] = {10, 13,
					18, 3};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix bMatrix = generateMatrix(b, 2, 2);
	Matrix resultMatrix = generateMatrix(result, 2, 2);
	
	aMatrix += bMatrix;
	assertTrue("Matrix matrix addition failed", compareMatrixEqual(aMatrix, resultMatrix));
}
	
void testMatrixMatrixInPlaceSubtraction(){
	int a[] = {8, 4,
			   9, 2};
	
	int b[] = {2, 9,
			   9, 1};

	int result[] = {6, -5,
					0, 1};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix bMatrix = generateMatrix(b, 2, 2);
	Matrix resultMatrix = generateMatrix(result, 2, 2);
	
	aMatrix -= bMatrix;
	assertTrue("Matrix matrix subtraction failed", compareMatrixEqual(aMatrix, resultMatrix));
}

void testMatrixAssignment(){
	int a[] = {8, 4,
			   9, 2};

	Matrix aMatrix = generateMatrix(a, 2, 2);
	Matrix resultMatrix = aMatrix;
	
	assertTrue("Matrix matrix subtraction failed", compareMatrixEqual(aMatrix, resultMatrix));
}
