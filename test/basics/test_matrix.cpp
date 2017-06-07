#include <iostream>
#include "../../src/matrix.h"
#include "../../test/common/asserts.h"
#include "test_matrix.h"

int main(){
	runAllMatrixTests();
}

void runAllMatrixTests(){
	testCompareMatrixEqual();
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

