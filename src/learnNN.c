#include <stdio.h>
#include "matrix.h"

int main(){
	printf("Neural Network!\n");

	Matrix a(2, 3);
	Matrix b(1, 2);

	for (int i = 0; i < 6; i++)
		a.set_value(i % 2, i / 2, i);

	
	for (int i = 0; i < 2; i++)
		b.set_value(i % 1, i / 1, i);

	Matrix c = a.dot(b);
	printf("a: ");
	printMatrix(a);
	printf("b: ");
	printMatrix(b);
	printf("c: ");
	printMatrix(c);
	printf("Done!\n");

}
