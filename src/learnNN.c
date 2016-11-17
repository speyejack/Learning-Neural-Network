#include <stdio.h>
#include "matrix.h"

int main(){
	printf("Matrix Math Testing!\n");

	Matrix a(2, 3);
	Matrix b(1, 2);

	for (int i = 0; i < 6; i++)
		a.set_value(i % 2, i / 2, i);

	
	for (int i = 0; i < 2; i++)
		b.set_value(i % 1, i / 1, i);
	
	Matrix c;
	
	c = a.dot(b);
	printf("a:\t");
	printMatrix(a);
	printf("b:\t");
	printMatrix(b);
	printf("c (a.b):");
	printMatrix(c);

	Matrix n(1, 3);

	for (int i = 0; i < 3; i++){
		//printf("Row: %d\tCol: %d\n", i % 1, i);
		n.set_value(i % 1, i, 1);
	}

	printf("n:\t");
	printMatrix(n);
	Matrix d;
	d = (c + n);
	printf("c:\t");
	printMatrix(c);
	printf("d (c+n):");
	printMatrix(d);

	c += n;
	printf("c+= n:\t");
	printMatrix(c);
	
	printf("Done!\n");

}
