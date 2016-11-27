#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "trainer.h"
#include "network.h"
#include <cstdlib>
#include <ctime>


int main(){
	std::vector<int> layers = {2, 1};
	Network n(layers, {1,1});
	Trainer t(&n, 10, 0.1);
	srand((unsigned) time(NULL));
	int q = 0;
	for(int i = 0; i < 100000; i++){
		int count = rand() % 4;
		
		std::vector<double> input = {(double) (count & 1), (double) ((count >> 1) & 1)};
		switch( count){
		case 0:
			break;
		case 1:
			q = 1;
			break;
		case 2:
			q = 0;
			break;
		case 3:
			q != q;
			break;
		}
		std::vector<double> true_output = {(double) (q)};
		t.train(input, true_output);
		if (!(i % 10))
			q = 0;
	}
	q = 0;
	for(int i = 0; i < 4; i++){
		int count = i % 4;
		
		std::vector<double> input = {(double) (count & 1), (double) ((count >> 1) & 1)};
		std::vector<double> output = n.forward_prop(input);
		printf("Vect: \n");
		Vector input_vec(input);
		Vector output_vec(output);
		printMatrix(input_vec);
		printMatrix(output_vec);
	}
}
