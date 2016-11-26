#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "trainer.h"
#include "network.h"

int main(){
	std::vector<int> layers = {2, 1};
	Network n(layers, {1,1});
	Trainer t(&n, 100, 0.1);
	for(int i = 0; i < 100000; i++){
		int count = i % 4;
		
		std::vector<double> input = {(double) (count & 1), (double) ((count >> 1) & 1)};
		std::vector<double> true_output = {(double) (count == 3)};
		t.train(input, true_output);
	}
	
	for(int i = 0; i < 4; i++){
		int count = i % 4;
		
		std::vector<double> input = {(double) (count & 1), (double) ((count >> 1) & 1)};
		std::vector<double> output = n.forward_prop(input);
		std::vector<double> true_output = {(double) (count == 3)};
		printf("Vect: \n");
		Vector input_vec(input);
		Vector output_vec(output);
		Vector true_output_vec(true_output);
		printMatrix(input_vec);
		printMatrix(output_vec);
		printMatrix(true_output_vec);
	}
}
